#!/usr/bin/env python3
"""MMH DICOM tri-view fusion (Axial/Coronal/Sagittal PD-FS).
Reuses the original 2026-09 GPU grid_sample + intensity fusion core.
DICOM decoding and geometry are handled with SimpleITK/GDCM.
No TotalSegmentator or ROI stage.
"""
from __future__ import annotations
import argparse, gc, inspect, json, shutil, time, traceback, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
import SimpleITK as sitk
DEFAULT_SEQUENCE = "Fluid_Sensitive_1_Fat_Suppression_1"

def get_spacing(img: nib.Nifti1Image):
    return tuple(
        float(x)
        for x in img.header.get_zooms()[:3]
    )


def json_dump(data, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_nifti(
    data,
    affine,
    path: Path,
    dtype,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    img = nib.Nifti1Image(
        np.asarray(
            data,
            dtype=dtype,
        ),
        affine,
    )

    img.set_data_dtype(dtype)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)

    nib.save(
        img,
        str(path),
    )


def save_like_reference(
    data,
    reference: nib.Nifti1Image,
    path: Path,
    dtype=np.uint8,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference.header.copy()
    header.set_data_dtype(dtype)

    img = nib.Nifti1Image(
        np.asarray(
            data,
            dtype=dtype,
        ),
        reference.affine.copy(),
        header=header,
    )

    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)

    if qform is not None:
        img.set_qform(
            qform,
            int(qcode),
        )

    if sform is not None:
        img.set_sform(
            sform,
            int(scode),
        )

    nib.save(
        img,
        str(path),
    )


def nifti_info(path: Path) -> dict:
    image = nib.load(
        str(path)
    )

    if len(image.shape) < 3:
        raise ValueError(
            f"Not a 3D NIfTI: {path}"
        )

    shape = tuple(
        int(v)
        for v in image.shape[:3]
    )

    spacing = tuple(
        float(v)
        for v in image.header.get_zooms()[:3]
    )

    slice_axis = int(
        np.argmax(spacing)
    )

    return {
        "path":
            str(
                path.resolve()
            ),

        "shape":
            shape,

        "spacing":
            spacing,

        "slice_axis":
            slice_axis,

        "slice_count":
            int(
                shape[
                    slice_axis
                ]
            ),

        "through_plane_spacing":
            float(
                spacing[
                    slice_axis
                ]
            ),
    }


def choose_best_series(
    series_dir: Path,
) -> dict:
    if not series_dir.is_dir():
        raise FileNotFoundError(
            f"Missing series directory:\n{series_dir}"
        )

    candidates = []

    for path in sorted(
        series_dir.glob("*.nii.gz")
    ):
        try:
            info = nifti_info(path)

            if info[
                "slice_count"
            ] <= 1:
                continue

            candidates.append(info)

        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"No usable NIfTI series in:\n{series_dir}"
        )

    candidates.sort(
        key=lambda item: (
            -item["slice_count"],
            item["through_plane_spacing"],
            Path(
                item["path"]
            ).name,
        )
    )

    return candidates[0]


def path_contains_ignored_folder(
    path: Path,
    ignored_name: str = "niftymic_output",
) -> bool:
    """
    Treat every folder named niftymic_output as nonexistent.
    Matching is case-insensitive.
    """
    ignored = ignored_name.lower()

    return any(
        str(part).lower() == ignored
        for part in path.parts
    )


def find_case_view_dir(
    case_dir: Path,
    requested_name: str,
) -> Path | None:
    """
    Resolve Axial / Coronal / Sagittal case-insensitively.
    Never traverse niftymic_output.
    """
    requested = requested_name.lower()

    for child in case_dir.iterdir():
        if not child.is_dir():
            continue

        if child.name.lower() == "niftymic_output":
            continue

        if child.name.lower() == requested:
            return child

    return None


def choose_preferred_series(
    view_dir: Path,
    preferred_sequence: str,
) -> dict:
    """
    Selection policy for one view:

    1) Prefer <view>/Fluid_Sensitive_1_Fat_Suppression_1 (or supplied preferred).
    2) Only if unavailable/unusable, search other NIfTI series in the view.
    3) Ignore every path containing a folder named niftymic_output.
    """

    if not view_dir.is_dir():
        raise FileNotFoundError(
            f"View directory not found:\n{view_dir}"
        )

    preferred_dir = None

    for child in view_dir.iterdir():
        if not child.is_dir():
            continue

        if child.name.lower() == "niftymic_output":
            continue

        if child.name.lower() == preferred_sequence.lower():
            preferred_dir = child
            break

    if preferred_dir is not None:
        try:
            info = choose_best_series(
                preferred_dir
            )

            info = dict(info)
            info["selection_mode"] = "preferred_sequence"
            info["sequence_folder"] = preferred_dir.name

            return info

        except Exception as exc:
            print(
                f"  Preferred sequence unusable: {preferred_dir}"
            )
            print(
                f"  Falling back to other sequences. Reason: {exc}"
            )

    candidates = []

    for path in sorted(
        view_dir.rglob("*.nii.gz")
    ):
        try:
            relative = path.relative_to(
                view_dir
            )

            if path_contains_ignored_folder(
                relative
            ):
                continue

            info = nifti_info(
                path
            )

            if info["slice_count"] <= 1:
                continue

            info = dict(info)
            info["selection_mode"] = "fallback"
            info["sequence_folder"] = (
                path.parent.name
                if path.parent != view_dir
                else "<directly_in_view_folder>"
            )

            candidates.append(
                info
            )

        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            "No usable NIfTI series found after excluding niftymic_output:\n"
            f"{view_dir}"
        )

    candidates.sort(
        key=lambda item: (
            -item["slice_count"],
            item["through_plane_spacing"],
            Path(
                item["path"]
            ).name,
        )
    )

    return candidates[0]


def robust_statistics(
    data: np.ndarray,
):
    data = np.asarray(
        data,
        dtype=np.float32,
    )

    finite = np.isfinite(data)

    foreground = (
        finite
        & (np.abs(data) > 1e-6)
    )

    values = data[
        foreground
    ]

    if values.size < 1000:
        values = data[
            finite
        ]

    if values.size == 0:
        raise RuntimeError(
            "No finite MRI voxels found."
        )

    if values.size > 2_000_000:
        indices = np.linspace(
            0,
            values.size - 1,
            2_000_000,
        ).astype(np.int64)

        values = values[
            indices
        ]

    p25, p50, p75 = np.percentile(
        values,
        [
            25,
            50,
            75,
        ],
    )

    iqr = float(
        p75 - p25
    )

    robust_scale = (
        iqr
        / 1.349
    )

    if robust_scale < 1e-6:
        robust_scale = float(
            np.std(values)
        )

    if robust_scale < 1e-6:
        robust_scale = 1.0

    return (
        float(p50),
        float(robust_scale),
    )


def harmonize_array(
    data,
    source_median,
    source_scale,
    target_median,
    target_scale,
):
    data = np.asarray(
        data,
        dtype=np.float32,
    )

    foreground = (
        np.isfinite(data)
        & (np.abs(data) > 1e-6)
    )

    out = np.zeros_like(
        data,
        dtype=np.float32,
    )

    out[
        foreground
    ] = (
        (
            data[
                foreground
            ]
            - float(
                source_median
            )
        )
        / float(
            source_scale
        )
        * float(
            target_scale
        )
        + float(
            target_median
        )
    )

    return out


def image_world_bounds(
    image,
):
    shape = np.asarray(
        image.shape[:3],
        dtype=np.float64,
    )

    corners = np.array(
        [
            [
                x,
                y,
                z,
            ]
            for x in (
                -0.5,
                shape[0] - 0.5,
            )
            for y in (
                -0.5,
                shape[1] - 0.5,
            )
            for z in (
                -0.5,
                shape[2] - 0.5,
            )
        ],
        dtype=np.float64,
    )

    world = nib.affines.apply_affine(
        image.affine,
        corners,
    )

    return (
        world.min(
            axis=0
        ),
        world.max(
            axis=0
        ),
    )


def build_union_reference(
    images,
    spacing: float,
):
    lows = []
    highs = []

    for image in images:
        low, high = image_world_bounds(
            image
        )
        lows.append(low)
        highs.append(high)

    world_min = np.min(
        np.stack(
            lows,
            axis=0,
        ),
        axis=0,
    )

    world_max = np.max(
        np.stack(
            highs,
            axis=0,
        ),
        axis=0,
    )

    spacing = float(spacing)

    shape = (
        np.ceil(
            (
                world_max
                - world_min
            )
            / spacing
        )
        .astype(np.int64)
        + 1
    )

    affine = np.eye(
        4,
        dtype=np.float64,
    )

    affine[0, 0] = spacing
    affine[1, 1] = spacing
    affine[2, 2] = spacing
    affine[:3, 3] = world_min

    return (
        tuple(
            int(v)
            for v in shape
        ),
        affine,
    )


def _voxel_to_normalized(
    coord: torch.Tensor,
    size: int,
):
    if size <= 1:
        return torch.zeros_like(
            coord
        )

    return (
        2.0
        * coord
        / float(
            size - 1
        )
        - 1.0
    )


def gpu_resample_volume_chunked(
    source_xyz,
    source_affine,
    reference_shape_xyz,
    reference_affine,
    device,
    chunk_depth,
):
    sx, sy, sz = (
        int(
            source_xyz.shape[0]
        ),
        int(
            source_xyz.shape[1]
        ),
        int(
            source_xyz.shape[2]
        ),
    )

    ox, oy, oz = (
        int(
            reference_shape_xyz[0]
        ),
        int(
            reference_shape_xyz[1]
        ),
        int(
            reference_shape_xyz[2]
        ),
    )

    source_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                source_xyz.transpose(
                    2,
                    1,
                    0,
                )
            )
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    valid_tensor = torch.ones(
        (
            1,
            1,
            sz,
            sy,
            sx,
        ),
        dtype=torch.float32,
        device=device,
    )

    target_to_source = (
        np.linalg.inv(
            source_affine
        )
        @ reference_affine
    )

    transform = torch.as_tensor(
        target_to_source,
        dtype=torch.float32,
        device=device,
    )

    x_idx = torch.arange(
        ox,
        dtype=torch.float32,
        device=device,
    )

    y_idx = torch.arange(
        oy,
        dtype=torch.float32,
        device=device,
    )

    out_xyz = np.zeros(
        (
            ox,
            oy,
            oz,
        ),
        dtype=np.float32,
    )

    out_valid_xyz = np.zeros(
        (
            ox,
            oy,
            oz,
        ),
        dtype=np.uint8,
    )

    for z0 in range(
        0,
        oz,
        int(chunk_depth),
    ):
        z1 = min(
            z0 + int(
                chunk_depth
            ),
            oz,
        )

        z_idx = torch.arange(
            z0,
            z1,
            dtype=torch.float32,
            device=device,
        )

        zz, yy, xx = torch.meshgrid(
            z_idx,
            y_idx,
            x_idx,
            indexing="ij",
        )

        src_x = (
            transform[0, 0] * xx
            + transform[0, 1] * yy
            + transform[0, 2] * zz
            + transform[0, 3]
        )

        src_y = (
            transform[1, 0] * xx
            + transform[1, 1] * yy
            + transform[1, 2] * zz
            + transform[1, 3]
        )

        src_z = (
            transform[2, 0] * xx
            + transform[2, 1] * yy
            + transform[2, 2] * zz
            + transform[2, 3]
        )

        grid = torch.stack(
            [
                _voxel_to_normalized(
                    src_x,
                    sx,
                ),
                _voxel_to_normalized(
                    src_y,
                    sy,
                ),
                _voxel_to_normalized(
                    src_z,
                    sz,
                ),
            ],
            dim=-1,
        ).unsqueeze(0)

        sampled = F.grid_sample(
            source_tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        sampled_valid = F.grid_sample(
            valid_tensor,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )

        chunk_xyz = (
            sampled[
                0,
                0,
            ]
            .permute(
                2,
                1,
                0,
            )
            .contiguous()
            .cpu()
            .numpy()
        )

        valid_chunk_xyz = (
            sampled_valid[
                0,
                0,
            ]
            .permute(
                2,
                1,
                0,
            )
            .contiguous()
            .cpu()
            .numpy()
            > 0.5
        )

        out_xyz[
            :,
            :,
            z0:z1,
        ] = chunk_xyz

        out_valid_xyz[
            :,
            :,
            z0:z1,
        ] = valid_chunk_xyz.astype(
            np.uint8
        )

        del (
            zz,
            yy,
            xx,
            src_x,
            src_y,
            src_z,
            grid,
            sampled,
            sampled_valid,
            chunk_xyz,
            valid_chunk_xyz,
        )

    del source_tensor
    del valid_tensor

    return (
        out_xyz,
        out_valid_xyz,
    )


def reconstruct_triview(
    root: Path,
    case_id: str,
    sequence: str,
    spacing_mm: float,
    chunk_depth: int,
    torch_device: str,
    harmonize: bool,
    output_path: Path,
):
    if (
        torch_device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    device = torch.device(
        torch_device
    )

    case_dir = (
        root
        / case_id
    )

    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"Case directory not found:\n{case_dir}"
        )

    selected = {}

    for view_name, folder_name in (
        (
            "axial",
            "Axial",
        ),
        (
            "coronal",
            "Coronal",
        ),
        (
            "sagittal",
            "Sagittal",
        ),
    ):
        view_dir = find_case_view_dir(
            case_dir,
            folder_name,
        )

        if view_dir is None:
            raise FileNotFoundError(
                f"Missing {folder_name} view in case:\n{case_dir}"
            )

        selected[
            view_name
        ] = choose_preferred_series(
            view_dir,
            preferred_sequence=sequence,
        )

    print("\n" + "=" * 96)
    print("STEP 1 — TRI-VIEW 3D MRI FUSION")
    print("=" * 96)

    print(
        f"Case     : {case_id}"
    )
    print(
        f"Preferred sequence : {sequence}"
    )
    print(
        f"Spacing  : {spacing_mm:.3f} mm isotropic"
    )
    print(
        f"Device   : {device}"
    )

    if device.type == "cuda":
        print(
            "GPU      :",
            torch.cuda.get_device_name(
                device
            ),
        )

    canonical = {}
    source_arrays = {}
    source_statistics = {}

    for view_name in (
        "axial",
        "coronal",
        "sagittal",
    ):
        info = selected[
            view_name
        ]

        print(
            f"\n[{view_name.upper()}]"
        )
        print(
            f"  {info['path']}"
        )
        print(
            f"  shape   = {info['shape']}"
        )
        print(
            f"  spacing = {info['spacing']}"
        )
        print(
            f"  sequence= {info.get('sequence_folder', '?')}"
        )
        print(
            f"  mode    = {info.get('selection_mode', '?')}"
        )

        source = nib.load(
            info["path"]
        )

        image = nib.as_closest_canonical(
            source
        )

        canonical[
            view_name
        ] = image

        data = np.asarray(
            image.dataobj,
            dtype=np.float32,
        )

        median, scale = robust_statistics(
            data
        )

        source_statistics[
            view_name
        ] = {
            "median":
                median,

            "scale":
                scale,
        }

        source_arrays[
            view_name
        ] = data

    reference_median = None
    reference_scale = None

    if harmonize:
        reference_median = float(
            np.median(
                [
                    source_statistics[
                        name
                    ][
                        "median"
                    ]
                    for name in (
                        "axial",
                        "coronal",
                        "sagittal",
                    )
                ]
            )
        )

        reference_scale = float(
            np.median(
                [
                    source_statistics[
                        name
                    ][
                        "scale"
                    ]
                    for name in (
                        "axial",
                        "coronal",
                        "sagittal",
                    )
                ]
            )
        )

        for view_name in (
            "axial",
            "coronal",
            "sagittal",
        ):
            source_arrays[
                view_name
            ] = harmonize_array(
                source_arrays[
                    view_name
                ],
                source_median=(
                    source_statistics[
                        view_name
                    ][
                        "median"
                    ]
                ),
                source_scale=(
                    source_statistics[
                        view_name
                    ][
                        "scale"
                    ]
                ),
                target_median=(
                    reference_median
                ),
                target_scale=(
                    reference_scale
                ),
            )

    reference_shape, reference_affine = build_union_reference(
        list(
            canonical.values()
        ),
        spacing=spacing_mm,
    )

    print(
        "\nReference shape:",
        reference_shape,
    )

    numerator = np.zeros(
        reference_shape,
        dtype=np.float32,
    )

    denominator = np.zeros(
        reference_shape,
        dtype=np.float32,
    )

    coverage_summary = {}

    start_total = time.perf_counter()

    for view_name in (
        "axial",
        "coronal",
        "sagittal",
    ):
        print(
            f"GPU resampling {view_name}..."
        )

        start = time.perf_counter()

        resampled, valid = gpu_resample_volume_chunked(
            source_xyz=(
                source_arrays[
                    view_name
                ]
            ),
            source_affine=(
                canonical[
                    view_name
                ].affine
            ),
            reference_shape_xyz=(
                reference_shape
            ),
            reference_affine=(
                reference_affine
            ),
            device=device,
            chunk_depth=chunk_depth,
        )

        valid_bool = (
            valid > 0
        )

        numerator[
            valid_bool
        ] += resampled[
            valid_bool
        ]

        denominator[
            valid_bool
        ] += 1.0

        elapsed = (
            time.perf_counter()
            - start
        )

        coverage_summary[
            view_name
        ] = {
            "valid_voxels":
                int(
                    valid_bool.sum()
                ),

            "coverage_fraction":
                float(
                    valid_bool.mean()
                ),

            "gpu_resample_seconds":
                float(
                    elapsed
                ),
        }

        print(
            f"  done in {elapsed:.2f} s"
        )

        del (
            resampled,
            valid,
            valid_bool,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline = np.zeros(
        reference_shape,
        dtype=np.float32,
    )

    union = (
        denominator > 0
    )

    baseline[
        union
    ] = (
        numerator[
            union
        ]
        / denominator[
            union
        ]
    )

    total_elapsed = (
        time.perf_counter()
        - start_total
    )

    save_nifti(
        baseline,
        reference_affine,
        output_path,
        np.float32,
    )

    print(
        f"\nFused MRI saved:\n{output_path}"
    )

    metadata = {
        "case_id":
            case_id,

        "sequence":
            sequence,

        "spacing_mm":
            float(
                spacing_mm
            ),

        "reference_shape":
            [
                int(x)
                for x in reference_shape
            ],

        "reference_affine":
            reference_affine.tolist(),

        "selected_sources":
            selected,

        "source_statistics":
            source_statistics,

        "intensity_harmonization":
            bool(
                harmonize
            ),

        "harmonization_reference_median":
            reference_median,

        "harmonization_reference_scale":
            reference_scale,

        "coverage":
            coverage_summary,

        "fusion_seconds":
            float(
                total_elapsed
            ),
    }

    del (
        baseline,
        numerator,
        denominator,
        union,
        source_arrays,
        canonical,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metadata



# ============================================================================
# DICOM --> geometry-preserving NIfTI, then use original GPU fusion pipeline.
# ============================================================================
BASE_DIR = Path(
    "/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513"
)
DEFAULT_DICOM = {
    "Axial": BASE_DIR / "3_Axi_PD_fs_Rt",
    "Coronal": BASE_DIR / "5_Cor_PD_fs_Rt",
    "Sagittal": BASE_DIR / "6_Sag_PD_fs_Rt",
}
DEFAULT_OUTPUT = BASE_DIR / "fusion_output" / "PD_FS_triview_fused_0p4mm.nii.gz"


def dicom_folder_to_nifti(dicom_dir: Path, output_file: Path) -> dict:
    """Read exactly one DICOM series and preserve its patient-world geometry.

    SimpleITK stores physical coordinates in LPS; nibabel uses RAS.
    Array conversion is [z,y,x] -> [x,y,z], with affine LPS -> RAS.
    """
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_dir}")
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise RuntimeError(f"No *.dcm files found in {dicom_dir}")

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"GDCM found no DICOM series in {dicom_dir}")
    if len(series_ids) != 1:
        raise RuntimeError(
            f"Expected 1 DICOM series in {dicom_dir}, found {len(series_ids)}: {series_ids}"
        )
    filenames = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(dicom_dir), series_ids[0]
    ))
    if len(filenames) != len(dcm_files):
        raise RuntimeError(
            f"DICOM count mismatch in {dicom_dir}: found {len(dcm_files)} "
            f"*.dcm but GDCM selected {len(filenames)} files"
        )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    img = reader.Execute()
    if img.GetDimension() != 3 or img.GetDepth() < 2:
        raise RuntimeError(f"Not a valid 3D series: {dicom_dir}")

    # Reject different FrameOfReferenceUID when this metadata is present.
    frame_uid = None
    if reader.HasMetaDataKey(0, "0020|0052"):
        frame_uid = reader.GetMetaData(0, "0020|0052").strip()

    arr_zyx = sitk.GetArrayFromImage(img)
    arr_xyz = np.ascontiguousarray(arr_zyx.transpose(2, 1, 0), dtype=np.float32)
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    origin = np.asarray(img.GetOrigin(), dtype=np.float64)
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin
    affine_ras = np.diag([-1, -1, 1, 1]) @ affine_lps

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(arr_xyz, affine_ras, output_file, np.float32)

    data = {
        "dicom_dir": str(dicom_dir),
        "number_of_files": len(filenames),
        "size_xyz": list(img.GetSize()),
        "spacing_xyz_mm": [float(x) for x in spacing],
        "origin_lps_mm": [float(x) for x in origin],
        "frame_of_reference_uid": frame_uid,
        "affine_ras": affine_ras.tolist(),
    }
    print(f"  {dicom_dir.name}: {len(filenames)} slices, "
          f"shape={img.GetSize()}, spacing={img.GetSpacing()}", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="MMH PD-FS tri-view DICOM to 0.4 mm isotropic MRI using CUDA grid_sample"
    )
    parser.add_argument("--axial", type=Path, default=DEFAULT_DICOM["Axial"])
    parser.add_argument("--coronal", type=Path, default=DEFAULT_DICOM["Coronal"])
    parser.add_argument("--sagittal", type=Path, default=DEFAULT_DICOM["Sagittal"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing", type=float, default=0.4)
    parser.add_argument("--chunk-depth", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-harmonize", action="store_true")
    args = parser.parse_args()
    if args.spacing <= 0 or args.chunk_depth < 1:
        parser.error("--spacing and --chunk-depth must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mmh_triview_") as tmp:
        tmp_root = Path(tmp)
        case_id = args.output.parent.parent.name
        source_info = {}
        print("[DICOM] Converting three input series to geometry-preserving NIfTI...", flush=True)
        for view, folder in [("Axial", args.axial), ("Coronal", args.coronal),
                             ("Sagittal", args.sagittal)]:
            nifti_dir = tmp_root / case_id / view / DEFAULT_SEQUENCE
            nifti_path = nifti_dir / "image.nii.gz"
            source_info[view.lower()] = dicom_folder_to_nifti(folder, nifti_path)

        known_uids = [v["frame_of_reference_uid"] for v in source_info.values()
                      if v["frame_of_reference_uid"]]
        if len(set(known_uids)) > 1:
            raise RuntimeError(
                "FrameOfReferenceUID mismatch across views; registration is required "
                "before fusion. UIDs: " + repr(known_uids)
            )

        metadata = reconstruct_triview(
            root=tmp_root,
            case_id=case_id,
            sequence=DEFAULT_SEQUENCE,
            spacing_mm=args.spacing,
            chunk_depth=args.chunk_depth,
            torch_device=args.device,
            harmonize=not args.no_harmonize,
            output_path=args.output,
        )
        # Replace temporary file paths with permanent source DICOM metadata.
        metadata["dicom_sources"] = source_info
        metadata.pop("selected_sources", None)
        metadata["fusion_kind"] = "geometry-based resampling and mean fusion; no super-resolution"

    metadata_path = Path(str(args.output) + ".json")
    json_dump(metadata, metadata_path)
    print("\nFUSED MRI:", args.output)
    print("METADATA :", metadata_path)
    print("Open the NIfTI in 3D Slicer to inspect all three planes before inference.")


if __name__ == "__main__":
    main()
