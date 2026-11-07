#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bone-guided Meniscus Automatic Point Selection
Version 1.1 - Dark Region Core Selection

Features
--------
1. Femur / tibia anatomical localization
2. Fibula-guided medial/lateral identification
3. Bone-distance meniscus candidate generation
4. Dark-region detection
5. Connected-component filtering
6. 3D distance-transform core selection
7. Anatomical boundary safety margin
8. Independent point-centered orthogonal previews
9. 3D Slicer FCSV export

This version DOES NOT run nnInteractive inference.

Author: MMH Knee MRI Research Pipeline
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scipy import ndimage as ndi


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(
    "/media/we66530/Transcend/MMH_knee_dcm/"
    "PACS_downloads/37321865_20260513/fusion_output"
)

IMAGE_PATH = (
    BASE / "PD_FS_triview_fused_0p4mm.nii.gz"
)

BONE_DIR = BASE / "bone_guided_nninteractive"

BONE_PATHS = {
    "femur": BONE_DIR / "femur_nninteractive.nii.gz",
    "tibia": BONE_DIR / "tibia_nninteractive.nii.gz",
    "fibula": BONE_DIR / "fibula_nninteractive.nii.gz",
    "patella": BONE_DIR / "patella_nninteractive.nii.gz",
}

OUTPUT_DIR = BASE / "meniscus_guided_nninteractive"


# ============================================================
# BASIC UTILITIES
# ============================================================

def log(message):
    print(message, flush=True)


def load_image(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    image = sitk.ReadImage(str(path))

    if image.GetDimension() != 3:
        raise ValueError(
            f"Expected 3D image: {path}"
        )

    return image


def same_geometry(a, b):

    return (
        a.GetSize() == b.GetSize()
        and np.allclose(
            a.GetSpacing(),
            b.GetSpacing(),
            atol=1e-4,
            rtol=0
        )
        and np.allclose(
            a.GetOrigin(),
            b.GetOrigin(),
            atol=1e-3,
            rtol=0
        )
        and np.allclose(
            a.GetDirection(),
            b.GetDirection(),
            atol=1e-5,
            rtol=0
        )
    )


def save_mask(array, reference, path):

    image = sitk.GetImageFromArray(
        array.astype(np.uint8)
    )

    image.CopyInformation(reference)

    sitk.WriteImage(
        image,
        str(path),
        True
    )

    log(f"[SAVED] {path.name}")


def load_bone_masks(reference, allow_resample=False):

    masks = {}

    for name, path in BONE_PATHS.items():

        image = load_image(path)

        if not same_geometry(image, reference):

            if not allow_resample:
                raise RuntimeError(
                    f"{name}: geometry mismatch. "
                    "Use --allow-resample only when "
                    "physical-space registration is correct."
                )

            log(f"[RESAMPLE] {name}")

            image = sitk.Resample(
                image,
                reference,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8
            )

        mask = sitk.GetArrayFromImage(image) > 0

        if not mask.any():
            raise RuntimeError(
                f"{name}: empty mask"
            )

        masks[name] = mask

        log(
            f"[BONE] {name:8s}: "
            f"{int(mask.sum()):,} voxels"
        )

    return masks


# ============================================================
# PHYSICAL COORDINATES
# ============================================================

def geometry_matrix(image):

    direction = np.asarray(
        image.GetDirection(),
        dtype=float
    ).reshape(3, 3)

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=float
    )

    reverse = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    matrix = (
        direction
        @ np.diag(spacing)
        @ reverse
    )

    origin = np.asarray(
        image.GetOrigin(),
        dtype=float
    )

    return origin, matrix


def voxel_to_lps(index_zyx, origin, matrix):

    return (
        origin
        + matrix @ np.asarray(
            index_zyx,
            dtype=float
        )
    )


def sample_physical_points(
    mask,
    origin,
    matrix,
    max_points=120000
):

    indices = np.argwhere(mask)

    if len(indices) == 0:
        raise RuntimeError("Empty mask")

    if len(indices) > max_points:

        step = int(
            np.ceil(len(indices) / max_points)
        )

        indices = indices[::step]

    return origin + indices @ matrix.T


# ============================================================
# ANATOMICAL LOCALIZATION
# ============================================================

def estimate_joint_anatomy(
    masks,
    origin,
    matrix
):

    physical = {}

    for name, mask in masks.items():

        physical[name] = sample_physical_points(
            mask,
            origin,
            matrix
        )

    tibia = physical["tibia"]
    femur = physical["femur"]
    fibula = physical["fibula"]

    # LPS Z increases superiorly.
    tibia_top = np.percentile(
        tibia[:, 2],
        99.2
    )

    femur_bottom = np.percentile(
        femur[:, 2],
        0.8
    )

    joint_z = (
        tibia_top + femur_bottom
    ) / 2

    gap = tibia_top - femur_bottom

    if abs(gap) > 35:

        raise RuntimeError(
            "Implausible tibiofemoral joint geometry. "
            f"Estimated gap: {gap:.2f} mm"
        )

    proximal_tibia = tibia[
        (tibia[:, 2] > tibia_top - 22)
        &
        (tibia[:, 2] < tibia_top - 3)
    ]

    if len(proximal_tibia) < 50:
        raise RuntimeError(
            "Insufficient proximal tibia voxels"
        )

    tibia_center = np.median(
        proximal_tibia[:, :2],
        axis=0
    )

    fibula_proximal = fibula[
        fibula[:, 2] >= np.percentile(
            fibula[:, 2],
            85
        )
    ]

    fibula_center = np.median(
        fibula_proximal[:, :2],
        axis=0
    )

    lateral = (
        fibula_center - tibia_center
    )

    lateral_distance = np.linalg.norm(lateral)

    if lateral_distance < 8:
        raise RuntimeError(
            "Unreliable fibula lateral landmark"
        )

    lateral = lateral / lateral_distance

    log("\n[ANATOMY]")
    log(f"  Tibia top: {tibia_top:.2f} mm")
    log(f"  Femur bottom: {femur_bottom:.2f} mm")
    log(f"  Joint Z: {joint_z:.2f} mm")
    log(f"  Lateral direction: {lateral}")

    return {
        "joint_z": joint_z,
        "tibia_top": tibia_top,
        "tibia_center": tibia_center,
        "lateral": lateral
    }


# ============================================================
# JOINT CROP
# ============================================================

def get_joint_crop(
    masks,
    anatomy,
    origin,
    matrix,
    shape
):

    joint_z = anatomy["joint_z"]

    selected_regions = []

    definitions = [
        ("tibia", joint_z - 25, joint_z + 8),
        ("femur", joint_z - 8, joint_z + 25)
    ]

    for name, z_min, z_max in definitions:

        indices = np.argwhere(masks[name])

        if len(indices) > 1500000:
            indices = indices[::2]

        physical_z = (
            origin[2]
            + indices @ matrix[2, :]
        )

        selected = indices[
            (physical_z >= z_min)
            &
            (physical_z <= z_max)
        ]

        if len(selected) == 0:
            raise RuntimeError(
                f"No {name} voxels near joint"
            )

        selected_regions.append(selected)

    combined = np.concatenate(
        selected_regions,
        axis=0
    )

    # Approximately 18 mm margin at 0.4 mm.
    margin = np.ceil(
        18.0 / np.linalg.norm(matrix, axis=0)
    ).astype(int)

    lower = np.maximum(
        combined.min(axis=0) - margin,
        0
    )

    upper = np.minimum(
        combined.max(axis=0) + margin + 1,
        np.asarray(shape)
    )

    slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(lower, upper)
    )

    log("\n[JOINT ROI]")
    log(f"  Lower ZYX: {lower}")
    log(f"  Upper ZYX: {upper}")

    return slices, lower


# ============================================================
# ANATOMICAL CANDIDATES
# ============================================================

def generate_candidates(
    mri,
    masks,
    anatomy,
    origin,
    matrix,
    slices,
    lower,
    max_femur_distance=14,
    max_tibia_distance=10,
    bone_exclusion=0.8
):

    crop = mri[slices]

    femur = masks["femur"][slices]
    tibia = masks["tibia"][slices]
    fibula = masks["fibula"][slices]
    patella = masks["patella"][slices]

    spacing_zyx = np.linalg.norm(
        matrix,
        axis=0
    )

    log("\n[DISTANCE] Computing bone distance maps")

    # Outside bone = positive distance.
    # Inside bone = zero.
    distance_femur = ndi.distance_transform_edt(
        ~femur,
        sampling=spacing_zyx
    ).astype(np.float32)

    distance_tibia = ndi.distance_transform_edt(
        ~tibia,
        sampling=spacing_zyx
    ).astype(np.float32)

    all_bones = (
        femur | tibia | fibula | patella
    )

    distance_bones = ndi.distance_transform_edt(
        ~all_bones,
        sampling=spacing_zyx
    ).astype(np.float32)

    z0, y0, x0 = lower

    z, y, x = np.ogrid[
        z0:z0 + crop.shape[0],
        y0:y0 + crop.shape[1],
        x0:x0 + crop.shape[2]
    ]

    px = (
        origin[0]
        + matrix[0, 0] * z
        + matrix[0, 1] * y
        + matrix[0, 2] * x
    )

    py = (
        origin[1]
        + matrix[1, 0] * z
        + matrix[1, 1] * y
        + matrix[1, 2] * x
    )

    pz = (
        origin[2]
        + matrix[2, 0] * z
        + matrix[2, 1] * y
        + matrix[2, 2] * x
    )

    center = anatomy["tibia_center"]
    lateral = anatomy["lateral"]
    joint_z = anatomy["joint_z"]

    signed = (
        (px - center[0]) * lateral[0]
        +
        (py - center[1]) * lateral[1]
    )

    transverse = (
        (px - center[0]) * (-lateral[1])
        +
        (py - center[1]) * lateral[0]
    )

    candidate = (
        (distance_femur < max_femur_distance)
        &
        (distance_tibia < max_tibia_distance)
        &
        (distance_bones > bone_exclusion)
        &
        (pz > joint_z - 10)
        &
        (pz < joint_z + 12)
        &
        (np.abs(transverse) < 43)
        &
        (np.abs(signed) < 53)
        &
        (np.abs(signed) > 7)
    )

    # Exclude central intercondylar region.
    candidate &= (
        (signed / 25) ** 2
        +
        (transverse / 29) ** 2
    ) > 0.20

    medial = (
        candidate & (signed < 0)
    )

    lateral_mask = (
        candidate & (signed > 0)
    )

    log("\n[ANATOMICAL CANDIDATES]")
    log(f"  Medial: {int(medial.sum()):,} voxels")
    log(f"  Lateral: {int(lateral_mask.sum()):,} voxels")

    return {
        "crop": crop,
        "medial": medial,
        "lateral": lateral_mask,
        "distance_femur": distance_femur,
        "distance_tibia": distance_tibia,
        "signed": signed,
        "physical_y": py
    }


# ============================================================
# DARK REGION CORE SELECTION
# ============================================================

def select_meniscus_points(
    candidate,
    crop,
    distance_femur,
    distance_tibia,
    signed,
    physical_y,
    lower,
    origin,
    matrix,
    side,
    low_signal_percentile=35,
    min_score=0.35,
    min_candidate_depth=0.8,
    min_dark_depth=0.6
):

    """
    Select 3 interior points:
        anterior
        body
        posterior

    Major changes from v1.0:
        - Smoothed intensity
        - Local dark threshold
        - Connected component filtering
        - Dark-region EDT
        - Candidate-region EDT
        - Interior safety constraints
    """

    points = []

    spacing_zyx = np.linalg.norm(
        matrix,
        axis=0
    )

    valid = (
        candidate
        &
        np.isfinite(crop)
    )

    if valid.sum() < 100:

        log(
            f"[WARNING] {side}: "
            "insufficient anatomical candidates"
        )

        return points, np.zeros_like(
            candidate,
            dtype=bool
        )

    # --------------------------------------------------------
    # 1. MRI smoothing
    # --------------------------------------------------------

    image = crop.astype(
        np.float32,
        copy=True
    )

    if not np.isfinite(image).all():

        median = float(
            np.median(image[valid])
        )

        image[~np.isfinite(image)] = median

    smoothed = ndi.gaussian_filter(
        image,
        sigma=1.0
    )

    # --------------------------------------------------------
    # 2. Candidate boundary distance
    # --------------------------------------------------------

    candidate_depth = ndi.distance_transform_edt(
        valid,
        sampling=spacing_zyx
    ).astype(np.float32)

    # --------------------------------------------------------
    # 3. AP zone separation
    # --------------------------------------------------------

    ap = np.broadcast_to(
        physical_y,
        crop.shape
    )

    boundaries = np.percentile(
        ap[valid],
        [0, 33.33, 66.67, 100]
    )

    zones = [
        "anterior",
        "body",
        "posterior"
    ]

    dark_region_all = np.zeros_like(
        candidate,
        dtype=bool
    )

    # --------------------------------------------------------
    # 4. Process each zone independently
    # --------------------------------------------------------

    for i, zone in enumerate(zones):

        zone_mask = (
            valid
            &
            (ap >= boundaries[i])
            &
            (
                (ap <= boundaries[i + 1])
                if i == 2
                else (ap < boundaries[i + 1])
            )
        )

        if zone_mask.sum() < 50:

            log(
                f"[WARNING] {side}/{zone}: "
                "too few candidates"
            )

            continue

        # ----------------------------------------------------
        # 5. Local dark intensity threshold
        # ----------------------------------------------------

        threshold = np.percentile(
            smoothed[zone_mask],
            low_signal_percentile
        )

        dark = (
            zone_mask
            &
            (smoothed <= threshold)
        )

        # Reduce isolated dark voxels.
        dark_support = ndi.gaussian_filter(
            dark.astype(np.float32),
            sigma=1.2
        )

        dark &= (
            dark_support >= 0.15
        )

        # ----------------------------------------------------
        # 6. Connected-component filtering
        # ----------------------------------------------------

        structure = ndi.generate_binary_structure(
            rank=3,
            connectivity=2
        )

        labels, n_components = ndi.label(
            dark,
            structure=structure
        )

        sizes = np.bincount(
            labels.ravel()
        )

        # Keep sufficiently large components.
        # Background component 0 must be excluded.
        min_component_voxels = 20

        keep_ids = np.flatnonzero(
            sizes >= min_component_voxels
        )

        keep_ids = keep_ids[
            keep_ids != 0
        ]

        dark = np.isin(
            labels,
            keep_ids
        )

        if not dark.any():

            log(
                f"[WARNING] {side}/{zone}: "
                "no reliable dark component"
            )

            continue

        dark_region_all |= dark

        # ----------------------------------------------------
        # 7. Dark-region distance transform
        # ----------------------------------------------------

        dark_depth = ndi.distance_transform_edt(
            dark,
            sampling=spacing_zyx
        ).astype(np.float32)

        dark_core_score = np.clip(
            dark_depth / 1.2,
            0,
            1
        )

        candidate_core_score = np.clip(
            candidate_depth / 1.2,
            0,
            1
        )

        # ----------------------------------------------------
        # 8. Intensity score
        # ----------------------------------------------------

        values = smoothed[zone_mask]

        low = np.percentile(
            values,
            5
        )

        high = np.percentile(
            values,
            75
        )

        denominator = max(
            float(high - low),
            1e-5
        )

        intensity_score = 1.0 - np.clip(
            (smoothed - low) / denominator,
            0,
            1
        )

        # ----------------------------------------------------
        # 9. Anatomical proximity score
        # ----------------------------------------------------

        proximity_score = (
            np.exp(
                -0.5 *
                ((distance_tibia - 3.0) / 3.5) ** 2
            )
            *
            np.exp(
                -0.5 *
                ((distance_femur - 5.0) / 5.5) ** 2
            )
        )

        # ----------------------------------------------------
        # 10. Combined score
        # ----------------------------------------------------

        score = (
            0.40 * dark_core_score
            +
            0.20 * candidate_core_score
            +
            0.25 * intensity_score
            +
            0.15 * proximity_score
        )

        # ----------------------------------------------------
        # 11. Safety constraints
        # ----------------------------------------------------

        safe = (
            dark
            &
            (candidate_depth >= min_candidate_depth)
            &
            (dark_depth >= min_dark_depth)
        )

        if not safe.any():

            log(
                f"[WARNING] {side}/{zone}: "
                "no sufficiently interior dark point"
            )

            continue

        score[~safe] = -np.inf

        # ----------------------------------------------------
        # 12. Select maximum-score interior point
        # ----------------------------------------------------

        best_index = np.unravel_index(
            np.argmax(score),
            score.shape
        )

        best_score = float(
            score[best_index]
        )

        if (
            not np.isfinite(best_score)
            or best_score < min_score
        ):

            log(
                f"[WARNING] {side}/{zone}: "
                f"score={best_score:.3f}, skipped"
            )

            continue

        global_index = tuple(
            int(a + b)
            for a, b in zip(
                best_index,
                lower
            )
        )

        lps = voxel_to_lps(
            global_index,
            origin,
            matrix
        )

        point = {
            "side": side,
            "zone": zone,
            "voxel_zyx": list(global_index),

            "lps_mm": lps.tolist(),

            "ras_mm": [
                -float(lps[0]),
                -float(lps[1]),
                float(lps[2])
            ],

            "score": best_score,

            "intensity": float(
                crop[best_index]
            ),

            "dark_depth_mm": float(
                dark_depth[best_index]
            ),

            "candidate_depth_mm": float(
                candidate_depth[best_index]
            ),

            "femur_distance_mm": float(
                distance_femur[best_index]
            ),

            "tibia_distance_mm": float(
                distance_tibia[best_index]
            )
        }

        points.append(point)

        log(
            f"[POINT] {side:7s} {zone:9s} "
            f"ZYX={global_index} "
            f"score={best_score:.3f} "
            f"dark_depth={dark_depth[best_index]:.2f}mm "
            f"candidate_depth={candidate_depth[best_index]:.2f}mm"
        )

    return points, dark_region_all


# ============================================================
# POINT EXPORT
# ============================================================

def save_points_json(points):

    path = OUTPUT_DIR / "positive_points.json"

    payload = {
        "version": "1.1",
        "method": "bone_guided_dark_region_core",
        "coordinate_system": "LPS",
        "voxel_order": "ZYX",
        "note": (
            "Points are heuristic proposals, "
            "not validated meniscus labels."
        ),
        "points": points
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2
        )


def save_points_csv(points):

    path = OUTPUT_DIR / "positive_points.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "side",
            "zone",
            "z",
            "y",
            "x",
            "LPS_x",
            "LPS_y",
            "LPS_z",
            "score",
            "intensity",
            "dark_depth_mm",
            "candidate_depth_mm",
            "femur_distance_mm",
            "tibia_distance_mm"
        ])

        for p in points:

            writer.writerow([
                p["side"],
                p["zone"],
                *p["voxel_zyx"],
                *p["lps_mm"],
                p["score"],
                p["intensity"],
                p["dark_depth_mm"],
                p["candidate_depth_mm"],
                p["femur_distance_mm"],
                p["tibia_distance_mm"]
            ])


def save_slicer_fcsv(points):

    path = (
        OUTPUT_DIR /
        "meniscus_positive_points_LPS.fcsv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Markups fiducial file version = 4.11\n"
        )

        # 1 = LPS.
        f.write(
            "# CoordinateSystem = 1\n"
        )

        f.write(
            "# columns = "
            "id,x,y,z,ow,ox,oy,oz,"
            "vis,sel,lock,label,desc,associatedNodeID\n"
        )

        writer = csv.writer(f)

        for i, p in enumerate(points, 1):

            x, y, z = p["lps_mm"]

            writer.writerow([
                f"point_{i}",
                x, y, z,
                0, 0, 0, 1,
                1, 1, 0,
                f"{p['side']}_{p['zone']}",
                f"score={p['score']:.3f}",
                ""
            ])


# ============================================================
# POINT-CENTERED PREVIEW
# ============================================================

def save_preview(mri, points):

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not points:
        return

    # Each point receives its own orthogonal views.
    # No longer project different-Z points onto
    # a single arbitrarily selected slice.

    n = len(points)

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12, max(3 * n, 6)),
        squeeze=False,
        constrained_layout=True
    )

    for row, p in enumerate(points):

        z, y, x = p["voxel_zyx"]

        views = [
            (
                mri[z, :, :],
                (x, y),
                "Array Z"
            ),
            (
                mri[:, y, :],
                (x, z),
                "Array Y"
            ),
            (
                mri[:, :, x],
                (y, z),
                "Array X"
            )
        ]

        for col, (image, center, title) in enumerate(views):

            ax = axes[row, col]

            finite = image[
                np.isfinite(image)
            ]

            if finite.size == 0:
                ax.set_axis_off()
                continue

            vmin, vmax = np.percentile(
                finite,
                [2, 98]
            )

            if vmax <= vmin:
                vmax = vmin + 1

            ax.imshow(
                image,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax
            )

            cx, cy = center

            ax.scatter(
                cx,
                cy,
                marker="+",
                color="red",
                s=90,
                linewidths=1.5
            )

            ax.set_xlim(
                max(0, cx - 75),
                min(image.shape[1] - 1, cx + 75)
            )

            ax.set_ylim(
                max(0, cy - 75),
                min(image.shape[0] - 1, cy + 75)
            )

            ax.set_title(
                f"{p['side']} / {p['zone']} / {title}"
            )

    fig.suptitle(
        "Meniscus Dark-Core Point Selection v1.1"
    )

    fig.savefig(
        OUTPUT_DIR / "point_selection_preview.png",
        dpi=140
    )

    plt.close(fig)

    log("[SAVED] point_selection_preview.png")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allow-resample",
        action="store_true"
    )

    parser.add_argument(
        "--max-femur-distance",
        type=float,
        default=14
    )

    parser.add_argument(
        "--max-tibia-distance",
        type=float,
        default=10
    )

    parser.add_argument(
        "--low-signal-percentile",
        type=float,
        default=35
    )

    parser.add_argument(
        "--min-candidate-depth",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--min-dark-depth",
        type=float,
        default=0.6
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    log("\n========================================")
    log(" Bone-guided Meniscus Prompting v1.1")
    log(" Dark Region Core Selection")
    log("========================================\n")

    image = load_image(
        IMAGE_PATH
    )

    mri = sitk.GetArrayFromImage(
        image
    ).astype(np.float32)

    log(f"[MRI] {IMAGE_PATH}")
    log(f"[SHAPE ZYX] {mri.shape}")
    log(f"[SPACING XYZ] {image.GetSpacing()}")

    masks = load_bone_masks(
        image,
        allow_resample=args.allow_resample
    )

    origin, matrix = geometry_matrix(
        image
    )

    anatomy = estimate_joint_anatomy(
        masks,
        origin,
        matrix
    )

    slices, lower = get_joint_crop(
        masks,
        anatomy,
        origin,
        matrix,
        mri.shape
    )

    result = generate_candidates(
        mri,
        masks,
        anatomy,
        origin,
        matrix,
        slices,
        lower,
        max_femur_distance=args.max_femur_distance,
        max_tibia_distance=args.max_tibia_distance
    )

    all_points = []

    multilabel = np.zeros(
        mri.shape,
        dtype=np.uint8
    )

    for side, label in [
        ("medial", 1),
        ("lateral", 2)
    ]:

        log(f"\n[PROCESS] {side}")

        # ----------------------------------------------------
        # Save anatomical candidate region
        # ----------------------------------------------------

        anatomical_full = np.zeros(
            mri.shape,
            dtype=np.uint8
        )

        anatomical_full[slices] = (
            result[side].astype(np.uint8)
        )

        save_mask(
            anatomical_full,
            image,
            OUTPUT_DIR /
            f"{side}_anatomical_candidate.nii.gz"
        )

        # ----------------------------------------------------
        # Dark-core point selection
        # ----------------------------------------------------

        points, dark_region = select_meniscus_points(
            result[side],
            result["crop"],
            result["distance_femur"],
            result["distance_tibia"],
            result["signed"],
            result["physical_y"],
            lower,
            origin,
            matrix,
            side,
            low_signal_percentile=args.low_signal_percentile,
            min_candidate_depth=args.min_candidate_depth,
            min_dark_depth=args.min_dark_depth
        )

        all_points.extend(points)

        # ----------------------------------------------------
        # Save dark-region mask separately
        # ----------------------------------------------------

        dark_full = np.zeros(
            mri.shape,
            dtype=np.uint8
        )

        dark_full[slices] = (
            dark_region.astype(np.uint8)
        )

        save_mask(
            dark_full,
            image,
            OUTPUT_DIR /
            f"{side}_dark_region.nii.gz"
        )

        multilabel[
            (dark_full > 0)
            &
            (multilabel == 0)
        ] = label

        del anatomical_full
        del dark_full

    # --------------------------------------------------------
    # Save combined dark-region preview
    # --------------------------------------------------------

    save_mask(
        multilabel,
        image,
        OUTPUT_DIR /
        "meniscus_dark_region_multilabel.nii.gz"
    )

    # --------------------------------------------------------
    # Save point coordinates
    # --------------------------------------------------------

    save_points_json(
        all_points
    )

    save_points_csv(
        all_points
    )

    save_slicer_fcsv(
        all_points
    )

    # --------------------------------------------------------
    # Save point-centered previews
    # --------------------------------------------------------

    save_preview(
        mri,
        all_points
    )

    log("\n========================================")
    log("[DONE] Meniscus prompting completed")
    log(f"[POINTS] {len(all_points)}")
    log(f"[OUTPUT] {OUTPUT_DIR}")
    log("========================================")


if __name__ == "__main__":
    main()