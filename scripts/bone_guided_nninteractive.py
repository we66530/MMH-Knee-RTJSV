#!/usr/bin/env python3
"""Bone-guided single-positive-point nnInteractive segmentation (four knee bones).

Dependencies: numpy, scipy, SimpleITK, torch + nnInteractive (for --mode infer).
No model downloads are initiated by this script.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

DEFAULT_ROOT = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513/fusion_output')
MASKS = {
    'femur': 'totalseg_bone_prior/total_mr/femur_right.nii.gz',
    'tibia': 'totalseg_bone_prior/appendicular_bones_mr/tibia.nii.gz',
    'fibula': 'totalseg_bone_prior/appendicular_bones_mr/fibula.nii.gz',
    'patella': 'totalseg_bone_prior/appendicular_bones_mr/patella.nii.gz',
}


def same_geometry(a, b, tol=1e-4):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol, rtol=0)
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol, rtol=0))


def mask_to_reference(mask, ref, allow_resample, name):
    if same_geometry(mask, ref):
        return mask
    if not allow_resample:
        raise RuntimeError(f'{name}: geometry differs from MRI. Verify spatial registration, then rerun with --allow-resample if both are in the SAME physical space.')
    print(f'[WARN] {name}: resampling mask to MRI grid (nearest-neighbor). Verify alignment visually.')
    return sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def choose_seed(mask_zyx, spacing_xyz):
    # Use 26-connectivity; mask is an array indexed [z, y, x].
    labels, n = ndi.label(mask_zyx, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        raise ValueError('Empty mask; no positive seed can be selected.')
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = labels == counts.argmax()
    # Physical distance (mm), in array-axis order z, y, x.
    distance = ndi.distance_transform_edt(largest, sampling=tuple(spacing_xyz[::-1]))
    zyx = tuple(int(v) for v in np.unravel_index(int(np.argmax(distance)), distance.shape))
    return zyx, float(distance[zyx]), int(largest.sum()), int(mask_zyx.sum())


def write_like(ref, data_zyx, dest):
    im = sitk.GetImageFromArray(data_zyx)
    im.CopyInformation(ref)
    sitk.WriteImage(im, str(dest), useCompression=True)


def get_model_path(user_path):
    # Explicit model path is the safest way to reuse existing checkpoints.
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f'Checkpoint folder not found: {p}')
        return p
    base = Path(os.environ.get('NNINTERACTIVE_MODEL_DIR', '~/.nninteractive')).expanduser()
    options = sorted(p for p in base.glob('nnInteractive*') if p.is_dir()) if base.exists() else []
    if len(options) == 1:
        print(f'[MODEL] Reusing existing checkpoint: {options[0]}')
        return options[0]
    if len(options) > 1:
        raise RuntimeError(f'Multiple local model directories under {base}: {options}. Specify --model-dir.')
    raise FileNotFoundError(f'No existing nnInteractive model found under {base}. Pass --model-dir pointing to the trained model folder. Script does not download weights.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--image', type=Path, default=None)
    p.add_argument('--mode', choices=['points', 'infer'], default='points')
    p.add_argument('--allow-resample', action='store_true', help='Nearest-neighbor resample masks ONLY if already co-registered in physical space')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--model-dir', type=str, default=None)
    args = p.parse_args()

    image_path = args.image or args.root / 'PD_FS_triview_fused_0p4mm.nii.gz'
    output = args.output_dir or args.root / 'bone_guided_nninteractive'
    output.mkdir(parents=True, exist_ok=True)
    print(f'[MRI] {image_path}')
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    ref = sitk.ReadImage(str(image_path))
    if ref.GetDimension() != 3:
        raise ValueError('Expected a 3D MRI volume.')

    prompts = []
    for bone, relpath in MASKS.items():
        path = args.root / relpath
        if not path.is_file():
            raise FileNotFoundError(path)
        mask_img = mask_to_reference(sitk.ReadImage(str(path)), ref, args.allow_resample, bone)
        arr = sitk.GetArrayFromImage(mask_img) != 0
        zyx, depth, component_size, full_size = choose_seed(arr, ref.GetSpacing())
        xyz = [zyx[2], zyx[1], zyx[0]]
        lps = tuple(float(v) for v in ref.TransformIndexToPhysicalPoint(tuple(xyz)))
        record = {
            'bone': bone,
            'source_mask': str(path),
            'point_zyx_nninteractive': list(zyx),
            'point_xyz_sitk': xyz,
            'physical_lps_mm': list(lps),
            'physical_ras_mm': [-lps[0], -lps[1], lps[2]],
            'seed_depth_mm': depth,
            'largest_component_voxels': component_size,
            'total_mask_voxels': full_size,
        }
        prompts.append(record)
        print(f'[SEED] {bone:7s}: zyx={zyx}  xyz={tuple(xyz)}  depth={depth:.2f} mm  largest={component_size:,} voxels')

    payload = {'image': str(image_path), 'image_shape_zyx': list(reversed(ref.GetSize())),
               'spacing_xyz_mm': list(ref.GetSpacing()), 'coordinate_note': 'nnInteractive points index the SimpleITK array as (z,y,x); SimpleITK spatial indices are (x,y,z).',
               'prompts': prompts}
    (output / 'prompts.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with (output / 'prompts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(prompts[0]))
        writer.writeheader()
        writer.writerows(prompts)
    print(f'[SAVED] {output / "prompts.json"}')
    if args.mode == 'points':
        print('[DONE] Point selection only. No GPU inference performed.')
        return

    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    checkpoint = get_model_path(args.model_dir)
    print(f'[MODEL] Loading {checkpoint}')
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda:0'), use_torch_compile=False,
        verbose=False, torch_n_threads=min(os.cpu_count() or 8, 16), do_autozoom=True)
    session.initialize_from_trained_model_folder(str(checkpoint))
    img_zyx = sitk.GetArrayFromImage(ref)
    # Keep original intensity and precision; do not apply windowing/normalization.
    session.set_image(img_zyx[None])
    buffer = torch.zeros(img_zyx.shape, dtype=torch.uint8)
    session.set_target_buffer(buffer)
    multilabel = np.zeros(img_zyx.shape, dtype=np.uint8)
    overlaps = {}
    for label, record in enumerate(prompts, start=1):
        bone = record['bone']
        session.reset_interactions()
        pt = tuple(record['point_zyx_nninteractive'])
        session.add_point_interaction(pt, include_interaction=True)
        mask = (buffer.cpu().numpy() > 0).astype(np.uint8).copy()
        destination = output / f'{bone}_nninteractive.nii.gz'
        write_like(ref, mask, destination)
        overlap = int(np.count_nonzero((multilabel > 0) & (mask > 0)))
        overlaps[bone] = overlap
        # Keep earlier labels in overlapping voxels for preview only.
        multilabel[(multilabel == 0) & (mask > 0)] = label
        print(f'[PRED] {bone:7s}: {mask.sum():,} voxels, overlap={overlap:,}; saved {destination.name}')
    write_like(ref, multilabel, output / 'four_bones_multilabel_preview.nii.gz')
    (output / 'overlaps.json').write_text(json.dumps(overlaps, indent=2), encoding='utf-8')
    print('[DONE] Four independent segmentations + multilabel preview saved to:', output)


if __name__ == '__main__':
    main()
