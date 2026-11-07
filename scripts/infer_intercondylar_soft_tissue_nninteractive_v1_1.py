#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run reviewed intercondylar soft-tissue prompts with local nnInteractive.

Modes:
  infer    Save positive-only and positive+negative masks, plus intersection QC.
  subtract Subtract ONLY the intersection with the existing raw v3.2 residual
           ROIs; output a new comparison version, never overwrite v3.2.

This is an exploratory exclusion mask, NOT a validated ACL/PCL segmentation.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

DEFAULT_CASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513')
DEFAULT_MODEL = Path.home() / '.nninteractive/models/nnInteractive_v1.0'


def say(*items):
    print(*items, flush=True)


def same_grid(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4, rtol=0)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0))


def read_mask(path, reference):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = sitk.ReadImage(str(path))
    if not same_grid(im, reference):
        raise RuntimeError(f'Geometry mismatch: {path}. No implicit resampling for quantitative masks.')
    return sitk.GetArrayFromImage(im) > 0


def save_mask(path, mask, reference):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = sitk.GetImageFromArray(mask.astype(np.uint8))
    im.CopyInformation(reference)
    sitk.WriteImage(im, str(path), True)
    say(f'[SAVE] {path.name}: {int(mask.sum()):,} voxels')


def extract_points(items, shape, reference, kind):
    result = []
    seen = set()
    for p in items:
        raw = p.get('voxel_zyx')
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError(f'Invalid {kind} point: {p}')
        if any(isinstance(v, bool) or int(v) != v for v in raw):
            raise ValueError(f'Noninteger {kind} coordinates: {raw}')
        xyz = tuple(int(v) for v in raw)
        if any(v < 0 or v >= bound for v, bound in zip(xyz, shape)):
            raise ValueError(f'{kind} point out of MRI bounds: {xyz}')
        if 'lps_mm' in p:
            lps = np.asarray(reference.TransformIndexToPhysicalPoint(tuple(xyz[::-1])))
            if not np.allclose(lps, np.asarray(p['lps_mm'], dtype=float), atol=0.5):
                raise RuntimeError(f'{kind} point LPS does not match MRI affine at {xyz}')
        if xyz not in seen:
            result.append(xyz)
            seen.add(xyz)
    return result


def valid_prompts(prompt_path, reference, image_shape):
    data = json.loads(prompt_path.read_text(encoding='utf-8'))
    if 'positive' not in data or 'negative' not in data:
        raise ValueError('Expected point_selection_v1/prompts.json with positive and negative lists')
    if 'image' in data:
        expected = Path(data['image'])
        # Dataset may have been moved; allow path movement but compare path when present.
        if expected.exists() and not same_grid(sitk.ReadImage(str(expected)), reference):
            raise RuntimeError('Prompts refer to another MRI geometry')
    pos = extract_points(data['positive'], image_shape, reference, 'positive')
    neg = extract_points(data['negative'], image_shape, reference, 'negative')
    if not pos:
        raise RuntimeError('No positive points; review point-selection output before inference')
    if set(pos) & set(neg):
        raise RuntimeError('A coordinate is both positive and negative')
    return pos, neg, data


def infer(args, base, out, reference):
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    prompt_file = args.prompts or (base / 'intercondylar_soft_tissue/point_selection_v1_1/prompts.json')
    if not prompt_file.is_file():
        raise FileNotFoundError(f'Generate point-selection first: {prompt_file}')
    image = sitk.GetArrayFromImage(reference)
    positives, negatives, prompt_data = valid_prompts(prompt_file, reference, image.shape)
    if not args.use_negatives:
        negatives = []
    checkpoint = args.model_dir / 'fold_0/checkpoint_final.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'nnInteractive checkpoint not found: {checkpoint}')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device available in this Python environment')
    say(f'[MODEL] {args.model_dir}; GPU={torch.cuda.get_device_name(0)}')
    say(f'[PROMPTS] {len(positives)} positive, {len(negatives)} negative')
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda:0'), use_torch_compile=False, verbose=False,
        torch_n_threads=min(os.cpu_count() or 4, 16), do_autozoom=True)
    session.initialize_from_trained_model_folder(str(args.model_dir))
    session.set_image(np.ascontiguousarray(image[None]))  # SimpleITK array is Z,Y,X
    target = torch.zeros(image.shape, dtype=torch.uint8)
    session.set_target_buffer(target)
    session.reset_interactions()
    target.zero_()
    for i, point in enumerate(positives, 1):
        say(f'[POS {i}/{len(positives)}] ZYX={point}')
        session.add_point_interaction(point, include_interaction=True)
    positive_mask = target.cpu().numpy().copy() > 0
    save_mask(out / 'intercondylar_positive_only.nii.gz', positive_mask, reference)
    for i, point in enumerate(negatives, 1):
        say(f'[NEG {i}/{len(negatives)}] ZYX={point}')
        session.add_point_interaction(point, include_interaction=False)
    final_mask = target.cpu().numpy().copy() > 0
    save_mask(out / 'intercondylar_soft_tissue_raw.nii.gz', final_mask, reference)
    # Bone masking is a post-inference exclusion, NOT an extra negative prompt.
    bones_dir = base / 'bone_guided_nninteractive'
    bone = np.zeros(image.shape, dtype=bool)
    for name in ('femur', 'tibia', 'fibula', 'patella'):
        bone |= read_mask(bones_dir / f'{name}_nninteractive.nii.gz', reference)
    bone_excluded = final_mask & ~bone
    save_mask(out / 'intercondylar_soft_tissue_bone_excluded.nii.gz',
              bone_excluded, reference)
    save_mask(out / 'intercondylar_removed_bone_QC.nii.gz',
              final_mask & bone, reference)
    base_res = base / 'jointspace_residual_volume_v3_2'
    intersection_stats = {}
    qc_multilabel = np.zeros(image.shape, dtype=np.uint8)
    for label, side in enumerate(('medial', 'lateral'), 1):
        residual = read_mask(base_res / f'{side}_residual_space.nii.gz', reference)
        cut = bone_excluded & residual
        save_mask(out / f'{side}_candidate_exclusion_QC.nii.gz', cut, reference)
        if np.any(qc_multilabel[cut] != 0):
            raise RuntimeError('Overlap between medial and lateral residual masks')
        qc_multilabel[cut] = label
        intersection_stats[side] = {'residual_voxels': int(residual.sum()),
                                    'potential_removal_voxels': int(cut.sum()),
                                    'potential_removal_fraction': float(cut.sum()/max(residual.sum(),1))}
        say(f'[QC] {side}: {cut.sum():,} voxels intersect residual; fraction={intersection_stats[side]["potential_removal_fraction"]:.2%}')
    save_mask(out / 'potential_exclusion_multilabel_QC.nii.gz', qc_multilabel, reference)
    meta = {'status': 'review_required', 'source_prompts': str(prompt_file),
            'positive_zyx': [list(x) for x in positives],
            'negative_zyx': [list(x) for x in negatives],
            'model_dir': str(args.model_dir),
            'positive_only_voxels': int(positive_mask.sum()),
            'final_raw_voxels': int(final_mask.sum()),
            'bone_excluded_voxels': int(bone_excluded.sum()),
            'bone_removed_voxels': int((final_mask & bone).sum()),
            'qc_intersection': intersection_stats,
            'note': 'The soft-tissue segmentation and potential exclusions require visual verification. No RTJSV changed.'}
    (out / 'inference_qc_report.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    say('[DONE] QC-only inference saved. Verify final mask and potential_exclusion_multilabel_QC in Slicer before subtract.')


def subtract(args, base, out, reference):
    if not args.confirm_reviewed:
        raise RuntimeError('--mode subtract requires --confirm-reviewed AFTER Slicer inspection of the predicted and removal masks')
    soft = read_mask(out / 'intercondylar_soft_tissue_bone_excluded.nii.gz', reference)
    base_res = base / 'jointspace_residual_volume_v3_2'
    destination = base / 'jointspace_residual_volume_v3_2_IC_excluded_v1_1'
    destination.mkdir(parents=True, exist_ok=True)
    voxel_ml = float(np.prod(reference.GetSpacing())) / 1000.
    total = np.zeros(soft.shape, dtype=np.uint8)
    removed_labels = np.zeros_like(total)
    rows = []
    for label, side in enumerate(('medial', 'lateral'), 1):
        residual = read_mask(base_res / f'{side}_residual_space.nii.gz', reference)
        removed = residual & soft
        corrected = residual & ~soft
        save_mask(destination / f'{side}_residual_corrected.nii.gz', corrected, reference)
        save_mask(destination / f'{side}_removed_intercondylar_QC.nii.gz', removed, reference)
        if np.any(total[corrected] != 0) or np.any(removed_labels[removed] != 0):
            raise RuntimeError('Compartment masks overlap')
        total[corrected] = label
        removed_labels[removed] = label
        n0, nr, n1 = int(residual.sum()), int(removed.sum()), int(corrected.sum())
        assert n0 == nr + n1
        rows.append(dict(compartment=side, original_voxels=n0, removed_voxels=nr,
                         corrected_voxels=n1, original_ml=n0*voxel_ml,
                         removed_ml=nr*voxel_ml, corrected_ml=n1*voxel_ml,
                         removed_fraction=nr/max(n0,1)))
        say(f'[RTJSV] {side}: {n0*voxel_ml:.4f} -> {n1*voxel_ml:.4f} mL; removed {nr*voxel_ml:.4f} mL')
    save_mask(destination / 'residual_corrected_multilabel.nii.gz', total, reference)
    save_mask(destination / 'removed_intercondylar_multilabel_QC.nii.gz', removed_labels, reference)
    with (destination / 'IC_correction_summary.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (destination / 'IC_correction_summary.json').write_text(json.dumps({
        'source_residual': str(base_res), 'source_soft_tissue': str(out / 'intercondylar_soft_tissue_bone_excluded.nii.gz'),
        'formula': 'corrected_residual = original_residual & ~intercondylar_soft_tissue',
        'reviewed_by_user_flag': True,
        'warning': 'Exploratory proxy, NOT validated cartilage volume. Review subtractions and record QC decision.',
        'rows': rows}, indent=2), encoding='utf-8')
    say(f'[DONE] Corrected results saved to {destination}; original v3.2 results untouched')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['infer','subtract'], default='infer')
    p.add_argument('--case-dir', type=Path, default=DEFAULT_CASE)
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--prompts', type=Path, help='point_selection_v1/prompts.json')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--use-negatives', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--confirm-reviewed', action='store_true')
    args = p.parse_args()
    base = args.case_dir / 'fusion_output'
    out = args.output_dir or (base / 'intercondylar_soft_tissue/nninteractive_inference_v1_1')
    out.mkdir(parents=True, exist_ok=True)
    mri_file = base / 'PD_FS_triview_fused_0p4mm.nii.gz'
    if not mri_file.is_file():
        raise FileNotFoundError(mri_file)
    reference = sitk.ReadImage(str(mri_file))
    if reference.GetDimension() != 3:
        raise RuntimeError('MRI must be 3D')
    if args.mode == 'infer':
        infer(args, base, out, reference)
    else:
        subtract(args, base, out, reference)

if __name__ == '__main__':
    main()
