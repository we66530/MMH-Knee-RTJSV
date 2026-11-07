#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTJSV = joint_space ROI - (meniscus union compartment high_signal candidates).
The result is NOT a directly measured cartilage volume.
Outputs separate masks, an overlap-safe two-label map, and CSV/JSON statistics.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4, rtol=0)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0))


def load_mask(path, reference, allow_resample=False):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = sitk.ReadImage(str(path))
    if not same_geometry(im, reference):
        if not allow_resample:
            raise RuntimeError(f'Geometry mismatch: {path}. Align in physical space before allowing nearest-neighbor resampling.')
        print(f'[RESAMPLE] {path}', flush=True)
        im = sitk.Resample(im, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def write_mask(path, ref, data):
    out = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path), True)
    print(f'[SAVED] {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, type=Path, help='Case fusion_output directory')
    p.add_argument('--roi-dir', type=Path, help='Override compartment ROI directory')
    p.add_argument('--out', type=Path, help='Override output directory')
    p.add_argument('--allow-resample', action='store_true', help='Only for already aligned physical spaces')
    a = p.parse_args()
    base = a.base.expanduser().resolve()
    roi = a.roi_dir or (base / 'compartment_roi_meniscus_v3_2')
    men = base / 'meniscus_guided_nninteractive' / 'nninteractive_inference'
    out = a.out or (base / 'jointspace_residual_volume_v3_2')
    out.mkdir(parents=True, exist_ok=True)
    ref = sitk.ReadImage(str(base / 'PD_FS_triview_fused_0p4mm.nii.gz'))
    if ref.GetDimension() != 3:
        raise RuntimeError('Expected a 3D reference image')
    voxel_ml = float(np.prod(ref.GetSpacing())) / 1000.
    combined = np.zeros(tuple(reversed(ref.GetSize())), dtype=np.uint8)
    rows = []
    ownership = np.zeros_like(combined, dtype=bool)
    for label, side in enumerate(('medial', 'lateral'), 1):
        joint = load_mask(roi / f'{side}_jointspace_roi.nii.gz', ref, a.allow_resample)
        mens = load_mask(men / f'{side}_meniscus.nii.gz', ref, a.allow_resample)
        high = load_mask(roi / f'{side}_high_signal.nii.gz', ref, a.allow_resample)
        if not joint.any():
            raise RuntimeError(f'{side} joint-space ROI is empty')
        mens_in = mens & joint
        high_in = high & joint
        overlapping = mens_in & high_in
        excluded = mens_in | high_in
        residual = joint & ~excluded
        if int(joint.sum()) != int(excluded.sum() + residual.sum()):
            raise RuntimeError('Volume-accounting failure')
        roi_overlap = residual & ownership
        if roi_overlap.any():
            raise RuntimeError(f'Residual ROIs overlap at {int(roi_overlap.sum())} voxels; examine compartment masks')
        ownership |= residual
        combined[residual] = label
        write_mask(out / f'{side}_residual_space.nii.gz', ref, residual)
        write_mask(out / f'{side}_excluded_region.nii.gz', ref, excluded)
        write_mask(out / f'{side}_meniscus_within_roi.nii.gz', ref, mens_in)
        write_mask(out / f'{side}_high_signal_within_roi.nii.gz', ref, high_in)
        n_joint = int(joint.sum()); n_ex = int(excluded.sum()); n_res = int(residual.sum())
        row = dict(compartment=side, voxel_volume_mm3=voxel_ml*1000,
                   joint_voxels=n_joint, meniscus_in_roi_voxels=int(mens_in.sum()),
                   high_signal_in_roi_voxels=int(high_in.sum()),
                   meniscus_fluid_overlap_voxels=int(overlapping.sum()),
                   excluded_union_voxels=n_ex, residual_voxels=n_res,
                   joint_volume_ml=round(n_joint*voxel_ml,6),
                   meniscus_in_roi_ml=round(int(mens_in.sum())*voxel_ml,6),
                   high_signal_in_roi_ml=round(int(high_in.sum())*voxel_ml,6),
                   excluded_union_ml=round(n_ex*voxel_ml,6),
                   residual_volume_ml=round(n_res*voxel_ml,6),
                   residual_fraction=round(n_res/n_joint,6),
                   meniscus_outside_roi_voxels=int((mens & ~joint).sum()),
                   high_signal_outside_roi_voxels=int((high & ~joint).sum()))
        rows.append(row)
        print(f'[RTJSV] {side:7s} joint={row["joint_volume_ml"]:.4f} mL, '
              f'meniscus={row["meniscus_in_roi_ml"]:.4f}, '
              f'high-signal={row["high_signal_in_roi_ml"]:.4f}, '
              f'residual={row["residual_volume_ml"]:.4f} mL '
              f'({row["residual_fraction"]:.1%})', flush=True)
        if row['meniscus_in_roi_voxels'] == 0:
            print(f'[INFO] {side}: meniscus overlap is zero, expected for an INNER-edge footprint ROI')
    write_mask(out / 'residual_space_multilabel.nii.gz', ref, combined)
    with (out / 'jointspace_volume_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = dict(metric='Residual Tibiofemoral Joint-Space Volume (RTJSV)',
                   roi_source=str(roi), roi_definition='meniscus INNER-edge cavity plus anterior/posterior tip chord',
                   formula='J \\ (M union H)',
                   warning='Residual joint space is NOT direct cartilage volume. High signal denotes fluid-like candidates, not verified synovial fluid.',
                   fluid_source=str(roi / '{side}_high_signal.nii.gz'),
                   volume_unit='mL', rows=rows,
                   total_residual_volume_ml=round(sum(r['residual_volume_ml'] for r in rows),6))
    (out / 'jointspace_volume_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[DONE] {out}')

if __name__ == '__main__':
    main()
