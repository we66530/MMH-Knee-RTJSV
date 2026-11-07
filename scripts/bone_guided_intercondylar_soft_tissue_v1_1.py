#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anatomy-guided intercondylar soft-tissue prompt proposals (POINTS ONLY).

Reads the v3.2 meniscus inner-footprint metadata (horn-tip chords) and the
existing nnInteractive bone/meniscus masks. Proposes 2-4 positive points in
central/notch-associated low-to-intermediate signal soft tissue, plus
conservative negative points. DOES NOT run nnInteractive or subtract RTJSV.

The output is a hypothesis requiring review in 3D Slicer: intensity is NOT a
validated tissue classifier. In particular, this is not an ACL/PCL diagnosis.

Usage:
  conda activate mricore
  python ~/bone_guided_intercondylar_soft_tissue.py --mode points
  python ~/bone_guided_intercondylar_soft_tissue.py --case-dir /path/to/case \\
      --max-positive 4 --inner-extension-mm 5 --height-above-joint-mm 22
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

DEFAULT_CASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513')


def info(message):
    print(message, flush=True)


def image_read(path: Path, reference=None, allow_resample=False):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    im = sitk.ReadImage(str(path))
    if im.GetDimension() != 3:
        raise ValueError(f'Expected 3D image: {path}')
    if reference is not None:
        same = (im.GetSize() == reference.GetSize()
                and np.allclose(im.GetSpacing(), reference.GetSpacing(), atol=1e-4, rtol=0)
                and np.allclose(im.GetOrigin(), reference.GetOrigin(), atol=1e-3, rtol=0)
                and np.allclose(im.GetDirection(), reference.GetDirection(), atol=1e-5, rtol=0))
        if not same:
            if not allow_resample:
                raise RuntimeError(f'Geometry mismatch: {path}\nVerify alignment first; then use --allow-resample.')
            im = sitk.Resample(im, reference, sitk.Transform(),
                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return im


def save_crop_mask(crop, slices, ref, path):
    full = np.zeros(tuple(reversed(ref.GetSize())), dtype=np.uint8)
    full[slices] = crop.astype(np.uint8)
    image = sitk.GetImageFromArray(full)
    image.CopyInformation(ref)
    sitk.WriteImage(image, str(path), True)
    info(f'[SAVE] {path.name}: {int(crop.sum()):,} voxels')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n < 1e-8:
        raise ValueError('Degenerate anatomical vector')
    return x / n


def geometry(image):
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    sp = np.asarray(image.GetSpacing())
    return np.asarray(image.GetOrigin()), direction @ np.diag(sp) @ np.eye(3)[:, ::-1]


def save_fcsv(points, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        f.write('# Markups fiducial file version = 4.11\n')
        f.write('# CoordinateSystem = 1\n')  # LPS
        f.write('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')
        wr = csv.writer(f)
        for i, p in enumerate(points, 1):
            wr.writerow([f'p{i}', *p['lps_mm'], 0, 0, 0, 1, 1, 1, 0,
                         p['label'], f"score={p['score']:.3f}", ''])


def save_preview(mri, positives, negatives, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        info('[WARN] matplotlib unavailable; skipping PNG preview')
        return
    points = positives + negatives
    if not points:
        return
    fig, axes = plt.subplots(len(points), 3, figsize=(12, max(3.1 * len(points), 5)),
                             squeeze=False, constrained_layout=True)
    for row, p in enumerate(points):
        z, y, x = p['voxel_zyx']
        for col, (image, cx, cy, title) in enumerate((
                (mri[z, :, :], x, y, 'Array-Z'),
                (mri[:, y, :], x, z, 'Array-Y'),
                (mri[:, :, x], y, z, 'Array-X'))):
            ax = axes[row, col]
            limits = np.percentile(image[np.isfinite(image)], [2, 98])
            ax.imshow(image, cmap='gray', origin='lower', vmin=limits[0],
                      vmax=max(limits[1], limits[0] + 1))
            ax.scatter([cx], [cy], s=75, color='lime' if p['type'] == 'positive' else 'red',
                       marker='+' if p['type'] == 'positive' else 'x', linewidths=2)
            ax.set_xlim(max(0, cx-85), min(image.shape[1]-1, cx+85))
            ax.set_ylim(max(0, cy-85), min(image.shape[0]-1, cy+85))
            ax.set_title(f"{p['label']} / {title}")
    fig.savefig(str(output), dpi=130)
    plt.close(fig)
    info(f'[SAVE] {output.name}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', choices=['points'], default='points')
    p.add_argument('--case-dir', type=Path, default=DEFAULT_CASE)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--allow-resample', action='store_true')
    p.add_argument('--max-positive', type=int, default=4)
    p.add_argument('--max-negative', type=int, default=0)
    p.add_argument('--inner-extension-mm', type=float, default=8.,
                   help='Permit seed candidates to cross from central corridor into the 2 inner footprints')
    p.add_argument('--height-below-joint-mm', type=float, default=7.)
    p.add_argument('--height-above-joint-mm', type=float, default=22.)
    p.add_argument('--ap-margin-mm', type=float, default=8.,
                   help='AP extension beyond horn-tip extrema')
    p.add_argument('--seed-min-bone-distance-mm', type=float, default=0.4)
    p.add_argument('--seed-min-meniscus-distance-mm', type=float, default=0.8)
    p.add_argument('--seed-min-fluid-distance-mm', type=float, default=0.4)
    p.add_argument('--candidate-min-depth-mm', type=float, default=0.8)
    p.add_argument('--min-positive-separation-mm', type=float, default=7.)
    p.add_argument('--min-negative-separation-mm', type=float, default=8.)
    p.add_argument('--min-component-mm3', type=float, default=10.)
    p.add_argument('--intensity-low-percentile', type=float, default=0.)
    p.add_argument('--intensity-high-percentile', type=float, default=75.)
    p.add_argument('--negative-distance-to-seed-mm', type=float, default=6.)
    args = p.parse_args()
    if args.max_positive < 1 or args.max_negative < 0:
        p.error('Invalid point limits')
    if not 0 <= args.intensity_low_percentile < args.intensity_high_percentile <= 100:
        p.error('Intensity percentiles require 0 <= low < high <= 100')

    base = args.case_dir / 'fusion_output'
    out = args.output_dir or (base / 'intercondylar_soft_tissue' / 'point_selection_v1_1')
    out.mkdir(parents=True, exist_ok=True)
    ref = image_read(base / 'PD_FS_triview_fused_0p4mm.nii.gz')
    full_shape = tuple(reversed(ref.GetSize()))
    origin, M = geometry(ref)
    spacing_zyx = np.linalg.norm(M, axis=0)

    metadata_path = base / 'compartment_roi_meniscus_v3_2' / 'roi_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if 'v3_2' not in metadata.get('model', ''):
        raise ValueError(f'Expected v3.2 inner-footprint metadata: {metadata_path}')
    frame = metadata['frame']
    center = np.asarray(frame['center_lps'], dtype=float)
    lateral = unit(frame['lateral_lps'])
    ap = unit(frame['ap_lps'])
    normal = unit(frame['plateau_normal_lps'])
    joint_h = float(frame['joint_h_mm'])
    tips = {side: np.asarray(metadata['footprints'][side]['tips_uv_mm'], dtype=float)
            for side in ('medial', 'lateral')}

    # For each chord, a signed 2D distance > 0 points toward the other meniscus.
    chords = {}
    for side in ('medial', 'lateral'):
        t0, t1 = tips[side]
        direction = unit(t1-t0)
        inward_normal = np.array([-direction[1], direction[0]])
        midpoint = (t0+t1)/2
        other_mid = tips['lateral' if side == 'medial' else 'medial'].mean(axis=0)
        if np.dot(other_mid-midpoint, inward_normal) < 0:
            inward_normal *= -1
        chords[side] = (midpoint, inward_normal)
    chord_sep = float(np.linalg.norm(tips['medial'].mean(axis=0) - tips['lateral'].mean(axis=0)))
    if chord_sep < 5:
        raise RuntimeError('Meniscus tip chords unusually close; inspect metadata')

    # Tight memory use: derive crop from the two joint-space ROIs, then add
    # physical margins. This avoids full-volume 3D distance maps (122M voxels).
    roi_dir = base / 'compartment_roi_meniscus_v3_2'
    r0 = sitk.GetArrayFromImage(image_read(roi_dir / 'medial_jointspace_roi.nii.gz', ref,
                                         allow_resample=args.allow_resample)) > 0
    r1 = sitk.GetArrayFromImage(image_read(roi_dir / 'lateral_jointspace_roi.nii.gz', ref,
                                         allow_resample=args.allow_resample)) > 0
    union_roi = r0 | r1
    if not union_roi.any():
        raise RuntimeError('Empty medial/lateral joint-space ROIs')
    box = ndi.find_objects(union_roi.astype(np.uint8))[0]
    pad = np.ceil(28. / spacing_zyx).astype(int)
    lo = np.maximum([a.start for a in box] - pad, 0)
    hi = np.minimum([a.stop for a in box] + pad, full_shape)
    sl = tuple(slice(int(a), int(b)) for a,b in zip(lo,hi))
    shape = tuple(int(b-a) for a,b in zip(lo,hi))
    info(f'[CROP] start ZYX={lo.tolist()}, end={hi.tolist()}, shape={shape}')
    del union_roi

    mri = sitk.GetArrayFromImage(ref).astype(np.float32)
    crop = mri[sl]
    bones_dir = base / 'bone_guided_nninteractive'
    masks = {}
    for name in ('femur','tibia','fibula','patella'):
        path = bones_dir / f'{name}_nninteractive.nii.gz'
        masks[name] = sitk.GetArrayFromImage(image_read(path, ref, args.allow_resample))[sl] > 0
    mn_dir = base / 'meniscus_guided_nninteractive' / 'nninteractive_inference'
    men = (sitk.GetArrayFromImage(image_read(mn_dir / 'medial_meniscus.nii.gz', ref, args.allow_resample))[sl] > 0) | (
           sitk.GetArrayFromImage(image_read(mn_dir / 'lateral_meniscus.nii.gz', ref, args.allow_resample))[sl] > 0)
    fluid_path = base / 'meniscus_guided_nninteractive' / 'fluid_nninteractive' / 'high_signal_candidates.nii.gz'
    fluid = sitk.GetArrayFromImage(image_read(fluid_path, ref, args.allow_resample))[sl] > 0
    residual = r0[sl] | r1[sl]
    del r0,r1
    bone = np.zeros(shape, bool)
    for m in masks.values(): bone |= m
    del masks

    # Local physical plateau-plane coordinate maps.
    z, y, x = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    u = np.zeros(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    h = np.zeros(shape, dtype=np.float32)
    for i, q in enumerate((lateral, ap, normal)):
        dest = (u, v, h)[i]
        dest[:] = ((origin-center)@q + (M[:,0]@q)*z +
                   (M[:,1]@q)*y + (M[:,2]@q)*x)
    del z,y,x

    # The normal half-planes for each horn chord intersect in the CENTRAL
    # corridor. Allow up to +extension into the inner sampling compartments.
    d0 = (u-chords['medial'][0][0])*chords['medial'][1][0] + (v-chords['medial'][0][1])*chords['medial'][1][1]
    d1 = (u-chords['lateral'][0][0])*chords['lateral'][1][0] + (v-chords['lateral'][0][1])*chords['lateral'][1][1]
    corridor = ((d0 >= -args.inner_extension_mm) & (d1 >= -args.inner_extension_mm))
    ap_min = min(tips['medial'][:,1].min(), tips['lateral'][:,1].min()) - args.ap_margin_mm
    ap_max = max(tips['medial'][:,1].max(), tips['lateral'][:,1].max()) + args.ap_margin_mm
    corridor &= ((v >= ap_min) & (v <= ap_max) &
                 (h >= joint_h-args.height_below_joint_mm) &
                 (h <= joint_h+args.height_above_joint_mm))
    del d0,d1,u,v

    # Distance exclusion prevents seed sitting *on* segmented structures.
    bone_distance = ndi.distance_transform_edt(~bone, sampling=spacing_zyx)
    men_distance = ndi.distance_transform_edt(~men, sampling=spacing_zyx)
    fluid_distance = ndi.distance_transform_edt(~fluid, sampling=spacing_zyx)
    anatomical = (corridor & ~bone & ~men & ~fluid &
                  (bone_distance >= args.seed_min_bone_distance_mm) &
                  (men_distance >= args.seed_min_meniscus_distance_mm) &
                  (fluid_distance >= args.seed_min_fluid_distance_mm))
    valid = anatomical & np.isfinite(crop)
    if np.count_nonzero(valid) < 100:
        raise RuntimeError('Too few safe soft tissue candidates; inspect corridor and masks')
    # Robust moderate/low signal interval inside anatomical ROI.
    smoothed = ndi.gaussian_filter(np.nan_to_num(crop, nan=float(np.median(crop[valid]))), sigma=1)
    low, high = np.percentile(smoothed[valid], [args.intensity_low_percentile,
                                               args.intensity_high_percentile])
    candidate = valid & (smoothed >= low) & (smoothed <= high)
    candidate_depth = ndi.distance_transform_edt(candidate, sampling=spacing_zyx)
    labels, n = ndi.label(candidate, structure=ndi.generate_binary_structure(3,2))
    counts = np.bincount(labels.ravel()); counts[0] = 0
    minvox = max(1, int(np.ceil(args.min_component_mm3/float(np.prod(ref.GetSpacing())))))
    eligible = counts >= minvox; eligible[0] = False
    candidate &= eligible[labels]
    candidate_depth = ndi.distance_transform_edt(candidate, sampling=spacing_zyx)
    safe = candidate & (candidate_depth >= args.candidate_min_depth_mm)
    info(f'[CANDIDATES] central={int(corridor.sum()):,}, anatomical={int(anatomical.sum()):,}, '
         f'low-mid-signal={int(candidate.sum()):,}, safe-core={int(safe.sum()):,}; '
         f'intensity [{low:.2f}, {high:.2f}]')

    # Prioritize internal soft tissue + superior portion near the protrusion.
    # Score is NOT a calibrated probability. Restrict to a band where spikes
    # were observed; select diverse points in 3D rather than forced zones.
    score = (0.56 * np.clip(candidate_depth / 3., 0, 1) +
             0.22 * np.clip(bone_distance / 5., 0, 1) +
             0.12 * np.clip((h-joint_h+2) / 13., 0, 1) +
             0.10 * (1 - np.clip((smoothed-low)/max(high-low,1e-5), 0, 1)))
    score[~safe] = -np.inf
    origin_index = np.array(lo)
    positives = []
    chosen_local = []
    for _ in range(args.max_positive):
        if not np.isfinite(score).any(): break
        idx = np.unravel_index(int(np.argmax(score)), shape)
        # Coordinates in ZYX; physical LPS uses SimpleITK transform.
        global_idx = np.array(idx) + origin_index
        xyz = tuple(int(i) for i in global_idx[::-1])
        lps = np.asarray(ref.TransformIndexToPhysicalPoint(xyz))
        positives.append({'type':'positive', 'label':f'IC_soft_positive_{len(positives)+1:02d}',
                          'voxel_zyx':global_idx.astype(int).tolist(), 'lps_mm':lps.tolist(),
                          'score':round(float(score[idx]),5),
                          'depth_mm':round(float(candidate_depth[idx]),3),
                          'intensity':round(float(crop[idx]),3),
                          'h_relative_joint_mm':round(float(h[idx]-joint_h),3),
                          'on_existing_residual':bool(residual[idx])})
        chosen_local.append(idx)
        # Suppress points too close in physical Euclidean distance.
        rz = np.ceil(args.min_positive_separation_mm / spacing_zyx).astype(int)
        lower = np.maximum(np.asarray(idx)-rz, 0); upper = np.minimum(np.asarray(idx)+rz+1, shape)
        local_sl = tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
        zz,yy,xx = np.ogrid[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
        dist2 = (((zz-idx[0])*spacing_zyx[0])**2 + ((yy-idx[1])*spacing_zyx[1])**2 +
                 ((xx-idx[2])*spacing_zyx[2])**2)
        patch = score[local_sl]
        patch[dist2 < args.min_positive_separation_mm**2] = -np.inf
        info(f'[POS] {positives[-1]["voxel_zyx"]}, depth={positives[-1]["depth_mm"]:.2f} mm')

    if not positives:
        info('[WARNING] No reliable soft-tissue positive found. Review masks; no automatic inference should run.')

    # Negative candidates: pre-existing meniscal or bone tissue, and definite
    # high-signal fluid; require distance from every proposed positive. These
    # are deliberately selected as separate classes for visual QC.
    negative_pool = [('bone', bone), ('meniscus', men), ('fluid', fluid)]
    negatives = []
    for kind, mask in negative_pool:
        if len(negatives) >= args.max_negative: break
        # Restrict negative search to the same corridor (not arbitrary anatomy).
        available = mask & corridor
        if not np.any(available): continue
        depth = ndi.distance_transform_edt(available, sampling=spacing_zyx)
        if kind == 'fluid':
            # Fluid mask can be very thin; use its most interior candidate.
            min_depth = 0.4
        else:
            min_depth = 1.0
        allowed = available & (depth >= min_depth)
        locs = np.argwhere(allowed)
        if not len(locs): continue
        # Avoid scores over the entire 3D volume for each negative.
        if chosen_local:
            pos = np.asarray(chosen_local)
            keep = np.ones(len(locs), dtype=bool)
            for pt in pos:
                d = (locs-pt) * spacing_zyx
                keep &= np.sum(d*d, axis=1) >= args.negative_distance_to_seed_mm**2
            locs = locs[keep]
        if not len(locs): continue
        ids = np.argmax(depth[tuple(locs.T)])
        idx = tuple(int(v) for v in locs[ids])
        global_idx = np.asarray(idx) + origin_index
        lps = np.asarray(ref.TransformIndexToPhysicalPoint(tuple(int(v) for v in global_idx[::-1])))
        negatives.append({'type':'negative','label':f'IC_negative_{kind}',
                          'voxel_zyx':global_idx.astype(int).tolist(), 'lps_mm':lps.tolist(),
                          'score':round(float(depth[idx]),5),'source_mask':kind})
        info(f'[NEG] {kind}: {negatives[-1]["voxel_zyx"]}')

    save_crop_mask(corridor, sl, ref, out/'central_corridor_roi.nii.gz')
    save_crop_mask(candidate, sl, ref, out/'soft_tissue_candidates.nii.gz')
    save_crop_mask(safe, sl, ref, out/'soft_tissue_safe_core.nii.gz')
    points_mask = np.zeros(shape, dtype=np.uint8)
    for pp in positives:
        ijk = np.asarray(pp['voxel_zyx']) - lo
        points_mask[tuple(ijk)] = 1
    for pp in negatives:
        ijk = np.asarray(pp['voxel_zyx']) - lo
        points_mask[tuple(ijk)] = 2
    save_crop_mask(points_mask, sl, ref, out/'prompt_voxels_multilabel.nii.gz')
    save_fcsv(positives, out/'positive_points_LPS.fcsv')
    save_fcsv(negatives, out/'negative_points_LPS.fcsv')
    payload = {'version':'1.1-dark-inclusive','mode':'points','image':str(base/'PD_FS_triview_fused_0p4mm.nii.gz'),
               'reference_roi_metadata':str(metadata_path),
               'definition':'heuristic intercondylar soft-tissue seeds (NOT validated ACL/PCL labels)',
               'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'thresholds':{'low':float(low),'high':float(high)},
               'chord_separation_mm':chord_sep,
               'positive':positives,'negative':negatives,
               'warning':'Inspect all prompt locations in 3D Slicer before nnInteractive. '
                         'Do not blindly subtract the segmentation from cartilage-related volumes.'}
    (out/'prompts.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    save_preview(mri, positives, negatives, out/'point_selection_preview.png')
    info(f'[DONE] {len(positives)} positives, {len(negatives)} negatives; {out}')


if __name__ == '__main__':
    main()
