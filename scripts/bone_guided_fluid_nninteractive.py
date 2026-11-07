#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bone-guided tibiofemoral fluid-like signal prompting, nnInteractive, and merging.

Modes: points, infer, merge. Run points first and REVIEW candidates/points in Slicer.
All arrays/point indices use SimpleITK NumPy Z,Y,X order. Physical positions are LPS.
High PD-FS signal is only a *fluid candidate*, not proof of synovial fluid.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

BASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513/fusion_output')
MRI = BASE / 'PD_FS_triview_fused_0p4mm.nii.gz'
BONES = BASE / 'bone_guided_nninteractive'
MEN = BASE / 'meniscus_guided_nninteractive'
BORDER_JSON = MEN / 'oblique_boundary_v12/prompts_with_negative.json'
MODEL = Path.home() / '.nninteractive/models/nnInteractive_v1.0'
OUT = MEN / 'fluid_nninteractive'


def say(*args):
    print(*args, flush=True)


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = sitk.ReadImage(str(path))
    if im.GetDimension() != 3:
        raise ValueError(f'Not 3D: {path}')
    return im


def matched(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5))


def load_mask(path, reference, allow_resample=False):
    im = read_image(path)
    if not matched(im, reference):
        if not allow_resample:
            raise RuntimeError(f'Geometry mismatch: {path}; verify registration, then add --allow-resample')
        say('[RESAMPLE]', path.name)
        im = sitk.Resample(im, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def save(arr, reference, path):
    im = sitk.GetImageFromArray(arr.astype(np.uint8))
    im.CopyInformation(reference)
    sitk.WriteImage(im, str(path), True)
    say('[SAVED]', path.name)


def affine_zyx(im):
    direction = np.asarray(im.GetDirection(), float).reshape(3, 3)
    matrix = direction @ np.diag(im.GetSpacing()) @ np.eye(3)[:, ::-1]
    return np.asarray(im.GetOrigin(), float), matrix


def physical(points_zyx, origin, matrix):
    return origin + np.asarray(points_zyx, float) @ matrix.T


def sample_phys(mask, origin, matrix, max_count=150000):
    pts = np.argwhere(mask)
    if pts.shape[0] > max_count:
        pts = pts[::int(np.ceil(pts.shape[0] / max_count))]
    if len(pts) == 0:
        raise ValueError('Empty bone mask')
    return physical(pts, origin, matrix)


def joint_geometry(bones, origin, matrix):
    tibia = sample_phys(bones['tibia'], origin, matrix)
    femur = sample_phys(bones['femur'], origin, matrix)
    fibula = sample_phys(bones['fibula'], origin, matrix)
    ttop = float(np.percentile(tibia[:, 2], 99.2))
    fbottom = float(np.percentile(femur[:, 2], 0.8))
    if abs(ttop - fbottom) > 35:
        raise RuntimeError('Femur and tibia appear misaligned: implausible joint gap')
    proximal = tibia[(tibia[:, 2] > ttop - 22) & (tibia[:, 2] < ttop - 3)]
    if len(proximal) < 50:
        raise RuntimeError('Insufficient proximal tibial mask')
    center = np.median(proximal[:, :2], axis=0)
    fib_top = fibula[fibula[:, 2] >= np.percentile(fibula[:, 2], 85)]
    lateral = np.median(fib_top[:, :2], axis=0) - center
    lateral /= max(np.linalg.norm(lateral), 1e-6)
    ap = np.array([-lateral[1], lateral[0]])
    return {'joint_z': 0.5 * (ttop + fbottom), 'tibia_center': center,
            'lateral': lateral, 'ap': ap, 'tibia_top': ttop, 'femur_bottom': fbottom}


def crop_joint(bones, geo, origin, matrix, shape, spacing_zyx, margin_mm=18):
    pts = []
    for name in ('femur', 'tibia'):
        vox = np.argwhere(bones[name])
        if len(vox) > 1800000:
            vox = vox[::2]
        z = origin[2] + vox @ matrix[2]
        near = vox[np.abs(z - geo['joint_z']) < 27]
        if not len(near):
            raise ValueError('No bone found near estimated joint plane')
        pts.append(near)
    pts = np.concatenate(pts)
    margin = np.ceil(margin_mm / spacing_zyx).astype(int)
    lo = np.maximum(pts.min(axis=0) - margin, 0)
    hi = np.minimum(pts.max(axis=0) + margin + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), lo


def geometry_arrays(shape, lower, origin, matrix):
    z, y, x = np.ogrid[lower[0]:lower[0] + shape[0],
                       lower[1]:lower[1] + shape[1],
                       lower[2]:lower[2] + shape[2]]
    return tuple(origin[i] + matrix[i, 0]*z + matrix[i, 1]*y + matrix[i, 2]*x for i in range(3))


def oblique_boundary_values(border, geo, signed, ap_coord, phys_z, outer_margin):
    """Approximate AP-extruded oblique side boundaries from v1.2 anchors.

    v1.2 stores separate boundaries per side/zone but no full surface.
    Use the median line per side to avoid discontinuities across zone transitions.
    If none exists for one side, fail rather than quietly dropping the boundary.
    """
    boundaries = border.get('boundaries', [])
    permitted = np.ones(signed.shape, dtype=bool)
    details = {}
    for side, sign in [('medial', -1), ('lateral', 1)]:
        entries = [b for b in boundaries if b.get('side') == side]
        if not entries:
            raise ValueError(f'No {side} oblique boundary in {BORDER_JSON}')
        # Anchors are [LPS-Z, outward signed-distance (mm)].
        zt, st = np.median(np.array([e['tibia_anchor'] for e in entries], float), axis=0)
        zf, sf = np.median(np.array([e['femur_anchor'] for e in entries], float), axis=0)
        if abs(zf - zt) < 2:
            raise ValueError(f'Invalid {side} oblique boundary anchors')
        boundary = st + (sf - st) * (phys_z - zt) / (zf - zt)
        outside = sign * signed - boundary
        side_region = (sign * signed > 0)
        permitted &= (~side_region) | (outside <= outer_margin)
        details[side] = {'tibia_anchor': [float(zt), float(st)],
                         'femur_anchor': [float(zf), float(sf)]}
    return permitted, details


def select_fluid_candidates(image_crop, bone_crop, meniscus_crop, joint_crop, spacing_zyx,
                            args):
    finite = np.isfinite(image_crop)
    valid = joint_crop & (~bone_crop) & (~meniscus_crop) & finite
    if valid.sum() < 500:
        raise RuntimeError('Too few candidate voxels: joint space/boundary too restrictive')
    # Values are thresholded only within the anatomical ROI; no rescaling of model input.
    smoothed = ndi.gaussian_filter(np.nan_to_num(image_crop, nan=0.0), sigma=0.8)
    t_high = float(np.percentile(smoothed[valid], args.high_percentile))
    t_low = float(np.percentile(smoothed[valid], args.low_percentile))
    support = valid & (smoothed >= t_low)
    seeds = valid & (smoothed >= t_high)
    # 6 connectivity limits diagonal-only bridges.
    conn = ndi.generate_binary_structure(3, 1)
    support_labels, count = ndi.label(support, structure=conn)
    if not count:
        return np.zeros_like(valid), [], {'high': t_high, 'low': t_low}
    touched = np.unique(support_labels[seeds])
    touched = touched[touched > 0]
    grown = np.isin(support_labels, touched)
    labels, n = ndi.label(grown, structure=conn)
    voxel_mm3 = float(np.prod(spacing_zyx))
    components = []
    if n:
        slices = ndi.find_objects(labels)
        for lab, bbox in enumerate(slices, 1):
            if bbox is None:
                continue
            local = labels[bbox] == lab
            volume = int(local.sum()) * voxel_mm3
            if volume < args.min_pool_mm3:
                continue
            components.append((lab, bbox, volume))
    components.sort(key=lambda c: -c[2])
    return grown, components[:args.max_pools], {'high': t_high, 'low': t_low, 'components_found': n,
                                                'components_over_min_volume': len(components)}


def write_fcsv(points, path):
    with open(path, 'w', newline='', encoding='utf8') as f:
        f.write('# Markups fiducial file version = 4.11\n# CoordinateSystem = 1\n')
        f.write('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')
        w = csv.writer(f)
        for idx, p in enumerate(points, 1):
            w.writerow([f'fid{idx}', *p['lps_mm'], 0, 0, 0, 1, 1, 1, 0,
                        p['label'], p.get('description', ''), ''])


def prepare_points(ref, args):
    origin, matrix = affine_zyx(ref)
    spacing = np.asarray(ref.GetSpacing()[::-1], float)
    bones = {k: load_mask(BONES / f'{k}_nninteractive.nii.gz', ref, args.allow_resample)
             for k in ('femur', 'tibia', 'fibula', 'patella')}
    menisci = [MEN / 'nninteractive_inference/medial_meniscus.nii.gz',
               MEN / 'nninteractive_inference/lateral_meniscus.nii.gz']
    if not all(p.exists() for p in menisci):
        raise FileNotFoundError('Both medial/lateral nnInteractive meniscus masks are required')
    men = np.zeros(tuple(reversed(ref.GetSize())), dtype=bool)
    for path in menisci:
        men |= load_mask(path, ref, args.allow_resample)
    geo = joint_geometry(bones, origin, matrix)
    sl, lo = crop_joint(bones, geo, origin, matrix, np.array(men.shape), spacing)
    img = sitk.GetArrayFromImage(ref)[sl].astype(np.float32)
    bone = np.zeros(img.shape, dtype=bool)
    for b in bones.values():
        bone |= b[sl]
    men = men[sl]
    pf, pt = ndi.distance_transform_edt(~bones['femur'][sl], sampling=spacing), \
             ndi.distance_transform_edt(~bones['tibia'][sl], sampling=spacing)
    px, py, pz = geometry_arrays(img.shape, lo, origin, matrix)
    center, lat, ap_axis = geo['tibia_center'], geo['lateral'], geo['ap']
    signed = (px - center[0]) * lat[0] + (py - center[1]) * lat[1]
    ap_coord = (px - center[0]) * ap_axis[0] + (py - center[1]) * ap_axis[1]
    with open(BORDER_JSON, encoding='utf8') as f:
        border = json.load(f)
    within, boundary_details = oblique_boundary_values(border, geo, signed, ap_coord, pz, args.outer_margin_mm)
    joint = ((pf < args.femur_distance) & (pt < args.tibia_distance)
             & (np.abs(pz - geo['joint_z']) <= args.joint_half_height)
             & (np.abs(ap_coord) <= args.ap_half_width)
             & (np.abs(signed) <= args.lateral_half_width)
             & within)
    # Remove near-bone voxels; keep clear of the existing meniscus predictions.
    bone_safe = ndi.binary_dilation(bone, iterations=max(1, round(args.bone_clearance_mm / min(spacing))))
    men_safe = ndi.binary_dilation(men, iterations=max(1, round(args.meniscus_clearance_mm / min(spacing))))
    grown, components, thresholds = select_fluid_candidates(img, bone_safe, men_safe, joint, spacing, args)
    say('[JOINT]', int(joint.sum()), 'voxels; [HIGH SIGNAL]', int(grown.sum()), 'voxels', thresholds)
    labels, _ = ndi.label(grown, structure=ndi.generate_binary_structure(3, 1))
    pool_points = []
    bone_interiors = {}
    for name in ('femur', 'tibia'):
        bmask = bones[name][sl]
        dep = ndi.distance_transform_edt(bmask, sampling=spacing)
        bone_interiors[name] = np.argwhere(dep >= args.min_bone_negative_depth_mm)
    core_centers = []
    for lab, bbox, vol in components:
        blob = labels[bbox] == lab
        # Distance to component boundary; high-signal core is preferred.
        dist = ndi.distance_transform_edt(np.pad(blob, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        sm = ndi.gaussian_filter(np.nan_to_num(img[bbox], nan=0), sigma=0.8)
        threshold = thresholds['high']
        core = blob & (sm >= threshold)
        safe = core & (dist >= args.min_seed_depth_mm)
        if not safe.any():
            say(f'[SKIP] component {lab}: no safe high-signal core')
            continue
        # Rank by interior depth first; use brightness only for tie-breaking.
        rank = np.where(safe, dist + 0.10 * (sm - threshold) / max(abs(threshold), 1), -np.inf)
        loc = np.asarray(np.unravel_index(np.argmax(rank), rank.shape))
        global_zyx = np.array([s.start for s in bbox]) + loc + lo
        lps = physical(global_zyx, origin, matrix)
        uid = f'pool_{len(pool_points)+1:03d}'
        bone_negatives = []
        seed_local = global_zyx - lo
        for name in ('tibia', 'femur'):
            bone_pts = bone_interiors[name]
            if not len(bone_pts):
                continue
            delta = (bone_pts - seed_local) * spacing
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            pos = int(np.argmin(distance))
            if distance[pos] <= args.bone_negative_max_distance_mm:
                bone_negatives.append((bone_pts[pos] + lo).astype(int).tolist())
        pool_points.append({'id': uid, 'label': uid + '_positive', 'voxel_zyx': global_zyx.tolist(),
                            'lps_mm': lps.tolist(), 'candidate_volume_mm3': float(vol),
                            'seed_depth_mm': float(dist[tuple(loc)]), 'source_component_label': int(lab),
                            'bone_negative_zyx': bone_negatives})
        core_centers.append(global_zyx)
        say('[SEED]', uid, tuple(global_zyx), f'{vol:.1f} mm3')
    OUT.mkdir(parents=True, exist_ok=True)
    # Save the full ROI candidate for visual review without allocating more full volumes than needed.
    full = np.zeros(tuple(reversed(ref.GetSize())), dtype=np.uint8)
    full[sl] = joint.astype(np.uint8)
    save(full, ref, OUT / 'joint_space_roi.nii.gz')
    full[sl] = grown.astype(np.uint8)
    save(full, ref, OUT / 'high_signal_candidates.nii.gz')
    full.fill(0)
    for i, p in enumerate(pool_points, 1):
        z, y, x = p['voxel_zyx']
        full[z, y, x] = i if i < 256 else 255
    save(full, ref, OUT / 'seed_voxels.nii.gz')
    with open(OUT / 'pool_prompts.json', 'w', encoding='utf8') as f:
        json.dump({'MRI': str(MRI), 'shape_zyx': list(ref.GetSize()[::-1]),
                   'parameters': vars(args), 'thresholds': thresholds,
                   'boundary': boundary_details, 'pools': pool_points,
                   'warning': 'High-signal regions are candidate fluid only; inspect before inference.'},
                  f, indent=2, default=str)
    write_fcsv(pool_points, OUT / 'fluid_positive_points_LPS.fcsv')
    say('[DONE POINTS]', len(pool_points), 'potential pools; review in Slicer')
    return pool_points


def create_session(ref, args):
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available')
    if not (args.model_dir / 'fold_0/checkpoint_final.pth').exists():
        raise FileNotFoundError(f'Checkpoint missing in {args.model_dir}')
    session = nnInteractiveInferenceSession(device=torch.device('cuda:0'), use_torch_compile=False,
                                            verbose=False, torch_n_threads=min(os.cpu_count() or 4, 16),
                                            do_autozoom=True)
    session.initialize_from_trained_model_folder(str(args.model_dir))
    array = sitk.GetArrayFromImage(ref)
    session.set_image(np.ascontiguousarray(array[None]))
    target = torch.zeros(array.shape, dtype=torch.uint8)
    session.set_target_buffer(target)
    return session, target


def do_infer(ref, args):
    prompt_path = OUT / 'pool_prompts.json'
    if not prompt_path.exists():
        raise FileNotFoundError('Run --mode points first')
    data = json.loads(prompt_path.read_text())
    if list(ref.GetSize()[::-1]) != data['shape_zyx']:
        raise RuntimeError('MRI shape changed after point selection')
    pools = data['pools']
    if not pools:
        raise RuntimeError('No candidate pools; review ROI/thresholds')
    session, target = create_session(ref, args)
    stats = []
    for p in pools:
        uid = p['id']
        point = tuple(map(int, p['voxel_zyx']))
        say('[INFER]', uid, 'positive:', point)
        session.reset_interactions()
        session.add_point_interaction(point, include_interaction=True)
        # Save positive-only to facilitate comparison.
        positive = target.numpy().copy() > 0
        save(positive, ref, OUT / f'{uid}_positive_only.nii.gz')
        # Negative prompts from already reviewed oblique boundary JSON, nearest same-side
        # points only. They are NOT derived from other candidate fluid pools.
        negatives = [tuple(map(int, q)) for q in p.get('bone_negative_zyx', [])]
        if args.use_boundary_negatives and BORDER_JSON.exists():
            boundary = json.loads(BORDER_JSON.read_text())
            candidates = boundary.get('negative', [])
            # Determine medial/lateral from lateral signed coordinate in saved geometry.
            # Use the nearest 2 in physical distance, because the existing v1.2 negatives
            # were generated for meniscal segmentation, not fluid.
            pp = np.asarray(p['lps_mm'], float)
            candidates = sorted(candidates, key=lambda q: np.linalg.norm(np.asarray(q['lps_mm']) - pp))
            for q in candidates:
                qp = tuple(q['voxel_zyx'])
                if np.linalg.norm(np.asarray(q['lps_mm']) - pp) < args.negative_max_distance_mm:
                    if qp != point and qp not in negatives:
                        negatives.append(qp)
                if len(negatives) == args.max_negative_per_pool:
                    break
        for neg in negatives:
            say('   [NEG]', neg)
            session.add_point_interaction(neg, include_interaction=False)
        final = target.numpy().copy() > 0
        save(final, ref, OUT / f'{uid}_raw.nii.gz')
        stats.append({'id': uid, 'positive_zyx': list(point), 'negative_zyx': [list(n) for n in negatives],
                      'positive_voxels': int(positive.sum()), 'raw_voxels': int(final.sum())})
    (OUT / 'inference_stats.json').write_text(json.dumps(stats, indent=2))
    say('[DONE INFER]', len(stats), 'raw segmentations')


def do_merge(ref, args):
    data = json.loads((OUT / 'pool_prompts.json').read_text())
    pools = data['pools']
    if not pools:
        raise RuntimeError('No pool prompts')
    shape = tuple(reversed(ref.GetSize()))
    union = np.zeros(shape, dtype=bool)
    raw_stats = []
    for p in pools:
        path = OUT / f"{p['id']}_raw.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        pred = load_mask(path, ref)
        union |= pred
        raw_stats.append({'id': p['id'], 'volume_mm3': int(pred.sum()) * float(np.prod(ref.GetSpacing()))})
    # Union all overlapping masks, then split strictly by 6-connected components.
    # No dilation/closing: avoids inventing bridges between physically separate pools.
    conn = ndi.generate_binary_structure(3, 1)
    labels, count = ndi.label(union, structure=conn)
    voxel_mm3 = float(np.prod(ref.GetSpacing()))
    sizes = np.bincount(labels.ravel())
    keep = [(i, int(sizes[i])) for i in range(1, count + 1)
            if sizes[i] * voxel_mm3 >= args.min_final_pool_mm3]
    keep.sort(key=lambda q: -q[1])
    if len(keep) > 65535:
        raise RuntimeError('Too many output labels')
    result = np.zeros(shape, dtype=np.uint16)
    report = []
    for new_label, (old, size) in enumerate(keep, 1):
        result[labels == old] = new_label
        report.append({'label': new_label, 'volume_mm3': size * voxel_mm3, 'voxel_count': size})
    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(ref)
    sitk.WriteImage(out_img, str(OUT / 'fluid_pools_multilabel.nii.gz'), True)
    (OUT / 'merge_report.json').write_text(json.dumps({'raw': raw_stats, 'final_components': report,
        'rule': 'union of predictions then 6-connected components; no bridging dilation'}, indent=2))
    say('[DONE MERGE]', len(report), 'final connected components')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['points', 'infer', 'merge'], default='points')
    parser.add_argument('--allow-resample', action='store_true')
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--high-percentile', type=float, default=90)
    parser.add_argument('--low-percentile', type=float, default=75)
    parser.add_argument('--min-pool-mm3', type=float, default=10)
    parser.add_argument('--max-pools', type=int, default=12)
    parser.add_argument('--min-seed-depth-mm', type=float, default=0.8)
    parser.add_argument('--femur-distance', type=float, default=14)
    parser.add_argument('--tibia-distance', type=float, default=10)
    parser.add_argument('--joint-half-height', type=float, default=12)
    parser.add_argument('--ap-half-width', type=float, default=43)
    parser.add_argument('--lateral-half-width', type=float, default=53)
    parser.add_argument('--outer-margin-mm', type=float, default=2)
    parser.add_argument('--bone-clearance-mm', type=float, default=0.8)
    parser.add_argument('--meniscus-clearance-mm', type=float, default=0.8)
    parser.add_argument('--min-bone-negative-depth-mm', type=float, default=2)
    parser.add_argument('--bone-negative-max-distance-mm', type=float, default=32)
    parser.add_argument('--use-boundary-negatives', action='store_true', help='Optional additional meniscus-era boundary negative points; inspect carefully')
    parser.add_argument('--negative-max-distance-mm', type=float, default=18)
    parser.add_argument('--max-negative-per-pool', type=int, default=2)
    parser.add_argument('--min-final-pool-mm3', type=float, default=10)
    args = parser.parse_args()
    if not (0 < args.low_percentile < args.high_percentile < 100):
        parser.error('Require 0 < low-percentile < high-percentile < 100')
    if args.max_pools < 1:
        parser.error('--max-pools must be >= 1')
    OUT.mkdir(parents=True, exist_ok=True)
    ref = read_image(MRI)
    say('[MRI]', MRI, 'size:', ref.GetSize(), 'spacing:', ref.GetSpacing())
    if args.mode == 'points':
        prepare_points(ref, args)
    elif args.mode == 'infer':
        do_infer(ref, args)
    else:
        do_merge(ref, args)


if __name__ == '__main__':
    main()
