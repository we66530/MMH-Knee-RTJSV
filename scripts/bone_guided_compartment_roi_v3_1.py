#!/usr/bin/env python3
"""Meniscus-footprint-guided medial/lateral tibiofemoral weight-bearing ROIs.

The medial/lateral boundaries come from each corresponding segmented meniscus:
its outer C-shaped contour plus a straight chord between its two inner horn tips.
The footprints are extruded along an estimated tibial plateau normal and
intersected with a joint-space envelope between the segmented femur and tibia.

This produces sampling ROIs, not validated cartilage/fluid segmentations.

Requirements: numpy scipy SimpleITK matplotlib (for optional QC PNG).

Usage:
 python ~/bone_guided_compartment_roi_v3.py
 python ~/bone_guided_compartment_roi_v3.py --footprint-grid-mm 0.6 --joint-half-height-mm 12
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay

BASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513/fusion_output')


def log(s):
    print(s, flush=True)


def load(path, ref=None, mask=True, allow_resample=False):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = sitk.ReadImage(str(path))
    if ref is not None:
        same = (im.GetSize() == ref.GetSize() and
                np.allclose(im.GetOrigin(), ref.GetOrigin(), atol=1e-3, rtol=0) and
                np.allclose(im.GetSpacing(), ref.GetSpacing(), atol=1e-4, rtol=0) and
                np.allclose(im.GetDirection(), ref.GetDirection(), atol=1e-5, rtol=0))
        if not same:
            if not allow_resample:
                raise RuntimeError(f'Geometry mismatch: {path}; check registration before --allow-resample')
            im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,
                               0, sitk.sitkUInt8 if mask else sitk.sitkFloat32)
    return im


def unit(v):
    v = np.asarray(v, float)
    d = float(np.linalg.norm(v))
    if d < 1e-8:
        raise RuntimeError('Degenerate anatomical vector')
    return v / d


def geometry(im):
    D = np.asarray(im.GetDirection()).reshape(3, 3)
    S = np.asarray(im.GetSpacing())
    return np.asarray(im.GetOrigin()), D @ np.diag(S) @ np.eye(3)[:, ::-1]


def physical_samples(mask, origin, M, max_points=250000):
    pts = np.argwhere(mask)
    if not len(pts):
        raise RuntimeError('Empty segmentation')
    step = max(1, int(np.ceil(len(pts) / max_points)))
    return origin + pts[::step] @ M.T


def axes_from_bones(tibia, fibula):
    top_z = np.percentile(tibia[:, 2], 99)
    proximal = tibia[(tibia[:, 2] >= top_z - 18) & (tibia[:, 2] <= top_z - 3)]
    distal = tibia[(tibia[:, 2] >= top_z - 65) & (tibia[:, 2] <= top_z - 42)]
    if len(proximal) < 100 or len(distal) < 100:
        raise RuntimeError('Insufficient proximal tibial shaft for orientation')
    center = np.median(proximal, axis=0)
    n = unit(np.median(proximal, axis=0) - np.median(distal, axis=0))
    if n[2] < 0:
        n = -n
    fprox = fibula[fibula[:, 2] >= np.percentile(fibula[:, 2], 85)]
    lateral = unit(np.median(fprox, axis=0) - center)
    lateral = unit(lateral - np.dot(lateral, n) * n)
    ap = unit(np.cross(n, lateral))
    if ap[1] < 0:
        ap = -ap  # positive AP toward posterior in LPS
    lateral = unit(lateral - np.dot(lateral, ap) * ap)
    return center, lateral, ap, n


def fit_plateau(tib, center, lateral, ap, n, central_width=8):
    rel = tib - center
    u = rel @ lateral
    v = rel @ ap
    h = rel @ n
    eligible = (np.abs(u) > central_width) & (np.abs(u) < 38) & (np.abs(v) < 35) & (h >= np.percentile(h, 81))
    u, v, h = u[eligible], v[eligible], h[eligible]
    if len(h) < 1000:
        return n, {'status': 'shaft_fallback'}
    # Robust top envelope within each ~3-mm 2D cell.
    ub = np.floor((u + 50) / 3).astype(int)
    vb = np.floor((v + 50) / 3).astype(int)
    ids = ub * 100 + vb
    unique, inverse = np.unique(ids, return_inverse=True)
    maximum = np.full(len(unique), -np.inf)
    np.maximum.at(maximum, inverse, h)
    bu = np.bincount(inverse, weights=u) / np.bincount(inverse)
    bv = np.bincount(inverse, weights=v) / np.bincount(inverse)
    if len(maximum) < 20:
        return n, {'status': 'shaft_fallback_bins'}
    X = np.column_stack([bu, bv, np.ones(len(bu))])
    keep = np.ones(len(maximum), bool)
    for _ in range(5):
        abc, *_ = np.linalg.lstsq(X[keep], maximum[keep], rcond=None)
        residual = maximum - X @ abc
        med = np.median(residual[keep])
        mad = max(np.median(np.abs(residual[keep] - med)) * 1.4826, 1)
        newkeep = np.abs(residual - med) <= 3 * mad
        if newkeep.sum() < 15:
            break
        keep = newkeep
    n_fit = unit(n - abc[0] * lateral - abc[1] * ap)
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(n_fit, n), -1, 1))))
    if tilt > 20:
        return n, {'status': 'shaft_fallback_tilt', 'tilt_deg': tilt}
    return n_fit, {'status': 'fitted', 'tilt_deg': tilt, 'bins': int(len(maximum))}


def largest_component(binary):
    lbl, count = ndi.label(binary, structure=ndi.generate_binary_structure(binary.ndim, 2))
    if not count:
        return np.zeros_like(binary, dtype=bool)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == np.argmax(sizes)


def disk(radius):
    r = int(np.ceil(radius))
    a, b = np.ogrid[-r:r+1, -r:r+1]
    return a*a+b*b <= radius*radius


def line_mask(shape, a, b, radius_px=1):
    mask = np.zeros(shape, bool)
    length = max(2, int(np.ceil(np.linalg.norm(np.asarray(b) - a) * 3)))
    coords = np.round(np.linspace(a, b, length)).astype(int)
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < shape[1])]
    mask[coords[:, 0], coords[:, 1]] = True
    return ndi.binary_dilation(mask, structure=disk(radius_px))


def convex_footprint(binary):
    pts = np.argwhere(binary)
    if len(pts) < 3:
        raise RuntimeError('Not enough meniscus footprint points for convex hull')
    hull = ConvexHull(pts)
    tri = Delaunay(pts[hull.vertices])
    yy, xx = np.indices(binary.shape)
    inside = tri.find_simplex(np.column_stack([yy.ravel(), xx.ravel()])) >= 0
    return inside.reshape(binary.shape)


def meniscus_footprint(meniscus, origin, M, center, lateral, ap, mm_per_px,
                       closing_mm=1.2, tip_fraction=0.12,
                       min_hull_ratio=0.60, border_mm=0.0):
    """Build the closed meniscus footprint from the outer C contour and its
    ACTUAL open-ended tip-to-tip bridge (the long unsupported convex-hull edge).

    The previous implementation used AP quantiles and selected the innermost
    points, which can land on the middle of the meniscus, not at the two C tips.
    The hull's unsupported edge identifies the gap between both free ends.

    This pilot uses the convex envelope of the projected meniscus. It must be
    reviewed especially for distorted, torn, or postsurgical menisci.
    """
    pts = np.argwhere(meniscus)
    if len(pts) < 50:
        raise RuntimeError('Meniscus mask too small')
    xyz = origin + pts @ M.T
    rel = xyz - center
    u = rel @ lateral
    v = rel @ ap
    u0 = float(np.floor(u.min() / mm_per_px)*mm_per_px - 5)
    v0 = float(np.floor(v.min() / mm_per_px)*mm_per_px - 5)
    nu = int(np.ceil((u.max()-u0+5)/mm_per_px))+1
    nv = int(np.ceil((v.max()-v0+5)/mm_per_px))+1
    if max(nu,nv) > 500:
        raise RuntimeError('Implausibly large projected meniscus mask')
    iu = np.clip(np.rint((u-u0)/mm_per_px).astype(int),0,nu-1)
    iv = np.clip(np.rint((v-v0)/mm_per_px).astype(int),0,nv-1)
    projection = np.zeros((nv,nu),bool)
    projection[iv,iu]=True
    projection = ndi.binary_closing(projection,structure=disk(max(1, closing_mm/mm_per_px)))
    projection = largest_component(projection)
    if projection.sum() < 10:
        raise RuntimeError('Empty projected meniscus')

    coords = np.argwhere(projection)  # (AP, ML) pixel positions
    hull_obj = ConvexHull(coords)
    vertices = coords[hull_obj.vertices]
    hull = convex_footprint(projection)
    # Find the convex-hull edge running through the open gap of the C.
    # Its interior is furthest from actual meniscus tissue (unlike boundary
    # edges lying on the peripheral meniscal contour).
    distance_to_meniscus = ndi.distance_transform_edt(~projection)
    best = None
    for i in range(len(vertices)):
        p0 = vertices[i].astype(float)
        p1 = vertices[(i+1)%len(vertices)].astype(float)
        length = float(np.linalg.norm(p1-p0))
        if length < 4:
            continue
        n_samples = max(10,int(np.ceil(length*3)))
        samples = np.rint(np.linspace(p0,p1,n_samples)).astype(int)
        samples[:,0] = np.clip(samples[:,0],0,nv-1)
        samples[:,1] = np.clip(samples[:,1],0,nu-1)
        d = distance_to_meniscus[samples[:,0],samples[:,1]]
        # Suppress short spurious notches. A true chord has a lengthy middle
        # portion detached from the projected meniscus.
        missing_fraction = float(np.mean(d > 1.5))
        avg_gap = float(np.mean(np.maximum(d-1.0,0)))
        score = length*missing_fraction*avg_gap
        if best is None or score > best[0]:
            best = (score,p0,p1,length,missing_fraction,avg_gap)
    if best is None or best[4] < .20:
        raise RuntimeError('Could not identify a reliable open C-tip chord; review meniscus mask')

    _,tip0,tip1,chord_length,gap_fraction,gap_depth=best
    tips=[np.rint(tip0).astype(int),np.rint(tip1).astype(int)]
    chord=line_mask(projection.shape,tips[0],tips[1],radius_px=1)

    # A convex envelope closes the desired purple tip-to-tip boundary even
    # when there are minor gaps in the external meniscus contour.
    footprint=hull.copy()
    if border_mm > 0:
        footprint=ndi.binary_dilation(footprint,structure=disk(border_mm/mm_per_px))
    elif border_mm < 0:
        footprint=ndi.binary_erosion(footprint,structure=disk(-border_mm/mm_per_px))
    if not footprint.any():
        raise RuntimeError('Empty footprint after border margin')
    tips_uv=[[float(u0+t[1]*mm_per_px),float(v0+t[0]*mm_per_px)] for t in tips]
    return {'footprint':footprint,'projection':projection,'chord':chord,
            'u0':u0,'v0':v0,'grid_mm':mm_per_px,
            'tips_uv_mm':tips_uv,'fill_to_hull_ratio':1.0,
            'convex_fallback':False,'construction':'outer_contour_convex_hull_tip_bridge',
            'bridge_length_mm':float(chord_length*mm_per_px),
            'bridge_gap_fraction':gap_fraction,
            'bridge_gap_depth_pixels':gap_depth,
            'pixel_area_mm2':float(footprint.sum()*mm_per_px**2)}


def write_mask_crop(crop, sl, reference, target):
    full = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint8)
    full[sl] = crop.astype(np.uint8)
    img = sitk.GetImageFromArray(full)
    img.CopyInformation(reference)
    sitk.WriteImage(img, str(target), True)
    log(f'[SAVE] {target.name}: {int(crop.sum()):,} voxels')


def sampling_footprint(fp, u, v):
    # nearest 2D footprint lookup, with safe bounds
    iu = np.rint((u-fp['u0'])/fp['grid_mm']).astype(np.int32)
    iv = np.rint((v-fp['v0'])/fp['grid_mm']).astype(np.int32)
    arr = fp['footprint']
    inside = (iu >= 0) & (iu < arr.shape[1]) & (iv >= 0) & (iv < arr.shape[0])
    ans = np.zeros(u.shape, bool)
    ans[inside] = arr[iv[inside], iu[inside]]
    return ans


def save_footprint_preview(out, fps):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for i, side in enumerate(('medial', 'lateral')):
            fp = fps[side]
            ax[i].imshow(fp['footprint'], cmap='Greens', origin='lower', alpha=.65)
            ax[i].contour(fp['projection'].astype(float), levels=[.5], colors=['purple'], linewidths=.9)
            tip = [((v-fp['v0'])/fp['grid_mm'], (u-fp['u0'])/fp['grid_mm']) for u,v in fp['tips_uv_mm']]
            ax[i].plot([tip[0][1], tip[1][1]], [tip[0][0], tip[1][0]], color='purple', lw=3)
            ax[i].set_title(side + ' (tip-to-tip closure)')
            ax[i].set_xlabel('ML local grid')
            ax[i].set_ylabel('AP local grid')
        fig.savefig(out/'footprint_qc.png', dpi=150)
        plt.close(fig)
        log('[SAVE] footprint_qc.png')
    except Exception as exc:
        log(f'[WARN] QC PNG failed: {exc}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--allow-resample', action='store_true')
    p.add_argument('--footprint-grid-mm', type=float, default=.6)
    p.add_argument('--close-mm', type=float, default=1.2)
    p.add_argument('--border-margin-mm', type=float, default=0.0,
                   help='Positive dilates footprint, negative erodes it; use 0 to follow contour')
    p.add_argument('--joint-half-height-mm', type=float, default=12.0)
    p.add_argument('--max-femur-distance-mm', type=float, default=14.0)
    p.add_argument('--max-tibia-distance-mm', type=float, default=10.0)
    p.add_argument('--crop-margin-mm', type=float, default=18.0)
    p.add_argument('--normal-angle-deg', type=float, default=45.0)
    p.add_argument('--cartilage-shell-mm', type=float, default=5.0)
    p.add_argument('--min-hull-ratio', type=float, default=.60)
    args = p.parse_args()
    if args.footprint_grid_mm <= 0 or args.cartilage_shell_mm <= 0:
        p.error('Grid/shell lengths must be positive')
    base = args.base
    output = args.output or (base/'compartment_roi_meniscus_v3_1')
    output.mkdir(parents=True, exist_ok=True)
    ref = load(base/'PD_FS_triview_fused_0p4mm.nii.gz')
    shape = tuple(reversed(ref.GetSize()))
    origin, M = geometry(ref)
    spacing = np.linalg.norm(M, axis=0)
    bone_dir = base/'bone_guided_nninteractive'
    bones = {name: sitk.GetArrayFromImage(load(bone_dir/f'{name}_nninteractive.nii.gz', ref, allow_resample=args.allow_resample)) > 0
             for name in ('femur', 'tibia', 'fibula')}
    men_dir = base/'meniscus_guided_nninteractive/nninteractive_inference'
    menisci = {side: sitk.GetArrayFromImage(load(men_dir/f'{side}_meniscus.nii.gz', ref, allow_resample=args.allow_resample)) > 0
               for side in ('medial', 'lateral')}
    high_path = base/'meniscus_guided_nninteractive/fluid_nninteractive/high_signal_candidates.nii.gz'
    high = sitk.GetArrayFromImage(load(high_path, ref, allow_resample=args.allow_resample)) > 0
    clouds = {name: physical_samples(mask, origin, M) for name, mask in bones.items()}
    center, lateral, ap, n0 = axes_from_bones(clouds['tibia'], clouds['fibula'])
    n, fit = fit_plateau(clouds['tibia'], center, lateral, ap, n0)
    lateral = unit(lateral - np.dot(lateral, n)*n)
    ap = unit(np.cross(n, lateral))
    if ap[1] < 0: ap = -ap
    tib_h = (clouds['tibia']-center)@n
    fem_h = (clouds['femur']-center)@n
    joint_h = float((np.percentile(tib_h, 99.2)+np.percentile(fem_h, .8))/2)
    log(f'[PLATEAU] n={n.round(4).tolist()} fit={fit}, joint_h={joint_h:.2f} mm')
    fps = {}
    for side in ('medial', 'lateral'):
        # Largest 3D CC removes stray disconnected meniscal predictions.
        mask = largest_component(menisci[side])
        fp = meniscus_footprint(mask, origin, M, center, lateral, ap,
                                mm_per_px=args.footprint_grid_mm,
                                closing_mm=args.close_mm,
                                min_hull_ratio=args.min_hull_ratio,
                                border_mm=args.border_margin_mm)
        fps[side] = fp
        log(f'[FOOTPRINT] {side}: area={fp["pixel_area_mm2"]:.1f}mm², '
            f'tips={fp["tips_uv_mm"]}, convex_fallback={fp["convex_fallback"]}, '
            f'filled/hull={fp["fill_to_hull_ratio"]:.2f}')
    # Limit working grid to joint-level bone + meniscus location.
    joint_clouds = []
    for name in ('femur', 'tibia'):
        cloud = clouds[name]
        selected = cloud[np.abs((cloud-center)@n-joint_h) <= 30]
        joint_clouds.append(selected)
    selected = np.vstack(joint_clouds)
    inv = np.linalg.inv(M)
    corners = (selected-origin)@inv.T
    margin = np.ceil(args.crop_margin_mm/spacing).astype(int)
    lower = np.maximum(np.floor(corners.min(axis=0)).astype(int)-margin, 0)
    upper = np.minimum(np.ceil(corners.max(axis=0)).astype(int)+margin+1, np.asarray(shape))
    sl = tuple(slice(int(a), int(b)) for a,b in zip(lower,upper))
    crop_shape = tuple(int(b-a) for a,b in zip(lower, upper))
    log(f'[CROP] {lower.tolist()} -> {upper.tolist()} = {crop_shape}')
    zz,yy,xx = np.ogrid[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    def part(j):
        return (origin[j]+M[j,0]*zz+M[j,1]*yy+M[j,2]*xx).astype(np.float32)
    px,py,pz = (part(j) for j in range(3))
    u = (px-center[0])*lateral[0]+(py-center[1])*lateral[1]+(pz-center[2])*lateral[2]
    v = (px-center[0])*ap[0]+(py-center[1])*ap[1]+(pz-center[2])*ap[2]
    h = (px-center[0])*n[0]+(py-center[1])*n[1]+(pz-center[2])*n[2]
    del px,py,pz,zz,yy,xx
    f = bones['femur'][sl]; t = bones['tibia'][sl]
    log('[DISTANCE] Computing distance to femur / tibia...')
    df = ndi.distance_transform_edt(~f, sampling=spacing).astype(np.float32)
    dt = ndi.distance_transform_edt(~t, sampling=spacing).astype(np.float32)
    joint = (~f) & (~t) & (df <= args.max_femur_distance_mm) & (dt <= args.max_tibia_distance_mm)
    joint &= np.abs(h-joint_h) <= args.joint_half_height_mm
    # Two independent 2D footprints extruded along plateau normal.
    rois = {}
    for side in ('medial', 'lateral'):
        selection = sampling_footprint(fps[side], u, v)
        rois[side] = joint & selection
        log(f'[ROI] {side}: {int(rois[side].sum()):,} voxels')
    overlap = rois['medial'] & rois['lateral']
    if overlap.any():
        # Cannot claim independent compartments for overlapping footprints.
        # Assign neither ambiguous voxel; save overlap QC mask for review.
        log(f'[WARNING] {int(overlap.sum())} overlapping ROI voxels; excluding ambiguous overlap')
        rois['medial'] &= ~overlap
        rois['lateral'] &= ~overlap
    central = joint & ~rois['medial'] & ~rois['lateral']
    for side in ('medial', 'lateral'):
        write_mask_crop(rois[side], sl, ref, output/f'{side}_jointspace_roi.nii.gz')
        write_mask_crop(rois[side] & high[sl], sl, ref, output/f'{side}_high_signal.nii.gz')
    write_mask_crop(central, sl, ref, output/'outside_footprints_within_joint.nii.gz')
    write_mask_crop(overlap, sl, ref, output/'footprint_overlap_qc.nii.gz')
    combined = np.zeros(crop_shape, dtype=np.uint8)
    combined[rois['medial']] = 1
    combined[rois['lateral']] = 2
    combined[central] = 3
    write_mask_crop(combined, sl, ref, output/'compartment_multilabel.nii.gz')
    # Save normalized-facing cartilage shell ROIs, still merely candidates.
    # Restrict to near-surface and joint envelope (not used to exclude fluid).
    axis_unit = M / spacing[None,:]
    normal_idx = n @ axis_unit
    cos_limit = float(np.cos(np.deg2rad(args.normal_angle_deg)))
    for name,bone,sign in [('femoral', f, -1), ('tibial', t, 1)]:
        interior = ndi.distance_transform_edt(bone, sampling=spacing).astype(np.float32)
        exterior = ndi.distance_transform_edt(~bone, sampling=spacing).astype(np.float32)
        sdf = ndi.gaussian_filter(exterior-interior, 1.25)
        del interior
        grads = np.gradient(sdf, *spacing)
        gdot = sum(g*q for g,q in zip(grads, normal_idx))
        gnorm = np.sqrt(sum(g*g for g in grads))
        del grads, sdf
        face = (~bone) & (exterior <= args.cartilage_shell_mm) & (sign*gdot >= cos_limit*np.maximum(gnorm, 1e-6))
        del exterior,gdot,gnorm
        for side in ('medial','lateral'):
            mask = face & rois[side]
            write_mask_crop(mask, sl, ref, output/f'{side}_{name}_surface_roi.nii.gz')
    save_footprint_preview(output, fps)
    metadata = {'model': 'meniscus_footprint_hull_tip_bridge_v3_1',
                'image': str(base/'PD_FS_triview_fused_0p4mm.nii.gz'),
                'input_high_signal': str(high_path),
                'note': 'Not a validated cartilage/fluid segmentation; verify meniscus tip/chord and ROI placement in Slicer.',
                'frame': {'center_lps': center.tolist(), 'lateral_lps': lateral.tolist(),
                          'ap_lps': ap.tolist(), 'plateau_normal_lps': n.tolist(),
                          'joint_h_mm': joint_h, 'plateau_fit': fit},
                'parameters': {k:v for k,v in vars(args).items() if isinstance(v, (str,int,float,bool))},
                'footprints': {side: {k:v for k,v in fp.items() if k not in ('footprint','projection','chord')}
                               for side, fp in fps.items()},
                'roi_voxels': {side: int(mask.sum()) for side,mask in rois.items()},
                'overlap_voxels': int(overlap.sum())}
    (output/'roi_metadata.json').write_text(json.dumps(metadata, indent=2))
    log(f'[DONE] {output}')


if __name__ == '__main__':
    main()
