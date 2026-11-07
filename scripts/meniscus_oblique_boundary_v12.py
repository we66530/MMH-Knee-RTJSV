#!/usr/bin/env python3
"""Anatomical oblique-boundary negative prompt proposal (NO inference).
Consumes v1.1 meniscus points and nnInteractive bone masks. Coordinates: SITK LPS, array ZYX.
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

DEFAULT_BASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513/fusion_output')

def get_array(path, ref):
    im = sitk.ReadImage(str(path))
    if (im.GetSize() != ref.GetSize() or not np.allclose(im.GetOrigin(), ref.GetOrigin(), atol=1e-3)
        or not np.allclose(im.GetSpacing(), ref.GetSpacing(), atol=1e-4)
        or not np.allclose(im.GetDirection(), ref.GetDirection(), atol=1e-5)):
        raise ValueError(f'Geometry mismatch: {path}. Align before using this script.')
    return sitk.GetArrayFromImage(im)

def sample_mask(mask, image, limit=400000):
    coords = np.argwhere(mask)
    if not len(coords): raise ValueError('Empty bone mask')
    if len(coords) > limit: coords = coords[::int(np.ceil(len(coords)/limit))]
    xyz = coords[:, ::-1].astype(float)
    D = np.asarray(image.GetDirection()).reshape(3,3)
    return np.asarray(image.GetOrigin()) + (xyz * np.asarray(image.GetSpacing())) @ D.T

def lps_of_indices(zyx, image):
    xyz = np.asarray(zyx)[..., ::-1].astype(float)
    D = np.asarray(image.GetDirection()).reshape(3,3)
    return np.asarray(image.GetOrigin()) + (xyz * np.asarray(image.GetSpacing())) @ D.T

def record_point(side, zone, kind, idx, image, extra=None):
    p = lps_of_indices(np.asarray(idx, dtype=float), image)
    o = {'side':side, 'zone':zone, 'type':kind,
         'voxel_zyx':[int(t) for t in idx], 'lps_mm':[float(t) for t in p]}
    if extra: o.update(extra)
    return o

def export_fcsv(points, path):
    with open(path, 'w', newline='') as f:
        f.write('# Markups fiducial file version = 4.11\n# CoordinateSystem = 1\n')
        f.write('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')
        writer=csv.writer(f)
        for i,p in enumerate(points):
            writer.writerow([f'p{i+1}',*p['lps_mm'],0,0,0,1,1,1,0,
                             f"{p['type']}_{p['side']}_{p['zone']}",'',''])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--outer-margin-mm',type=float,default=2.0,
                    help='Allow this much excursion of positive points beyond line')
    ap.add_argument('--negative-start-mm',type=float,default=3.5)
    ap.add_argument('--negative-end-mm',type=float,default=7.0)
    ap.add_argument('--ap-window-mm',type=float,default=10.0)
    ap.add_argument('--max-negative-per-side',type=int,default=2)
    args=ap.parse_args()
    base=args.base
    ref=sitk.ReadImage(str(base/'PD_FS_triview_fused_0p4mm.nii.gz'))
    src=base/'bone_guided_nninteractive'
    masks={k:get_array(src/f'{k}_nninteractive.nii.gz',ref)>0 for k in ['femur','tibia','fibula','patella']}
    pts_path=base/'meniscus_guided_nninteractive/positive_points.json'
    raw=json.loads(pts_path.read_text())
    positives=raw['points'] if isinstance(raw,dict) else raw
    img=get_array(base/'PD_FS_triview_fused_0p4mm.nii.gz',ref).astype(np.float32)
    physical={k:sample_mask(m,ref) for k,m in masks.items()}
    t,f,fb=physical['tibia'], physical['femur'], physical['fibula']
    tibia_top=float(np.percentile(t[:,2],99.2))
    femur_bottom=float(np.percentile(f[:,2],0.8))
    tibia_local=t[(t[:,2]>tibia_top-22)&(t[:,2]<tibia_top-3)]
    if len(tibia_local)<100: raise RuntimeError('Proximal tibia landmark insufficient')
    t_center=np.median(tibia_local[:,:2],axis=0)
    fb_local=fb[fb[:,2]>np.percentile(fb[:,2],85)]
    lat=np.median(fb_local[:,:2],axis=0)-t_center
    lat/=np.linalg.norm(lat)
    ap_axis=np.array([-lat[1],lat[0]])
    def latcoord(x): return (x[:,:2]-t_center)@lat
    def apcoord(x): return (x[:,:2]-t_center)@ap_axis
    def side_coordinate(x,side): return latcoord(x)*(1 if side=='lateral' else -1)
    out=base/'meniscus_guided_nninteractive/oblique_boundary_v12'
    out.mkdir(parents=True, exist_ok=True)
    accepted=[]; rejected=[]; negs=[]; lines=[]; boundary_markers=[]
    # A separate local oblique line is fitted for each positive-point AP location.
    for p in positives:
        side=p['side']; zone=p['zone']
        point_lps=np.asarray(p['lps_mm'],float)
        ap_target=float((point_lps[:2]-t_center)@ap_axis)
        anchors=[]
        for bone, zref, zlo,zhi in [
            ('tibia',tibia_top,tibia_top-12,tibia_top-3),
            ('femur',femur_bottom,femur_bottom+3,femur_bottom+12)]:
            coords=physical[bone]
            chosen=coords[(np.abs(apcoord(coords)-ap_target)<args.ap_window_mm)&
                          (coords[:,2]>=zlo)&(coords[:,2]<=zhi)]
            if len(chosen)<50:
                print(f'[SKIP] {side}/{zone}: no reliable {bone} side rim'); anchors=[];break
            s=side_coordinate(chosen,side)
            # Robust outer bone contour: outermost 4% rather than one extreme voxel.
            outer=s>=np.percentile(s,96)
            anchors.append((float(np.median(chosen[outer,2])),float(np.median(s[outer]))))
        if len(anchors)<2: rejected.append({**p,'reason':'boundary_unavailable'}); continue
        (z_t,s_t),(z_f,s_f)=anchors
        if abs(z_f-z_t)<2: rejected.append({**p,'reason':'anchor_z_too_close'});continue
        slope=(s_f-s_t)/(z_f-z_t)
        def rim_at_z(z): return s_t+slope*(z-z_t)
        pos_side=float(((point_lps[:2]-t_center)@lat)*(1 if side=='lateral' else -1))
        excursion=pos_side-rim_at_z(point_lps[2])
        # Export physical endpoint fiducials of each oblique boundary.
        for name, z_val, s_val in [('tibia',z_t,s_t),('femur',z_f,s_f)]:
            local_xy=t_center + ap_target*ap_axis + s_val*(1 if side=='lateral' else -1)*lat
            boundary_markers.append({'side':side, 'zone':zone, 'type':f'boundary_{name}',
                                     'lps_mm':[float(local_xy[0]),float(local_xy[1]),float(z_val)]})
        lines.append(dict(side=side,zone=zone, tibia_anchor_zs=[z_t,s_t],
                          femur_anchor_zs=[z_f,s_f], slope=slope, point_excursion_mm=excursion))
        if excursion>args.outer_margin_mm:
            rejected.append({**p,'reason':'outside_oblique_boundary','excursion_mm':float(excursion)})
            print(f'[REJECT +] {side}/{zone}: {excursion:.1f} mm beyond line')
        else:
            accepted.append(p)
            print(f'[KEEP +] {side}/{zone}: excursion={excursion:.1f} mm')
        # Find safe negative point OUTSIDE the oblique line, not on bone.
        # Restrict search to same AP zone and approx joint height; for memory only crop around positive.
        idx=np.asarray(p['voxel_zyx'],int)
        spacing=np.asarray(ref.GetSpacing())[::-1]
        half=np.ceil(np.array([14,22,22])/spacing).astype(int)
        lo=np.maximum(idx-half,0);hi=np.minimum(idx+half+1,np.asarray(img.shape))
        sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        crop=img[sl]
        zz,yy,xx=np.ogrid[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        D=np.asarray(ref.GetDirection()).reshape(3,3)
        M=D@np.diag(np.asarray(ref.GetSpacing()))[:,::-1]
        o=np.asarray(ref.GetOrigin())
        px=o[0]+M[0,0]*zz+M[0,1]*yy+M[0,2]*xx
        py=o[1]+M[1,0]*zz+M[1,1]*yy+M[1,2]*xx
        pz=o[2]+M[2,0]*zz+M[2,1]*yy+M[2,2]*xx
        outward=((px-t_center[0])*lat[0]+(py-t_center[1])*lat[1])*(1 if side=='lateral' else -1)
        apgrid=(px-t_center[0])*ap_axis[0]+(py-t_center[1])*ap_axis[1]
        beyond=outward-rim_at_z(pz)
        exclude=np.zeros(crop.shape,bool)
        for m in masks.values(): exclude|=m[sl]
        valid=(~exclude)&np.isfinite(crop)&(beyond>=args.negative_start_mm)&(beyond<=args.negative_end_mm)&\
              (np.abs(apgrid-ap_target)<=args.ap_window_mm)&(np.abs(pz-point_lps[2])<=8)
        if valid.sum()<20:
            print(f'[SKIP -] {side}/{zone}: outside region too small');continue
        dark=valid & (crop<=np.percentile(crop[valid],40))
        # Favor a continuous dark core, rather than a single noisy pixel.
        depth=ndi.distance_transform_edt(dark,sampling=spacing)
        if depth.max()<0.5:
            print(f'[SKIP -] {side}/{zone}: no stable dark region');continue
        loc=np.unravel_index(np.argmax(depth),depth.shape)
        global_zyx=np.asarray(loc)+lo
        negative=record_point(side,zone,'negative',global_zyx,ref,
                              {'dark_depth_mm':float(depth[loc]), 'beyond_line_mm':float(beyond[loc])})
        negs.append(negative)
        print(f'[PROPOSE -] {side}/{zone}: {negative["voxel_zyx"]} ({negative["beyond_line_mm"]:.1f} mm outside)')
    # Select at most 1-2 spatially separated negatives per side.
    limited=[]
    for side in ['medial','lateral']:
        choices=[n for n in negs if n['side']==side]
        choices.sort(key=lambda x:x['dark_depth_mm'],reverse=True)
        chosen=[]
        for p in choices:
            if all(np.linalg.norm(np.array(p['lps_mm'])-np.array(q['lps_mm']))>8 for q in chosen):
                chosen.append(p)
            if len(chosen)>=args.max_negative_per_side: break
        limited.extend(chosen)
    payload={'positive':accepted,'rejected_positive':rejected,'negative':limited,
             'boundaries':lines,'note':'Proposals only. Inspect in Slicer before inference.'}
    (out/'prompts_with_negative.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    export_fcsv(accepted,out/'positive_kept_LPS.fcsv')
    export_fcsv(limited,out/'negative_proposed_LPS.fcsv')
    export_fcsv(boundary_markers,out/'boundary_endpoints_LPS.fcsv')
    print(f'\n[DONE] kept + {len(accepted)}, rejected + {len(rejected)}, proposed - {len(limited)}')
    print('[OUTPUT]',out)
    print('IMPORTANT: if a compartment has zero accepted positive points, DO NOT run nnInteractive on it.')

if __name__=='__main__': main()
