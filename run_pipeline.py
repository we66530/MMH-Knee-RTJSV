#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMH Knee RTJSV: finalized intercondylar (IC) soft-tissue exclusion.

RESEARCH PROTOTYPE; NOT A MEASURE OF CARTILAGE VOLUME.

Modes
-----
points    Generate up to 8 positive / 0 negative IC prompts (P0-P95, 10 mm extension).
infer     Run nnInteractive with those points; store raw and bone-excluded predictions.
quantify  Recalculate medial/lateral RTJSV by subtracting the IC prediction.
ic        Resume points -> infer -> quantify (default).
full      Resume pre-existing DICOM -> RTJSV master, then IC stages.

All outputs are new; original v3.2 RTJSV masks remain untouched.
The subtraction is a QC candidate until reviewed against source MRI.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import sys
import hashlib

import numpy as np

CASE_DEFAULT = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513')
MODEL_DEFAULT = Path.home()/'.nninteractive/models/nnInteractive_v1.0'
POINTS_DEFAULT = Path(__file__).resolve().parent/'scripts/bone_guided_intercondylar_soft_tissue_v1_1.py'
INFER_DEFAULT = Path(__file__).resolve().parent/'scripts/infer_intercondylar_soft_tissue_nninteractive_v1_1.py'
MASTER_DEFAULT = Path(__file__).resolve().parent/'scripts/run_mmh_rtjsv_pipeline.py'

TAG = 'IC_P8_N0_Ext10_P0to95_Depth0p4'
SETTINGS = {
    'max_positive': 8,
    'max_negative': 0,
    'inner_extension_mm': 10.0,
    'intensity_low_percentile': 0.0,
    'intensity_high_percentile': 95.0,
    'candidate_min_depth_mm': 0.4,
}


def log(*args):
    print(*args, flush=True)


def nonempty(paths):
    return all(p.is_file() and p.stat().st_size > 0 for p in paths)


def fresh(outputs, inputs):
    if not nonempty(outputs) or not nonempty(inputs):
        return False
    newest_in = max(p.stat().st_mtime_ns for p in inputs)
    return min(p.stat().st_mtime_ns for p in outputs) >= newest_in


def run_command(command, dry_run=False):
    log('  $', shlex.join(str(x) for x in command))
    if not dry_run:
        subprocess.run([str(x) for x in command], check=True)


def paths_for(case):
    root = case/'fusion_output'
    ic_root = root/'intercondylar_soft_tissue'
    return dict(
        root=root,
        image=root/'PD_FS_triview_fused_0p4mm.nii.gz',
        points_dir=ic_root/f'point_selection_{TAG}',
        infer_dir=ic_root/f'nninteractive_{TAG}',
        output=root/f'jointspace_residual_volume_v3_2_{TAG}',
        raw_rtjsv=root/'jointspace_residual_volume_v3_2',
        bone_dir=root/'bone_guided_nninteractive',
        roi_dir=root/'compartment_roi_meniscus_v3_2',
        men_dir=root/'meniscus_guided_nninteractive'/'nninteractive_inference',
        fluid=root/'meniscus_guided_nninteractive'/'fluid_nninteractive'/'high_signal_candidates.nii.gz',
    )


def fixed_prompts(json_path):
    """Refuse to run inference on stale points from an earlier parameter trial."""
    if not json_path.is_file():
        return False, 0
    try:
        obj = json.loads(json_path.read_text(encoding='utf-8'))
        params = obj['parameters']
        for key, expected in SETTINGS.items():
            if key not in params or not np.isclose(float(params[key]), expected, atol=1e-7, rtol=0):
                return False, 0
        pts = obj.get('positive', [])
        negatives = obj.get('negative', [])
        return bool(pts) and len(pts) <= 8 and not negatives, len(pts)
    except (OSError, KeyError, ValueError, TypeError):
        return False, 0


def check_fluid_settings(pp):
    """Refuse silently mixing older fluid thresholds with current IC/RTJSV run."""
    prompt_file = pp['fluid'].parent / 'pool_prompts.json'
    if not prompt_file.is_file():
        log('[WARNING] Fluid pool metadata not found; cannot verify P99/P93 provenance:', prompt_file)
        return
    data = json.loads(prompt_file.read_text(encoding='utf-8'))
    pars = data.get('parameters', {})
    expected = {'high_percentile':99, 'low_percentile':93,
                'min_pool_mm3':5, 'max_pools':12}
    differences = []
    for key, value in expected.items():
        try:
            actual = float(pars[key])
            if not np.isclose(actual, float(value), rtol=0, atol=1e-8):
                differences.append(f'{key}={actual} (expected {value})')
        except (KeyError,TypeError,ValueError):
            differences.append(f'{key}=UNKNOWN (expected {value})')
    if differences:
        raise RuntimeError('Current high_signal_candidates may not be the chosen P99/P93 set: '
                           + '; '.join(differences) + '. Re-run fluid points and v3.2 ROI/RTJSV first.')
    log('[FLUID] Verified P99/P93, minimum pool 5 mm3, maximum 12 pools')


def points_stage(args, pp, force=False):
    args.points_script = args.points_script.expanduser()
    prompts = pp['points_dir']/'prompts.json'
    ok, count = fixed_prompts(prompts)
    required = [pp['image'],
                pp['roi_dir']/'roi_metadata.json',
                pp['roi_dir']/'medial_jointspace_roi.nii.gz',
                pp['roi_dir']/'lateral_jointspace_roi.nii.gz',
                pp['fluid']]
    required += [pp['bone_dir']/f'{b}_nninteractive.nii.gz'
                 for b in ('femur','tibia','fibula','patella')]
    required += [pp['men_dir']/f'{x}_meniscus.nii.gz' for x in ('medial','lateral')]
    if not args.dry_run and nonempty([pp['fluid']]):
        check_fluid_settings(pp)
    if not args.dry_run and not nonempty(required):
        raise FileNotFoundError(f'Point stage prerequisites missing: {[str(p) for p in required if not p.is_file()]}')
    outputs = [prompts, pp['points_dir']/'positive_points_LPS.fcsv',
               pp['points_dir']/'point_selection_preview.png']
    dependencies = required + ([args.points_script] if args.points_script.is_file() else [])
    if not force and ok and fresh(outputs, dependencies):
        log(f'[SKIP points] Existing matching {count} positive points: {prompts}')
        return False
    if not args.points_script.is_file() and not args.dry_run:
        raise FileNotFoundError(args.points_script)
    cmd = [args.python, args.points_script,
           '--case-dir', args.case_dir, '--output-dir', pp['points_dir'], '--mode','points',
           '--max-positive',8, '--max-negative',0,
           '--inner-extension-mm',10,
           '--intensity-low-percentile',0,
           '--intensity-high-percentile',95,
           '--candidate-min-depth-mm',0.4]
    log('[RUN points] Fixed IC settings: 8 positives / 0 negatives / extension 10 mm / P0-P95 / min depth 0.4 mm')
    run_command(cmd, args.dry_run)
    if not args.dry_run:
        ok, count = fixed_prompts(prompts)
        if not ok:
            raise RuntimeError('Point generator returned missing/incorrect settings, zero positive points, or unexpected negatives')
        log(f'[OK points] {count} positive points; review: {pp["points_dir"]/"point_selection_preview.png"}')
    return True


def infer_stage(args, pp, force=False):
    args.infer_script = args.infer_script.expanduser()
    prompts = pp['points_dir']/'prompts.json'
    ok, count = fixed_prompts(prompts)
    if not args.dry_run and not ok:
        raise RuntimeError('Prompts do not match fixed IC settings; run --mode points or --mode ic first')
    if not args.dry_run and not args.infer_script.is_file():
        raise FileNotFoundError(args.infer_script)
    if not args.dry_run and not (args.model_dir/'fold_0'/'checkpoint_final.pth').is_file():
        raise FileNotFoundError('Missing nnInteractive checkpoint')
    out = pp['infer_dir']
    outputs = [out/'intercondylar_positive_only.nii.gz',
               out/'intercondylar_soft_tissue_raw.nii.gz',
               out/'intercondylar_soft_tissue_bone_excluded.nii.gz',
               out/'inference_qc_report.json']
    inputs = [prompts,pp['image'],args.infer_script] + [pp['bone_dir']/f'{b}_nninteractive.nii.gz'
                                                     for b in ('femur','tibia','fibula','patella')]
    if not force and fresh(outputs, inputs):
        log(f'[SKIP infer] Existing prediction: {out}')
        return False
    cmd = [args.python, args.infer_script, '--mode','infer',
           '--case-dir',args.case_dir, '--prompts',prompts, '--output-dir',out,
           '--model-dir',args.model_dir, '--no-use-negatives']
    log('[RUN infer] nnInteractive positive-only, then subtract known bone masks')
    run_command(cmd,args.dry_run)
    if not args.dry_run and not nonempty(outputs):
        raise RuntimeError('Inference script did not produce the expected outputs: '+str(outputs))
    return True


def load_mask(path, ref, sitk):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = sitk.ReadImage(str(path))
    if im.GetSize() != ref.GetSize() or not all((
            np.allclose(im.GetSpacing(),ref.GetSpacing(),atol=1e-4,rtol=0),
            np.allclose(im.GetOrigin(),ref.GetOrigin(),atol=1e-3,rtol=0),
            np.allclose(im.GetDirection(),ref.GetDirection(),atol=1e-5,rtol=0))):
        raise RuntimeError(f'Geometry mismatch (no implicit resampling): {path}')
    return sitk.GetArrayFromImage(im) != 0


def save_mask(path, array, ref, sitk):
    path.parent.mkdir(parents=True,exist_ok=True)
    im = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
    im.CopyInformation(ref)
    sitk.WriteImage(im,str(path),True)
    log('[SAVE]',path.name,'voxels:',int(np.count_nonzero(array)))


def input_digest(paths):
    # Source fingerprint is fast; records file size and exact mtime, not patient content.
    items=[(str(x), x.stat().st_size, x.stat().st_mtime_ns) for x in paths]
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def quantify_stage(args,pp,force=False):
    ic_path = args.ic_mask.expanduser().resolve() if args.ic_mask else pp['infer_dir']/'intercondylar_soft_tissue_raw.nii.gz'
    mask_files=[pp['image'],ic_path]
    bone_files=[pp['bone_dir']/f'{b}_nninteractive.nii.gz' for b in ('femur','tibia','fibula','patella')]
    residual_files=[pp['raw_rtjsv']/f'{side}_residual_space.nii.gz' for side in ('medial','lateral')]
    source_files=mask_files+bone_files+residual_files
    corridor_path=pp['points_dir']/'central_corridor_roi.nii.gz'
    if args.cap_to_corridor:
        source_files.append(corridor_path)
    if not args.dry_run and not nonempty(source_files):
        raise FileNotFoundError('Missing required source(s): '+str([str(q) for q in source_files if not q.is_file()]))
    dest=pp['output']
    manifest=dest/'volume_report.json'
    outputs=[dest/f'{side}_residual_corrected.nii.gz' for side in ('medial','lateral')]
    outputs += [dest/'residual_corrected_multilabel.nii.gz', dest/'RTJSV_comparison.csv', manifest]
    if not force and not args.dry_run and nonempty(outputs):
        try:
            old=json.loads(manifest.read_text(encoding='utf-8'))
            if old['source_fingerprint']==input_digest(source_files) and old['cap_to_corridor']==args.cap_to_corridor:
                log(f'[SKIP quantify] Inputs unchanged, existing report: {dest/"RTJSV_comparison.csv"}')
                return False
        except (OSError,KeyError,ValueError,TypeError):
            pass
    if args.dry_run:
        log('[RUN quantify] Proposed: RTJSV_v3.2 AND NOT (IC_prediction AND NOT bone)')
        log('[OUT]',dest)
        return True
    import SimpleITK as sitk
    ref=sitk.ReadImage(str(pp['image']))
    if ref.GetDimension()!=3:
        raise RuntimeError('Fused MRI must be 3D')
    ic_raw=load_mask(ic_path,ref,sitk)
    if not ic_raw.any():
        raise RuntimeError('Empty nnInteractive prediction; refusing to report corrected volumes')
    bone=np.zeros(ic_raw.shape, dtype=bool)
    for path in bone_files:
        bone |= load_mask(path,ref,sitk)
    ic_bone_clean=ic_raw & ~bone
    if args.cap_to_corridor:
        corridor=load_mask(corridor_path,ref,sitk)
        ic_for_subtraction=ic_bone_clean & corridor
    else:
        ic_for_subtraction=ic_bone_clean
    if not ic_for_subtraction.any():
        raise RuntimeError('IC exclusion empty after applying bone and optional corridor mask')
    dest.mkdir(parents=True,exist_ok=True)
    save_mask(dest/'intercondylar_bone_excluded.nii.gz', ic_bone_clean, ref,sitk)
    if args.cap_to_corridor:
        save_mask(dest/'intercondylar_within_corridor.nii.gz',ic_for_subtraction,ref,sitk)
    voxel_mm3=float(np.prod(ref.GetSpacing()))
    voxel_ml=voxel_mm3/1000.
    rows=[]
    corrected_total=np.zeros(ic_raw.shape,dtype=np.uint8)
    removed_total=np.zeros(ic_raw.shape,dtype=np.uint8)
    original_total=np.zeros(ic_raw.shape,dtype=bool)
    for label,side in enumerate(('medial','lateral'),1):
        residual=load_mask(pp['raw_rtjsv']/f'{side}_residual_space.nii.gz',ref,sitk)
        if not residual.any():
            raise RuntimeError(f'{side}: baseline residual is empty')
        if (residual & original_total).any():
            raise RuntimeError('Baseline medial and lateral residual masks overlap; refusing ambiguous volume accounting')
        original_total |= residual
        removed=residual & ic_for_subtraction
        corrected=residual & ~ic_for_subtraction
        assert (corrected & removed).sum()==0
        assert int(residual.sum()) == int(removed.sum())+int(corrected.sum())
        n0,nr,n1=[int(q.sum()) for q in (residual,removed,corrected)]
        row={
            'compartment':side,
            'RTJSV_v3_2_ml':round(n0*voxel_ml,6),
            'IC_excluded_ml':round(nr*voxel_ml,6),
            'RTJSV_corrected_ml':round(n1*voxel_ml,6),
            'removed_fraction_pct':round(100*nr/n0,3),
            'original_voxels':n0,
            'removed_voxels':nr,
            'corrected_voxels':n1,
            'qc_warning':('REVIEW: removed > '+str(args.warning_fraction_pct)+'%' if nr/n0*100>args.warning_fraction_pct else ''),
        }
        log(f'[VOLUME] {side:7s} {row["RTJSV_v3_2_ml"]:.4f} -> {row["RTJSV_corrected_ml"]:.4f} mL  (removed {row["IC_excluded_ml"]:.4f} mL, {row["removed_fraction_pct"]:.2f}%)')
        if row['qc_warning']:
            log('[WARNING]',side, row['qc_warning'])
        rows.append(row)
        save_mask(dest/f'{side}_residual_corrected.nii.gz',corrected,ref,sitk)
        save_mask(dest/f'{side}_removed_intercondylar_QC.nii.gz',removed,ref,sitk)
        corrected_total[corrected]=label
        removed_total[removed]=label
    save_mask(dest/'residual_corrected_multilabel.nii.gz',corrected_total,ref,sitk)
    save_mask(dest/'removed_intercondylar_multilabel_QC.nii.gz',removed_total,ref,sitk)
    csv_path=dest/'RTJSV_comparison.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    body={
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'version':TAG,
        'definition':'RTJSV_corrected = RTJSV_v3.2 AND NOT (IC_nnInteractive AND NOT bone)',
        'cap_to_corridor':bool(args.cap_to_corridor),
        'voxel_volume_mm3':voxel_mm3,
        'IC_raw_voxels':int(ic_raw.sum()),
        'IC_without_bone_voxels':int(ic_bone_clean.sum()),
        'IC_for_subtraction_voxels':int(ic_for_subtraction.sum()),
        'source_fingerprint':input_digest(source_files),
        'source_paths':[str(p) for p in source_files],
        'IC_source_mode': ('explicit_existing_mask' if args.ic_mask else 'pipeline_output'),
        'point_settings':SETTINGS,
        'compartments':rows,
        'qc_status':'REQUIRES_PHYSICIAN_REVIEW',
        'warning':'Not a direct cartilage volume measurement. Candidate exclusion may remove normal cartilage; compare MRI and QC masks.',
    }
    manifest.write_text(json.dumps(body,indent=2,ensure_ascii=False),encoding='utf-8')
    log('[DONE] Comparison:',csv_path)
    log('[QC] Compare removed_intercondylar_multilabel_QC.nii.gz against original MRI.')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=('points','infer','quantify','ic','full'),default='ic')
    p.add_argument('--case-dir',type=Path,default=CASE_DEFAULT)
    p.add_argument('--python',default=sys.executable,help='Python executable in active mricore environment')
    p.add_argument('--points-script',type=Path,default=POINTS_DEFAULT)
    p.add_argument('--infer-script',type=Path,default=INFER_DEFAULT)
    p.add_argument('--master-script',type=Path,default=MASTER_DEFAULT)
    p.add_argument('--model-dir',type=Path,default=MODEL_DEFAULT)
    p.add_argument('--ic-mask',type=Path,help='Use an already generated IC mask for --mode quantify (skip new nnInteractive inference)')
    p.add_argument('--force',action='store_true',help='Recompute every selected stage regardless of cached output')
    p.add_argument('--dry-run',action='store_true',help='Show execution plan; no model calls or file writes')
    p.add_argument('--cap-to-corridor',action=argparse.BooleanOptionalAction,default=False,
                   help='Restrict subtraction to point-generation central corridor, to reduce spread')
    p.add_argument('--warning-fraction-pct',type=float,default=20.0)
    args=p.parse_args(argv)
    args.case_dir=args.case_dir.expanduser().resolve()
    args.model_dir=args.model_dir.expanduser()
    pp=paths_for(args.case_dir)
    if args.cap_to_corridor:
        pp['output'] = Path(str(pp['output']) + '_corridor_capped')
    if args.ic_mask:
        mask_tag = hashlib.sha1(str(args.ic_mask.expanduser().resolve()).encode()).hexdigest()[:8]
        pp['output'] = Path(str(pp['output']) + '_external_' + mask_tag)
    log('[CASE]',args.case_dir)
    log('[MODE]',args.mode)
    log('[IC SETTINGS]',SETTINGS)
    log('[ORIGINAL v3.2]',pp['raw_rtjsv'])
    log('[NEW OUTPUT]',pp['output'])
    if args.mode=='full':
        cmd=[args.python,args.master_script,'--case-dir',args.case_dir]
        log('[STAGE 0] Resume original fusion->v3.2 RTJSV master')
        run_command(cmd,args.dry_run)
    if args.mode in ('points','ic','full'):
        changed=points_stage(args,pp,force=args.force)
    else:
        changed=False
    if args.mode in ('infer','ic','full'):
        changed=infer_stage(args,pp,force=(args.force or changed))
    if args.mode in ('quantify','ic','full'):
        quantify_stage(args,pp,force=(args.force or changed))
    if args.dry_run:
        log('[DRY RUN COMPLETE]')
    else:
        log('[PIPELINE COMPLETE]')

if __name__=='__main__':
    main()
