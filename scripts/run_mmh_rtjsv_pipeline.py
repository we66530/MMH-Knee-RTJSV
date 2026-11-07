#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end research pipeline: tri-view DICOM -> 0.4 mm MRI -> RTJSV.

Invokes already-validated stages separately. State is resumable by output existence.
Meniscus selection and nnInteractive inference use user's EXISTING working scripts;
patched COPIES are created for the selected case, originals are never edited.

USAGE:
 python run_mmh_rtjsv_pipeline.py --case-dir /media/.../37321865_20260513 --dry-run
 python run_mmh_rtjsv_pipeline.py --case-dir /media/.../37321865_20260513
 python run_mmh_rtjsv_pipeline.py --case-dir ... --from-stage meniscus_points --force-from

This is a research prototype; physician QC is required at stage gates.
"""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

STAGES = (
    'fusion', 'totalseg', 'bone_nninteractive',
    'meniscus_points', 'meniscus_boundary', 'meniscus_nninteractive',
    'fluid_points', 'compartment_roi', 'rtjsv',
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE = Path('/media/we66530/Transcend/MMH_knee_dcm/PACS_downloads/37321865_20260513')
DEFAULT_MODEL = Path.home() / '.nninteractive/models/nnInteractive_v1.0'


def emit(*items):
    print(*items, flush=True)


def patch_module_base(source_path, output_path, new_base):
    """Rewrite only the top-level assignment to BASE in a *copy* of old script.

    Handles multiline `BASE = Path(...)` without regex or touching other code.
    Fails closed if the expected source is absent or ambiguous.
    """
    src = Path(source_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f'Existing script is required: {src}')
    code = src.read_text(encoding='utf-8')
    tree = ast.parse(code, filename=str(src))
    matches = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == 'BASE' for t in targets):
                matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one global BASE assignment in {src}, found {len(matches)}')
    node = matches[0]
    lines = code.splitlines(keepends=True)
    replacement = f"BASE = Path({str(new_base)!r})\n"
    lines[node.lineno-1:node.end_lineno] = [replacement]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(lines), encoding='utf-8')
    ast.parse(output_path.read_text(), filename=str(output_path))
    return output_path


def checked_outputs(paths):
    return all(p.is_file() and p.stat().st_size > 0 for p in paths)


def stage_outputs(stage, base):
    bone = base / 'bone_guided_nninteractive'
    men = base / 'meniscus_guided_nninteractive'
    roi = base / 'compartment_roi_meniscus_v3_2'
    return {
      'fusion': [base / 'PD_FS_triview_fused_0p4mm.nii.gz'],
      'totalseg': [base/'totalseg_bone_prior/total_mr/femur_right.nii.gz',
                   *[base/f'totalseg_bone_prior/appendicular_bones_mr/{b}.nii.gz'
                     for b in ('tibia','fibula','patella')]],
      'bone_nninteractive': [bone/f'{b}_nninteractive.nii.gz' for b in ('femur','tibia','fibula','patella')],
      'meniscus_points': [men/'positive_points.json', men/'meniscus_positive_points_LPS.fcsv'],
      'meniscus_boundary': [men/'oblique_boundary_v12/prompts_with_negative.json'],
      'meniscus_nninteractive': [men/f'nninteractive_inference/{side}_meniscus.nii.gz'
                               for side in ('medial','lateral')],
      'fluid_points': [men/'fluid_nninteractive/pool_prompts.json',
                       men/'fluid_nninteractive/high_signal_candidates.nii.gz'],
      'compartment_roi': [roi/f'{side}_{suffix}.nii.gz'
                          for side in ('medial','lateral')
                          for suffix in ('jointspace_roi','high_signal')]
                         + [roi/'roi_metadata.json'],
      'rtjsv': [base/'jointspace_residual_volume_v3_2/jointspace_volume_summary.csv',
                base/'jointspace_residual_volume_v3_2/jointspace_volume_summary.json'],
    }[stage]


def generate_commands(stage, args, base):
    image = base/'PD_FS_triview_fused_0p4mm.nii.gz'
    bone_prior = base/'totalseg_bone_prior'
    model = args.model_dir.expanduser()
    py = args.python
    if stage == 'fusion':
        # Converting a new case: DICOM series folder names are configurable.
        fusion_launcher = ([args.conda_bin, 'run', '--no-capture-output', '-n', args.fusion_env, 'python']
                           if args.fusion_env else [py])
        cmd = [*fusion_launcher, str(args.scripts_dir/'reconstruct_triview_cuda_only_MMH_DICOM.py'),
               '--axial', str(args.case_dir/args.axial_dir),
               '--coronal', str(args.case_dir/args.coronal_dir),
               '--sagittal', str(args.case_dir/args.sagittal_dir),
               '--output', str(image), '--spacing', '0.4', '--device', 'cuda']
        return [cmd]
    if stage == 'totalseg':
        exe = args.totalseg_bin
        launcher = ([args.conda_bin,'run','--no-capture-output','-n',args.totalseg_env]
                    if args.totalseg_env else [])
        return [[*launcher,exe,'-i',str(image),'-o',str(bone_prior/'total_mr'),'--task','total_mr'],
                [*launcher,exe,'-i',str(image),'-o',str(bone_prior/'appendicular_bones_mr'),
                 '--task','appendicular_bones_mr']]
    if stage == 'bone_nninteractive':
        cmd=[py, str(args.scripts_dir/'bone_guided_nninteractive.py'),
             '--root',str(base),'--mode','infer','--allow-resample','--model-dir',str(model)]
        return [cmd]
    if stage == 'meniscus_points':
        return [[py, str(args.generated_dir/'meniscus_points_for_case.py'),
                 '--low-signal-percentile',str(args.meniscus_low_signal_percentile)]]
    if stage == 'meniscus_boundary':
        return [[py, str(args.scripts_dir/'meniscus_oblique_boundary_v12.py'),
                 '--base',str(base), '--outer-margin-mm','2.0',
                 '--max-negative-per-side','2']]
    if stage == 'meniscus_nninteractive':
        return [[py, str(args.generated_dir/'infer_meniscus_for_case.py'),
                 '--model-dir',str(model)]]
    if stage == 'fluid_points':
        return [[py, str(args.generated_dir/'fluid_for_case.py'),
                 '--mode','points','--high-percentile',str(args.fluid_high),
                 '--low-percentile',str(args.fluid_low),
                 '--min-pool-mm3',str(args.fluid_min_pool_mm3),
                 '--max-pools',str(args.fluid_max_pools)]]
    if stage == 'compartment_roi':
        return [[py, str(args.scripts_dir/'bone_guided_compartment_roi_v3_2.py'),
                 '--base',str(base), '--footprint-grid-mm','0.6',
                 '--close-mm','1.2', '--border-margin-mm','0',
                 '--joint-half-height-mm','12',
                 '--max-femur-distance-mm','14',
                 '--max-tibia-distance-mm','10',
                 '--normal-angle-deg','45', '--cartilage-shell-mm','5']]
    if stage == 'rtjsv':
        return [[py,str(args.scripts_dir/'calculate_jointspace_residual_volume.py'),
                 '--base',str(base),
                 '--roi-dir',str(base/'compartment_roi_meniscus_v3_2'),
                 '--out',str(base/'jointspace_residual_volume_v3_2')]]
    raise KeyError(stage)


def run_logged(cmd, case_dir, log_file):
    """Stream stage subprocess output to terminal and a per-case local log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open('a', encoding='utf-8') as log:
        log.write(f"\n[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {shlex.join(map(str,cmd))}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(case_dir), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = proc.wait()
        log.write(f"[RETURN_CODE] {code}\n")
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def write_manifest(out, args, success, stage_results):
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {'case_dir':str(args.case_dir),
            'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'success' if success else 'failed_or_incomplete',
            'python':args.python, 'model_dir':str(args.model_dir),
            'fluid_thresholds':{'high':args.fluid_high,'low':args.fluid_low,
                                'min_pool_mm3':args.fluid_min_pool_mm3,
                                'max_pools':args.fluid_max_pools},
            'stages':stage_results}
    out.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',type=Path,default=DEFAULT_CASE)
    p.add_argument('--scripts-dir',type=Path,default=SCRIPT_DIR)
    p.add_argument('--python',default=sys.executable,
                   help='Python with nnInteractive/torch/SimpleITK; currently activated mricore is recommended')
    p.add_argument('--totalseg-bin',default='TotalSegmentator',
                   help='TotalSegmentator CLI executable (its appendicular MR task may need a license)')
    p.add_argument('--conda-bin',default='conda')
    p.add_argument('--fusion-env',default='totalseg5090',
                   help='Conda env for GPU fusion; use empty string for activated Python')
    p.add_argument('--totalseg-env',default='totalseg5090',
                   help='Conda env with TotalSegmentator; use empty string for activated env')
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--meniscus-points-script',type=Path,
                   default=SCRIPT_DIR/'bone_guided_meniscus_nninteractive.py')
    p.add_argument('--meniscus-infer-script',type=Path,
                   default=SCRIPT_DIR/'infer_meniscus_nninteractive.py')
    p.add_argument('--axial-dir',default='3_Axi_PD_fs_Rt')
    p.add_argument('--coronal-dir',default='5_Cor_PD_fs_Rt')
    p.add_argument('--sagittal-dir',default='6_Sag_PD_fs_Rt')
    p.add_argument('--fluid-high',type=float,default=99.)
    p.add_argument('--fluid-low',type=float,default=93.)
    p.add_argument('--fluid-min-pool-mm3',type=float,default=5.)
    p.add_argument('--fluid-max-pools',type=int,default=12)
    p.add_argument('--meniscus-low-signal-percentile',type=float,default=35.)
    p.add_argument('--from-stage',choices=STAGES,default=STAGES[0])
    p.add_argument('--stop-after',choices=STAGES,default=STAGES[-1])
    p.add_argument('--force-from',action='store_true',
                   help='Recompute selected start stage and ALL its downstream stages')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args()
    args.case_dir=args.case_dir.expanduser().resolve()
    args.scripts_dir=args.scripts_dir.expanduser().resolve()
    base=args.case_dir/'fusion_output'
    args.generated_dir=base/'.pipeline_generated'
    start=STAGES.index(args.from_stage)
    end=STAGES.index(args.stop_after)
    if start > end:
        p.error('--from-stage cannot come after --stop-after')
    if not (0<args.fluid_low<args.fluid_high<100):
        p.error('Require 0 < --fluid-low < --fluid-high < 100')
    manifest=base/'pipeline_run_manifest.json'
    stage_results=[]
    emit('[CASE]',args.case_dir)
    emit('[MRI OUTPUT]',base/'PD_FS_triview_fused_0p4mm.nii.gz')
    emit('[STAGE RANGE]',args.from_stage,'->',args.stop_after)
    dirty=args.force_from
    try:
        for i,stage in enumerate(STAGES[start:end+1],start):
            outputs=stage_outputs(stage,base)
            exists=checked_outputs(outputs)
            rerun=dirty or not exists
            if not rerun:
                emit(f'[SKIP {i+1:02d}/{len(STAGES)}] {stage}: output files already present')
                stage_results.append({'stage':stage,'status':'skipped_existing'})
                continue
            emit(f'\n[RUN {i+1:02d}/{len(STAGES)}] {stage}')
            if stage in ('meniscus_points','meniscus_nninteractive','fluid_points') and not args.dry_run:
                source=(args.meniscus_points_script if stage=='meniscus_points' else
                        args.meniscus_infer_script if stage=='meniscus_nninteractive' else
                        args.scripts_dir/'bone_guided_fluid_nninteractive.py')
                name=('meniscus_points_for_case.py' if stage=='meniscus_points' else
                      'infer_meniscus_for_case.py' if stage=='meniscus_nninteractive' else
                      'fluid_for_case.py')
                patched=patch_module_base(source,args.generated_dir/name,base)
                emit('[SOURCE COPY]',source,'->',patched)
            old_mtimes = {str(q):q.stat().st_mtime_ns if q.is_file() else None for q in outputs}
            commands=generate_commands(stage,args,base)
            for cmd in commands:
                emit('  $',shlex.join(map(str,cmd)))
                if args.dry_run:
                    continue
                run_logged(cmd,args.case_dir,base/'pipeline_run.log')
            if not args.dry_run and not checked_outputs(outputs):
                raise RuntimeError(f'{stage}: expected outputs are missing/empty: {[str(q) for q in outputs if not q.is_file() or q.stat().st_size==0]}')
            if not args.dry_run and stage not in ('totalseg',):
                # Protect against silently keeping an OLD prediction if a stage
                # skips a target (e.g., no valid meniscus positive prompts).
                stale = [str(q) for q in outputs if old_mtimes[str(q)] is not None
                         and q.stat().st_mtime_ns <= old_mtimes[str(q)]]
                if stale:
                    raise RuntimeError(f'{stage}: outputs were not refreshed, may be stale: {stale}')
            dirty=True  # downstream artifacts depend on the results of this stage
            stage_results.append({'stage':stage,'status':'dry_run' if args.dry_run else 'completed',
                                  'outputs':[str(q) for q in outputs]})
            if not args.dry_run:
                write_manifest(manifest,args,False,stage_results)
        if args.dry_run:
            emit('[DRY RUN] Commands printed, no data modified')
        else:
            write_manifest(manifest,args,True,stage_results)
            emit('\n[DONE] RTJSV summary:',base/'jointspace_residual_volume_v3_2/jointspace_volume_summary.csv')
            emit('[QC] Review meniscus masks, footprint_qc.png, fluid candidate mask, and residual NIfTIs in Slicer.')
    except Exception as exc:
        if not args.dry_run:
            write_manifest(manifest,args,False,stage_results+[{'stage':stage,'status':'failed','error':repr(exc)}])
        raise


if __name__=='__main__':
    main()
