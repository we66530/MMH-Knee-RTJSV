#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatic Medial / Lateral Meniscus Segmentation
using nnInteractive.

Version 1.0

Input:
    PD_FS_triview_fused_0p4mm.nii.gz
    oblique_boundary_v12/prompts_with_negative.json

Workflow:
    1. Load existing nnInteractive model
    2. Load fused knee MRI
    3. Read positive and negative prompts
    4. Segment medial meniscus
    5. Segment lateral meniscus
    6. Save independent NIfTI masks
    7. Save multilabel preview and QC report

No Agent required.
No model download.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/media/we66530/Transcend/MMH_knee_dcm/"
    "PACS_downloads/37321865_20260513/fusion_output"
)

MRI_PATH = (
    BASE / "PD_FS_triview_fused_0p4mm.nii.gz"
)

PROMPT_PATH = (
    BASE
    / "meniscus_guided_nninteractive"
    / "oblique_boundary_v12"
    / "prompts_with_negative.json"
)

MODEL_DIR = (
    Path.home()
    / ".nninteractive"
    / "models"
    / "nnInteractive_v1.0"
)

OUTPUT_DIR = (
    BASE
    / "meniscus_guided_nninteractive"
    / "nninteractive_inference"
)

LABEL_MAP = {
    "medial": 1,
    "lateral": 2,
}


# ============================================================
# UTILITIES
# ============================================================

def log(message):
    print(message, flush=True)


def save_nifti(array, reference, path):

    output = sitk.GetImageFromArray(
        array.astype(np.uint8)
    )

    output.CopyInformation(reference)

    sitk.WriteImage(
        output,
        str(path),
        True
    )

    log(f"[SAVED] {path.name}")


def load_prompts(path, image_shape):

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt JSON not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid prompt JSON")

    prompts = {}

    for side in ["medial", "lateral"]:

        positives = [
            p for p in data.get("positive", [])
            if p.get("side") == side
        ]

        negatives = [
            p for p in data.get("negative", [])
            if p.get("side") == side
        ]

        # The boundary script may reject some positives.
        # Never silently reuse rejected points.
        if not positives:

            log(
                f"[WARNING] {side}: "
                "no accepted positive points"
            )

            prompts[side] = None
            continue

        def validate_points(items, point_type):

            result = []
            seen = set()

            for p in items:

                coords = p.get("voxel_zyx")

                if coords is None or len(coords) != 3:
                    raise ValueError(
                        f"{side}: invalid {point_type} point"
                    )

                point = tuple(
                    int(v) for v in coords
                )

                if any(
                    v < 0 or v >= size
                    for v, size in zip(
                        point,
                        image_shape
                    )
                ):
                    raise ValueError(
                        f"{side}: point outside MRI: {point}"
                    )

                if point not in seen:
                    result.append(point)
                    seen.add(point)

            return result

        positive_points = validate_points(
            positives,
            "positive"
        )

        negative_points = validate_points(
            negatives,
            "negative"
        )

        # A point cannot be both positive and negative.
        conflicts = (
            set(positive_points)
            &
            set(negative_points)
        )

        if conflicts:
            raise ValueError(
                f"{side}: conflicting prompts: {conflicts}"
            )

        prompts[side] = {
            "positive": positive_points,
            "negative": negative_points
        }

        log(
            f"[PROMPTS] {side}: "
            f"{len(positive_points)} positive, "
            f"{len(negative_points)} negative"
        )

    return prompts


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_nninteractive(model_dir):

    import torch

    from nnInteractive.inference.inference_session import (
        nnInteractiveInferenceSession
    )

    checkpoint = (
        model_dir / "fold_0" / "checkpoint_final.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    log(f"[GPU] {torch.cuda.get_device_name(0)}")
    log(f"[MODEL] {model_dir}")

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda:0"),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=min(
            os.cpu_count() or 4,
            16
        ),
        do_autozoom=True,
    )

    session.initialize_from_trained_model_folder(
        str(model_dir)
    )

    return session


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    session,
    target_buffer,
    prompts,
    reference_image,
):

    import torch

    predictions = {}
    report = {}

    voxel_volume_mm3 = float(
        np.prod(reference_image.GetSpacing())
    )

    for side in ["medial", "lateral"]:

        log(f"\n{'=' * 45}")
        log(f"[SEGMENT] {side.upper()}")
        log(f"{'=' * 45}")

        current_prompts = prompts.get(side)

        if current_prompts is None:
            log(f"[SKIP] {side}: no positive prompts")
            report[side] = {
                "status": "skipped_no_positive"
            }
            continue

        # Start independent segmentation.
        session.reset_interactions()

        target_buffer.zero_()

        positives = current_prompts["positive"]
        negatives = current_prompts["negative"]

        # ----------------------------------------
        # Positive prompts
        # ----------------------------------------

        for i, point in enumerate(positives, 1):

            log(
                f"[POSITIVE {i}/{len(positives)}] "
                f"ZYX={point}"
            )

            session.add_point_interaction(
                point,
                include_interaction=True
            )

        # Save initial segmentation before negatives.
        initial_mask = (
            target_buffer.cpu().numpy() > 0
        ).copy()

        initial_path = (
            OUTPUT_DIR
            / f"{side}_meniscus_positive_only.nii.gz"
        )

        save_nifti(
            initial_mask,
            reference_image,
            initial_path
        )

        initial_voxels = int(initial_mask.sum())

        log(
            f"[INITIAL] {initial_voxels:,} voxels"
        )

        # ----------------------------------------
        # Negative prompts
        # ----------------------------------------

        for i, point in enumerate(negatives, 1):

            log(
                f"[NEGATIVE {i}/{len(negatives)}] "
                f"ZYX={point}"
            )

            session.add_point_interaction(
                point,
                include_interaction=False
            )

        # Save final mask.
        final_mask = (
            target_buffer.cpu().numpy() > 0
        ).copy()

        final_path = (
            OUTPUT_DIR
            / f"{side}_meniscus.nii.gz"
        )

        save_nifti(
            final_mask,
            reference_image,
            final_path
        )

        final_voxels = int(final_mask.sum())

        # ----------------------------------------
        # Quality control
        # ----------------------------------------

        intersection = int(
            np.logical_and(
                initial_mask,
                final_mask
            ).sum()
        )

        removed_voxels = int(
            np.logical_and(
                initial_mask,
                ~final_mask
            ).sum()
        )

        added_voxels = int(
            np.logical_and(
                ~initial_mask,
                final_mask
            ).sum()
        )

        report[side] = {
            "status": "completed",
            "positive_points": [
                list(p) for p in positives
            ],
            "negative_points": [
                list(p) for p in negatives
            ],
            "initial_voxels": initial_voxels,
            "final_voxels": final_voxels,
            "initial_volume_ml": (
                initial_voxels
                * voxel_volume_mm3
                / 1000
            ),
            "final_volume_ml": (
                final_voxels
                * voxel_volume_mm3
                / 1000
            ),
            "intersection_voxels": intersection,
            "removed_voxels": removed_voxels,
            "added_voxels": added_voxels,
        }

        predictions[side] = final_mask

        log(
            f"[FINAL] {final_voxels:,} voxels"
        )

        log(
            f"[REMOVED] {removed_voxels:,} voxels"
        )

        log(
            f"[ADDED] {added_voxels:,} voxels"
        )

        log(
            f"[VOLUME] "
            f"{report[side]['final_volume_ml']:.2f} mL"
        )

        if final_voxels == 0:
            log(
                "[WARNING] Empty final segmentation!"
            )

        del initial_mask

    return predictions, report


# ============================================================
# MULTILABEL OUTPUT
# ============================================================

def create_multilabel(
    predictions,
    reference_image,
    report,
):

    shape = tuple(
        reversed(reference_image.GetSize())
    )

    multilabel = np.zeros(
        shape,
        dtype=np.uint8
    )

    overlap = np.zeros(
        shape,
        dtype=np.uint8
    )

    for side, mask in predictions.items():

        overlap += mask.astype(np.uint8)

        multilabel[
            mask & (multilabel == 0)
        ] = LABEL_MAP[side]

    overlap_voxels = int(
        np.count_nonzero(overlap > 1)
    )

    report["multilabel"] = {
        "overlap_voxels": overlap_voxels,
        "label_map": LABEL_MAP
    }

    save_nifti(
        multilabel,
        reference_image,
        OUTPUT_DIR / "meniscus_multilabel.nii.gz"
    )

    log(
        f"[OVERLAP] {overlap_voxels:,} voxels"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR)
    )

    parser.add_argument(
        "--prompt-json",
        type=str,
        default=str(PROMPT_PATH)
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    log("\n========================================")
    log(" Automatic Meniscus nnInteractive v1.0")
    log(" Positive + Negative Prompt Inference")
    log("========================================\n")

    # ----------------------------------------
    # Load image
    # ----------------------------------------

    image = sitk.ReadImage(
        str(MRI_PATH)
    )

    if image.GetDimension() != 3:
        raise ValueError(
            "Expected 3D MRI"
        )

    mri = sitk.GetArrayFromImage(
        image
    )

    log(f"[MRI] {MRI_PATH}")
    log(f"[SHAPE ZYX] {mri.shape}")
    log(f"[SPACING] {image.GetSpacing()}")

    # ----------------------------------------
    # Load prompts
    # ----------------------------------------

    prompts = load_prompts(
        Path(args.prompt_json),
        mri.shape
    )

    if all(
        p is None
        for p in prompts.values()
    ):
        raise RuntimeError(
            "No valid positive prompts for either meniscus"
        )

    # ----------------------------------------
    # Initialize nnInteractive
    # ----------------------------------------

    session = initialize_nninteractive(
        Path(args.model_dir).expanduser()
    )

    # IMPORTANT:
    # Preserve original MRI intensities.
    # Array order from SimpleITK is Z,Y,X.
    image_input = np.ascontiguousarray(
        mri[None]
    )

    session.set_image(
        image_input
    )

    import torch

    target_buffer = torch.zeros(
        mri.shape,
        dtype=torch.uint8
    )

    session.set_target_buffer(
        target_buffer
    )

    # ----------------------------------------
    # Run segmentation
    # ----------------------------------------

    predictions, report = run_inference(
        session,
        target_buffer,
        prompts,
        image
    )

    create_multilabel(
        predictions,
        image,
        report
    )

    # ----------------------------------------
    # Save report
    # ----------------------------------------

    report_path = (
        OUTPUT_DIR / "inference_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    log(f"[SAVED] {report_path}")

    log("\n========================================")
    log("[DONE] Meniscus segmentation completed")
    log(f"[OUTPUT] {OUTPUT_DIR}")
    log("========================================")


if __name__ == "__main__":
    main()
