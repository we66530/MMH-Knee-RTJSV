# MMH Knee RTJSV Pipeline

Automated MRI-based Residual Tibiofemoral Joint-Space Volume (RTJSV)
quantification pipeline.

## Research objective

To investigate whether MRI-derived RTJSV is associated with
osteoarthritis severity.

RTJSV is an exploratory imaging biomarker, not a direct
measurement of cartilage volume.

## Pipeline

1. Three-view PD-FS MRI reconstruction (0.4 mm isotropic grid)
2. TotalSegmentator bone priors
3. nnInteractive bone segmentation
4. Medial and lateral meniscus segmentation
5. High-signal fluid candidate detection
6. Anatomically constrained joint-space ROI (v3.2)
7. Initial RTJSV quantification
8. Intercondylar soft-tissue segmentation
9. Corrected medial/lateral RTJSV quantification

## Current configuration

- ROI algorithm: v3.2
- Intercondylar positive prompts: 8
- Negative prompts: 0
- Inner extension: 10 mm
- Intensity range: P0-P95
- Minimum candidate depth: 0.4 mm

## Requirements

- Python
- PyTorch with CUDA
- nnInteractive
- TotalSegmentator
- Medical imaging dependencies used by the scripts

The current implementation uses the mricore and totalseg5090
Conda environments.

## Usage

Activate the mricore environment:

    conda activate mricore

Dry-run:

    python run_pipeline.py --case-dir /path/to/case --mode full --dry-run

Full inference:

    python run_pipeline.py --case-dir /path/to/case --mode full

## Research status

Research prototype. End-to-end reproducibility on new cases
and clinical OA grading validation remain pending.

## Data privacy

No patient DICOM, NIfTI, clinical records, model weights,
or patient-level outputs are distributed in this repository.
