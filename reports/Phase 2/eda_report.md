# PneumoLens — Phase 2: Exploratory Data Analysis

**Status:** PASS_FOR_PHASE3_PREPROCESSING_PLANNING  
**Scope:** pediatric chest X-rays; folder-labeled NORMAL/PNEUMONIA; educational research only  
**Source:** `/kaggle/input/datasets/srabondcosta/pneumonia/ZhangLabData/CellData/chest_xray` (directory)  
**Notebook:** `02_EDA.ipynb`  
**Date (UTC):** 2026-11-22T06:24:51.332541+00:00

## 1. Dataset inventory and source partitions

A total of **5,856 CXR images** were discovered in the selected chest X-ray tree. Images from unrelated OCT folders were not included. Source partitions were preserved and were not shuffled, resampled or re-labeled. Test-set images were not used for pixel-intensity statistics, galleries, or preprocessing selection.

| Original split | NORMAL | PNEUMONIA | Total images |
| --- | --- | --- | --- |
| train | 1349 | 3883 | 5232 |
| test | 234 | 390 | 624 |

The original training set contains **1,349 NORMAL** and **3,883 PNEUMONIA** images (pneumonia-to-normal ratio: **2.8784:1**). Imbalance handling, if used, must be fitted on the *future approved training subset*, not test or validation data.

## 2. Native geometry and channel modes

X-ray image **headers** were read for all partitions: **0** header failure(s). Width, height, aspect ratio and original image modes were recorded. All geometry/encoding figures used for design comparisons draw on original **train** images.

| Image mode | Number of source images (all splits) |
| --- | --- |
| L | 5573 |
| RGB | 283 |

The train-only size distributions and aspect-ratio scatter are in `reports/figures/phase2_02_train_dimensions.png`. Large native variation supports testing **aspect-preserving** resize plus padding rather than squashing every source to a 224×224 square.

## 3. Train-only pixel properties and review flags

Successfully profiled **5,232/5,232** labeled training images with an aspect-preserving, up-to-256-pixel-side grayscale thumbnail. These descriptive values are not full-resolution intensity measurements or final model-input normalization statistics.

Mean of per-image training thumbnail means: **123.05/255**. Mean of per-image thumbnail intensity standard deviations: **56.52/255**. See published aggregate tables for class-wise distributions and local, git-ignored `data/processed/` for image-level review records.

| Heuristic review flag | Training images | Fraction of decoded train |
| --- | --- | --- |
| extreme_aspect | 56 | 1.07% |
| low_thumbnail_contrast | 6 | 0.11% |
| very_dark_thumbnail | 0 | 0.00% |
| very_bright_thumbnail | 1 | 0.02% |
| high_dark_fraction | 0 | 0.00% |
| high_bright_fraction | 0 | 0.00% |

Flags may overlap; they do **not** prove corruption, artifacts, poor medical quality, or exclusion eligibility. No image was automatically dropped.

## 4. Provisional preprocessing decisions (NOT final training settings)

- Use **224×224** as a reasonable first baseline input resolution, subject to visual inspection.
- Preserve aspect ratio using **letterboxing/padding**, not anisotropic stretching or aggressive cropping.
- Handle both grayscale and RGB source files deterministically; for a pretrained 3-channel CNN, consider grayscale-to-3-channel replication.
- Use the backbone's declared normalization or derive statistics using the finalized *train-only* transform; do not reuse pre-padding EDA moments blindly.
- Restrict mild geometric/brightness augmentation to the approved training split; avoid vertical flips.
- Address the observed class imbalance only with training-specific loss weighting/sampling.
- Keep original test locked and do **not** treat the numeric filename tokens as verified patient IDs.

## 5. Completion checks

- PASS: one chest xray dataset selected
- PASS: at least one train normal and pneumonia
- PASS: all source images have recognized split and class
- PASS: no conflicting filename class tags
- PASS: all source image headers readable
- PASS: all original train headers readable
- PASS: all original train pixels profiled
- PASS: test pixels not used for eda

**EDA gate: PASS_FOR_PHASE3_PREPROCESSING_PLANNING.** This gate is independent of approval to begin model training.

## 6. Open methodological limitations

- No existing validation partition; Phase 2 DOES NOT create one
- Patient-level independence remains unverified; filename numeric tokens are NOT verified patient IDs
- Phase 1 flagged exact duplicates and many dHash candidates; review them before approving training splits
- No external validation or clinical reliability can be claimed from this single dataset

The label PNEUMONIA and the filename tokens BACTERIAL/VIRAL are dataset labels, not microbiologically verified pathogen diagnoses. Image-level descriptive findings do not establish clinical accuracy, patient independence, or external validity. Pixel distributions can be affected by cropping, contrast, acquisition pipelines and text/edge artifacts.

## 7. File locations

- Summary: `reports/Phase 2/eda_summary.json`
- Aggregate CSVs: `reports/tables/phase2_*.csv`
- Reproducible charts: `reports/figures/phase2_*.png`
- Local-only image-level review and private gallery: `data/processed/phase2_*PRIVATE*`

## 8. Phase 3 handoff

Before starting **Phase 3: Preprocessing**, review the EDA figures, verify label and modality scope, and finalize a deterministic 224×224 transform with a visual sanity check. **Do not train or tune on the test partition.** Resolve Phase 1 patient/group-independence and near-duplicate findings before choosing a validation split and starting benchmark experiments.
