# PneumoLens — Phase 3: Image preprocessing

**Engineering gate:** PASS_FOR_SPLIT_REVIEW_AND_NEXT_PHASE_PREPARATION  
**Training gate:** BLOCKED_PENDING_PATIENT_DUPLICATE_AND_VALIDATION_SPLIT_REVIEW  
**Date (UTC):** 2026-11-22T06:56:11.277513+00:00  
**Source:** `/kaggle/input/datasets/srabondcosta/pneumonia/ZhangLabData/CellData/chest_xray`  

## 1. Preprocessing contract
- Task: folder-labeled pediatric NORMAL (0) vs PNEUMONIA (1), research only.
- Image decode: Pillow with EXIF orientation; deterministically convert L/RGB inputs to 8-bit grayscale.
- Geometry: 224×224 bilinear aspect-preserving resize and centered zero padding; no crop.
- Channels: replicate one grayscale channel three times for ImageNet-pretrained DenseNet121.
- Normalization: fixed ImageNet-1K mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
- Training augmentation: ±5° rotation, ±2% translation, ±5% brightness and contrast only.
- Evaluation/inference augmentation: none. Horizontal/vertical flips and random crops are prohibited.
- ImageNet pretraining on generic natural images is external pretraining, not an additional chest X-ray dataset.

## 2. Data and transform verification
- Original train images: **5,232**; successfully transformed: **5,232**; failures: **0**.
- Original test pixel data accessed for transform design: **0**.
- Unit tests: **PASS**; fixture parity: **True**; eval repeatability: **True**.
- Unusual train-only geometry/contrast review candidates: **116** (not exclusions).
- Class-wise mean padding-fraction difference: **0.1438** (descriptive; possible shortcut risk).

## 3. File and code identity
- `src/preprocessing/transforms.py` SHA256: `017bc7e5a6a05673dd22d4f8c1091dd5bef57d87d25801f4e75447bd68be93c4`
- `configs/preprocessing.json` SHA256: `2c3b3f325cbcfeecdc28672222bb703e56bd86c5b8d016aa7244b1836b08ed95`
- The same exported transform must be imported by Kaggle training and the Streamlit inference app.

## 4. Machine-readable and visual outputs
- Summary: `reports/Phase 3/preprocessing_summary.json`.
- Tables: `reports/tables/phase3_*.csv` (class-level aggregates only).
- Figures: `reports/figures/phase3_*.png` (aggregate figures only).
- Local-only private file-level profiles and radiograph gallery: `data/processed/phase3_*PRIVATE*`.
- Regression fixture and tests: `tests/fixtures/` and `tests/test_preprocessing.py`.

## 5. Completion checks
- PASS: source counts match phase2 reference
- PASS: all source labels known
- PASS: both original train classes present
- PASS: no new validation split created
- PASS: all original train images transformed
- PASS: source modes are expected L or RGB
- PASS: all tensor outputs finite
- PASS: unit tests passed
- PASS: fixture numerical parity
- PASS: evaluation repeatability
- PASS: rgb conversion parity
- PASS: train only gallery created
- PASS: aggregate figures created
- PASS: source code and config exported

## 6. Important unresolved methodological issues
- Source numeric filename tokens are NOT verified patient IDs; train/validation grouping remains unresolved.
- Prior exact duplicates and cross-split perceptual similarity candidates require a documented review.
- No validation partition is created in this notebook; original test is locked.
- Padding, encoding and size can correlate with class; monitor shortcut learning during modeling.
- Fixed ImageNet-1K statistics assume ImageNet-pretrained 3-channel DenseNet121; reconsider for a scratch-trained architecture.
- Preprocessing output is not a clinical model and has not been externally validated.

## 7. Handoff to Phase 4
- Approve leakage-aware train/validation splits and duplicate policies **before training**.
- When approved, calculate loss class weights from that approved training subset only.
- Benchmark the planned DenseNet121 baseline; evaluate sensitivity, specificity, ROC-AUC, PR-AUC and calibration later.
- Always record versioned transform/config hashes with model checkpoints.

**Research-only limitation:** This pipeline does not establish patient independence, calibration, external validity, lesion localization, or clinical diagnostic safety.
