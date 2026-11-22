# PneumoLens | Phase 2 Data Audit V2

UTC date: 2026-11-22T05:45:40.340990+00:00
Dataset source: `/kaggle/input/datasets` (directory)

## Dataset counts (image-level)
- All image files inventoried: 115,165
- Chest X-ray candidates: 5,856
- Excluded or uncertain non-CXR images: 109,309
- Original split counts: {'train': 5232, 'test': 624}
- Class counts: {'PNEUMONIA': 4273, 'NORMAL': 1583}
- Pneumonia filename subtype counts: {'BACTERIAL': 2780, 'VIRAL': 1493}

## Integrity and grouping
- Fully decoded images: 5,856 / 5,856
- SHA256 exact duplicate groups: 27 (56 image rows)
- Cross-split exact duplicate groups: 0
- Unverified numeric tokens: 3,081 (mixed subtype tokens: 809)
- Verified patient IDs available: 0 / 5,856
- Cross-split near-duplicate screening: COMPLETED (214758 candidates)

## Phase gates
- EDA gate: PASS
- Training/split-review gate: BLOCKED

## Warnings / required actions
- No approved validation split; original test must remain untouched
- Patient-level independence is NOT verified: trusted patient ID mapping incomplete
- Exact duplicates within source: 56 image rows in 27 groups
- Mixed bacterial/viral filename tokens in 809 UNVERIFIED numeric groups
- Cross-split dHash candidate pairs requiring manual review: 214758
- No verified patient-grouped validation split was requested

## Methodological limits
- Filename numeric tokens are NOT automatically verified patient IDs.
- Cross-split dHash matches are review candidates, not confirmed duplicates.
- Subtype filename tokens are not pathogen confirmation.
- Original test is never used for selecting architecture, hyperparameters, or decision threshold.
- Single-source pediatric chest radiographs do not establish clinical safety or cross-hospital generalization.
