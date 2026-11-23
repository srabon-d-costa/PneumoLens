# PneumoLens | Phase 4 V2 — completed similarity screen

**Executed:** 2026-11-23T17:34:46.479512+00:00  
**Automated technical gate:** `PASS_AUTOMATED_CHECKS_WITH_LIMITATIONS`  
**Research training gate:** `HOLD_PENDING_HUMAN_REVIEW_AND_RISK_ACCEPTANCE`  
**Patient independence:** `UNVERIFIED` (filename series ≠ verified patients)

## Data provenance
- Full inventory anchored to Phase 1 SHA-256: `a7b36506fbccf5f77b5d80a73b7a53cca8515dba45ae959a8d0a62e5f219a87b`.
- Source inspection mode: `FULL_SOURCE_RESCAN`. All original source files were freshly hashed in this run.
- Original locked test images: 624; unchanged and not used for model/fold selection.
- Exact redundant original-train images excluded: 26.
- Redundant original-test copies preserved: 3.

## Candidate split
- Training: **4166**; validation: **1040**; test: **624**.
- Validation share: 19.977% of deduplicated original training data.
- Reproducible split SHA-256 fingerprint: `6185405f23d72cecf4dfe63f7ba8002db97a79a39be21f5c3a6ef3d24f4f7a6b`.
- Previously assigned rows moved in V2: None (none if V1 comparison not mounted).

## Exhaustive pHash search and actual image-content checks
- All unique pHash comparisons completed: **17,143,440** (no LSH candidate limit).
- pHash Hamming distance <= 6: **1,987** candidates.
- Pairs image-scored: **1,987 / 1,987**.
- Source train/test boundary candidates: **312**.
- New train/validation boundary candidates: **487**.
- Priority counts: {'LOW_SCREENED': 1956, 'EXACT_KNOWN': 31}.
- HIGH/REVIEW image-content flags: **0**, with **0** crossing original source and **0** crossing final model-relevant partitions.

**Interpretation:** pHash similarity, even after pixel and gradient scoring, is a candidate-review signal — NOT a confirmed duplicate, lesion, shared patient, or clinical label. A low-priority score is NOT proof of independence. V2 fixes the resource-truncation bug without proving clinical generalization.

## Validation checks
- PASS: `all_rows_accounted_for`
- PASS: `original_test_exactly_preserved`
- PASS: `train_validation_groups_disjoint`
- PASS: `train_test_heuristic_groups_disjoint`
- PASS: `validation_test_heuristic_groups_disjoint`
- PASS: `train_validation_pixel_sha_disjoint`
- PASS: `train_test_pixel_sha_disjoint`
- PASS: `validation_test_pixel_sha_disjoint`
- PASS: `no_pixel_sha_label_conflicts`
- PASS: `no_file_sha_label_conflicts`
- PASS: `no_heuristic_series_cross_original_test`
- PASS: `all_original_train_accounted_for`
- PASS: `both_fit_classes_present`
- PASS: `both_validation_classes_present`
- PASS: `validation_fraction_within_declared_tolerance`
- PASS: `validation_class_rate_within_declared_tolerance`
- PASS: `no_high_priority_cross_original_source`
- PASS: `no_high_priority_cross_train_val`
- PASS: `no_actionable_cross_original_source`
- PASS: `no_actionable_cross_final_roles`
- PASS: `complete_pHash_screen_and_image_scoring`
- PASS: `prior_source_fingerprint_matches`
- PASS: `repeatable_split_assignment`
- PASS: `no_exact_original_train_test_overlap`
- PASS: `no_known_exact_label_conflicts`
- PASS: `no_actionable_cross_candidate_partitions`

## Next decision
Review all flagged HIGH/REVIEW images, with priority for any cross-original or cross-train/validation pairs. Use the private review CSV and contact sheet. Do not revise the locked test to improve outcomes. Complete `split_review_signoff_TEMPLATE.md` with documented disposition and limitations before independent training authorization. Do not claim patient-level independence without reliable patient IDs or external proof.

**Research prototype only; not for clinical diagnosis, triage, or treatment.**
