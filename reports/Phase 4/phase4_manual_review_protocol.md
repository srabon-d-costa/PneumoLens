# PneumoLens | Phase 4B Manual Similarity Review Protocol

**Status: PENDING HUMAN REVIEW / HOLD — no automatic approval.**

- Source fingerprint: `a7b36506fbccf5f77b5d80a73b7a53cca8515dba45ae959a8d0a62e5f219a87b`
- Split fingerprint: `6185405f23d72cecf4dfe63f7ba8002db97a79a39be21f5c3a6ef3d24f4f7a6b`
- Phase 4 V2 audit UTC: `2026-11-23T17:34:46.479512+00:00`
- Candidate population: 799 cross-final-evaluable-role pairs selected from 1,987 screened pHash candidates.
- Automated flags: 0 HIGH/REVIEW among these 799, but this is not patient independence.
- Targeted review: all 79 cross-boundary pairs with pHash distance 2 or 4.
- Supplemental review: 80 distance-6 pairs from four deterministic strata: original-test crossing vs. development train/validation crossing, each separated by same-label vs. different-label. Per stratum, 10 highest pixel correlations + 10 random among the others, random seed 2026.
- Total target: 159 pairs, deliberately sampled from 799; **640 distance-6 pairs are outside the targeted manual sample**.
- Review decisions: DISTINCT, DUPLICATE, UNCERTAIN; reviewer should compare full-resolution original images when uncertain.
- Review evidence: review sheets under `data/processed/phase4_manual_contact_sheets_PRIVATE/`; 799-row worksheet `data/processed/phase4_manual_review_worksheet_V2_PRIVATE.csv`.
- Exclude confirmed duplicate leakage and unresolved material ambiguity from any approval until addressed and fully documented.
- Patient identities unavailable; filenames are not verified patient IDs. A negative manual sample cannot establish patient independence.
- The original TEST images are only reviewed for leakage screening, never used to select validation folds or to tune a classifier.
- This workflow is **research-only**, not clinical diagnosis or deployment authorization.

## Human reviewer completion (NOT DONE)

- Reviewer: PENDING
- Review date UTC: PENDING
- Pairs actually inspected: 0 recorded (initial worksheet)
- Decision: HOLD
- Signature / rationale: PENDING
