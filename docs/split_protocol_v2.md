# PneumoLens | Phase 4 V2 Methods Contract

1. Same Phase-1-fingerprinted Kermany / ZhangLab pediatric CXR source; original test images stay locked.
2. V1 exact duplicate handling is preserved: only redundant original-training copies are excluded, while original-test copies remain unchanged.
3. The 64-bit DCT pHash is compared **exhaustively** for all unordered image pairs in constant-size blocks. A pHash Hamming distance <=6 defines a candidate for **image-content screening**, not a duplicate.
4. Candidate radiographs are read only for integrity screening. Their raw SHA-256 and DCT pHash are confirmed against the cached source inventory when resuming. No test image is used to choose the validation fold or tune its parameters.
5. All pHash candidates receive 96x96 grayscale pixel correlation, gradient correlation, intensity MAE and aspect-delta checks using previously frozen V1 thresholds. HIGH/REVIEW are not confirmed duplicate labels.
6. All HIGH pairs entirely inside original training become conservative related-image grouping constraints. The validation fold is chosen only within deduplicated original training using fixed StratifiedGroupKFold seeds/thresholds. Any discovered cross-partition actionable candidate blocks the automated gate.
7. Because trusted patient identifiers are unavailable, filename-series groups are explicitly weak proxies. A passing technical gate is not a guarantee of patient-level independence; no automatic research-training signoff is granted.
8. Original test duplicate membership is maintained for reproducibility. In subsequent evaluation, report a separate locked-test duplicate sensitivity analysis without changing the official fixed test set.
9. Private source manifests and images are stored under data/processed/, excluded from GitHub. Public reports are aggregate-only.
10. To finalize: audit relevant image pairs, resolve concerns without selecting to maximize test outcomes, document limitations and sign the research-use review form.
