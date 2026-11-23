

"""Phase 4 unit tests: run with `python -m unittest discover -s tests`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.split import (
    DisjointSet, build_split, duplicate_violations, pair_similarity,
    perceptual_hash, series_candidate, split_fingerprint, training_gate,
    screen_image_similarity,
)


class TestSplitIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = {
            "validation": {"n_splits": 5, "candidate_seeds": [2026, 2027, 2031],
                           "target_fraction": 0.20, "max_abs_fraction_deviation": 0.10,
                           "max_abs_positive_rate_deviation": 0.10},
            "near_duplicate_screen": {"max_phash_distance": 6},
        }

    def fixture(self, conflict=False, cross_test_exact=False):
        rows = []
        for source_split in ("train", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                count = 40 if source_split == "train" else 10
                for k in range(count):
                    token = 10000 + k
                    prefix = "NORMAL" if label == "NORMAL" else "BACTERIA"
                    file = f"{prefix}-{token}-0001.jpeg"
                    rel = f"{source_split}/{label}/{file}"
                    signature = f"HASH:{source_split}:{label}:{k}"
                    rows.append({"relative_path": rel, "original_split": source_split,
                                 "label": label, "label_index": 0 if label == "NORMAL" else 1,
                                 "series_group_UNVERIFIED": series_candidate(file)[0],
                                 "file_sha256": signature, "decoded_sha256": signature,
                                 "phash64": f"{k:016x}", "filename": file})
        out = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
        # A repeat under a different numeric token should not cross validation.
        train_normal = out.index[(out.original_split == "train") & (out.label == "NORMAL")]
        out.loc[train_normal[1], ["file_sha256", "decoded_sha256"]] = out.loc[train_normal[0], ["file_sha256", "decoded_sha256"]].to_numpy()
        if cross_test_exact:
            test_normal = out.index[(out.original_split == "test") & (out.label == "NORMAL")]
            out.loc[test_normal[0], ["file_sha256", "decoded_sha256"]] = out.loc[train_normal[0], ["file_sha256", "decoded_sha256"]].to_numpy()
        if conflict:
            train_p = out.index[(out.original_split == "train") & (out.label == "PNEUMONIA")]
            out.loc[train_p[0], ["file_sha256", "decoded_sha256"]] = out.loc[train_normal[0], ["file_sha256", "decoded_sha256"]].to_numpy()
        return out

    def test_series_key_preserves_prefix(self):
        self.assertNotEqual(series_candidate("BACTERIA-1234-0001.jpeg")[0],
                            series_candidate("VIRUS-1234-0001.jpeg")[0])

    def test_unparsed_token_not_patient(self):
        self.assertFalse(series_candidate("person3.jpeg")[1])

    def test_dct_phash_repeatable(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (96, 96), dtype=np.uint8)
        self.assertEqual(perceptual_hash(image), perceptual_hash(image.copy()))
        self.assertTrue(isinstance(perceptual_hash(image), int))

    def test_image_similarity_identity(self):
        image = np.arange(96 * 96, dtype=np.uint32).reshape(96, 96).astype(np.uint8)
        corr, grad_corr, mae = pair_similarity(image, image.copy())
        self.assertAlmostEqual(corr, 1, places=5)
        self.assertAlmostEqual(grad_corr, 1, places=5)
        self.assertEqual(mae, 0)

    def test_disjoint_set(self):
        ds = DisjointSet(5)
        ds.merge(0, 3)
        self.assertEqual(ds.find(0), ds.find(3))
        self.assertNotEqual(ds.find(1), ds.find(3))

    def test_train_test_duplicate_detected(self):
        v = duplicate_violations(self.fixture(cross_test_exact=True))
        self.assertGreater(v["decoded_sha256_cross_original_split_groups"], 0)

    def test_conflicting_label_detected(self):
        v = duplicate_violations(self.fixture(conflict=True))
        self.assertGreater(v["decoded_sha256_conflicting_label_groups"], 0)

    def test_candidate_split_reproducible(self):
        d = self.fixture()
        pair = pd.DataFrame()
        a, diag_a = build_split(d, pair, self.policy)
        b, diag_b = build_split(d, pair.copy(), self.policy)
        self.assertEqual(split_fingerprint(a), split_fingerprint(b))
        self.assertEqual(diag_a["selected_fold"], diag_b["selected_fold"])

    def test_locked_test_and_dedup(self):
        d, stats = build_split(self.fixture(), pd.DataFrame(), self.policy)
        self.assertTrue(stats["qa"]["original_test_exactly_preserved"])
        self.assertEqual(int((d.split_role == "TEST_LOCKED").sum()), 20)
        self.assertEqual(int((d.split_role == "EXCLUDED_EXACT_COPY").sum()), 1)
        self.assertTrue(stats["qa"]["train_validation_groups_disjoint"])

    def test_strong_grouping_does_not_cross_fit_val(self):
        raw = self.fixture()
        locs = raw.index[(raw.original_split == "train") & (raw.label == "NORMAL")]
        pa, pb = int(locs[3]), int(locs[4])
        paired = pd.DataFrame([{
            "index_a": pa, "index_b": pb,
            "source_split_a": "train", "source_split_b": "train",
            "screening_priority": "HIGH"}])
        d, diag = build_split(raw, paired, self.policy)
        self.assertEqual(d.at[pa, "candidate_group"], d.at[pb, "candidate_group"])
        self.assertTrue(diag["qa"]["no_high_priority_cross_train_val"])

    def test_training_gate_never_automatically_approves(self):
        _, stats = build_split(self.fixture(), pd.DataFrame(), self.policy)
        result = training_gate(stats, {"screen_incomplete_due_to_resource_budget": False,
                                      "complete_exhaustive_phash_search": True,
                                      "all_phash_candidates_image_scored": True}, self.policy)
        self.assertEqual(result["research_training_gate"], "HOLD_PENDING_HUMAN_REVIEW_AND_RISK_ACCEPTANCE")
        self.assertEqual(result["patient_independence"], "UNVERIFIED")

    def test_near_match_screen_flags_structurally_identical_pair(self):
        rng = np.random.default_rng(444)
        image = rng.integers(20, 210, (96, 96), dtype=np.uint8)
        transformed = np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        df = pd.DataFrame([
            {"phash64": f"{perceptual_hash(image):016x}", "relative_path": "train/NORMAL/one.jpeg",
             "file_sha256": "f1", "decoded_sha256": "p1", "original_split": "train", "label": "NORMAL",
             "aspect_ratio": 1.0},
            {"phash64": f"{perceptual_hash(transformed):016x}", "relative_path": "test/NORMAL/two.jpeg",
             "file_sha256": "f2", "decoded_sha256": "p2", "original_split": "test", "label": "NORMAL",
             "aspect_ratio": 1.0},
        ])
        limits = {"near_duplicate_screen": {
            "max_phash_distance": 6, "max_band_candidate_pairs": 1000, "max_scored_pairs": 1000,
            "strong_phash_max": 3, "strong_corr_min": 0.995, "strong_gradient_min": 0.975,
            "strong_mae_max": 0.045, "strong_aspect_max": 0.05,
            "review_corr_min": 0.975, "review_gradient_min": 0.93, "review_aspect_max": 0.10,
        }}
        queue, diagnostic = screen_image_similarity(df, np.stack([image, transformed]), limits)
        self.assertGreaterEqual(len(queue), 1)
        self.assertEqual(queue.iloc[0].screening_priority, "HIGH")
        self.assertEqual(diagnostic["high_priority_pairs"], 1)

    def test_strong_cross_original_pair_blocks_technical_gate(self):
        raw = self.fixture()
        i = int(raw.index[raw.original_split.eq("train")][0])
        j = int(raw.index[raw.original_split.eq("test")][0])
        pairs = pd.DataFrame([{"index_a": i, "index_b": j,
                               "source_split_a": "train", "source_split_b": "test",
                               "screening_priority": "HIGH"}])
        _, stats = build_split(raw, pairs, self.policy)
        self.assertFalse(stats["qa"]["no_high_priority_cross_original_source"])
        release = training_gate(stats, {"screen_incomplete_due_to_resource_budget": False,
                                      "complete_exhaustive_phash_search": True,
                                      "all_phash_candidates_image_scored": True}, self.policy)
        self.assertEqual(release["technical_split_gate"], "FAIL_NEEDS_INVESTIGATION")



class TestCompletePerceptualScreen(unittest.TestCase):
    def test_matches_independent_bruteforce(self):
        from src.data.split import exhaustive_phash_candidates
        rng = np.random.default_rng(2026)
        values = rng.integers(0, np.iinfo(np.uint64).max, 73, dtype=np.uint64)
        values[3] = values[8] ^ np.uint64(0b10101)
        values[9] = values[8] ^ np.uint64(0b111111)
        data = pd.DataFrame({"phash64": [f"{int(v):016x}" for v in values]})
        actual, info = exhaustive_phash_candidates(data, max_distance=6, block_size=7)
        observed = {(int(a), int(b), int(d)) for a,b,d in actual.itertuples(index=False, name=None)}
        expected = {(i,j,(int(values[i]) ^ int(values[j])).bit_count())
                    for i in range(len(values)) for j in range(i+1,len(values))
                    if (int(values[i]) ^ int(values[j])).bit_count() <= 6}
        self.assertEqual(observed, expected)
        self.assertTrue(info["complete_exhaustive_phash_search"])
        self.assertEqual(info["total_image_pairs_compared"], 73*72//2)

    def test_complete_search_many_identical_hashes_no_truncation(self):
        from src.data.split import exhaustive_phash_candidates
        df = pd.DataFrame({"phash64": ["0000000000000000"]*22})
        result, info = exhaustive_phash_candidates(df, block_size=4)
        self.assertEqual(len(result), 22*21//2)
        self.assertFalse(info["screen_incomplete_due_to_resource_budget"])

    def test_source_image_hash_mismatch_blocks_resume(self):
        import tempfile
        from PIL import Image
        from src.data.split import Source, load_candidate_thumbnails
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "train" / "NORMAL"
            p.mkdir(parents=True)
            Image.new("L", (64,64), 80).save(p / "a.jpeg")
            df = pd.DataFrame({"relative_path": ["train/NORMAL/a.jpeg"],
                               "file_sha256": ["not_the_actual_sha"], "phash64": ["0000000000000000"]})
            pairs = pd.DataFrame([{"index_a":0,"index_b":0,"phash_hamming":0}])
            with self.assertRaises(RuntimeError):
                load_candidate_thumbnails(Source("folder",Path(tmp)),df,pairs)

    def test_no_auto_signoff_even_full_screen(self):
        from src.data.split import training_gate
        gate=training_gate({"qa":{"source_ok": True}},
            {"screen_incomplete_due_to_resource_budget": False,
             "complete_exhaustive_phash_search": True,
             "all_phash_candidates_image_scored": True},
            {"near_duplicate_screen": {"max_phash_distance": 6}})
        self.assertIn("HOLD", gate["research_training_gate"])
        self.assertEqual(gate["patient_independence"], "UNVERIFIED")

    def test_review_queue_keeps_high_and_review_only(self):
        from src.data.split import action_review_queue
        scored = pd.DataFrame({"screening_priority": ["LOW_SCREENED","HIGH","EXACT_KNOWN","REVIEW"]})
        out = action_review_queue(scored)
        self.assertEqual(list(out.screening_priority), ["HIGH","REVIEW"])


if __name__ == "__main__":
    unittest.main()

