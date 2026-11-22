"""PneumoLens preprocessing regression and structural safety tests.

From repository root: python -m unittest discover -s tests -p 'test_preprocessing.py' -v
"""
import io
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.preprocessing.transforms import (
    CONFIG_VERSION, PneumoPreprocessor, bce_pos_weight_from_approved_train_labels,
    decode_grayscale, denormalize_grayscale, letterbox_grayscale, load_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "preprocessing.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures"


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        cls.fixture = Image.open(FIXTURE_DIR / "reference_synthetic.png").convert("L")

    def test_config_contract_and_mapping(self):
        self.assertEqual(self.config["version"], CONFIG_VERSION)
        self.assertEqual(self.config["class_to_index"], {"NORMAL": 0, "PNEUMONIA": 1})
        validate_config(self.config)

    def test_shape_dtype_and_range(self):
        tensor = PneumoPreprocessor(self.config, mode="eval")(self.fixture)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_eval_determinism_and_pil_bytes_path_equivalence(self):
        proc = PneumoPreprocessor(self.config, mode="eval")
        path = FIXTURE_DIR / "reference_synthetic.png"
        a = proc(self.fixture)
        b = proc(self.fixture)
        c = proc(path)
        d = proc(path.read_bytes())
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))
        self.assertTrue(torch.equal(a, d))

    def test_reference_tensor_numerical_parity(self):
        current = PneumoPreprocessor(self.config, mode="eval")(self.fixture).numpy()
        reference = np.load(FIXTURE_DIR / "reference_eval_tensor.npy", allow_pickle=False)
        np.testing.assert_allclose(current, reference, rtol=0, atol=1e-6)

    def test_aspect_ratio_and_center_padding(self):
        a = Image.new("L", (420, 140), 200)
        result, info = letterbox_grayscale(a, size=224)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(info["resized_width"], 224)
        self.assertEqual(info["resized_height"], 75)
        self.assertEqual(info["pad_left"], 0)
        self.assertEqual(info["pad_top"], (224 - 75) // 2)
        self.assertAlmostEqual(info["content_fraction"] + info["padding_fraction"], 1.)
        self.assertEqual(result.getpixel((0, 0)), 0)

    def test_grayscale_rgb_replication_before_normalization(self):
        l = self.fixture
        rgb = Image.merge("RGB", (l, l, l))
        proc = PneumoPreprocessor(self.config, mode="eval")
        torch.testing.assert_close(proc(l), proc(rgb), rtol=0, atol=1e-6)
        tensor = proc(l)
        grayscale = denormalize_grayscale(tensor, self.config)
        self.assertEqual(grayscale.dtype, np.uint8)
        self.assertEqual(grayscale.shape, (224, 224))

    def test_mode_boundaries(self):
        with self.assertRaises(ValueError):
            PneumoPreprocessor(self.config, mode="test")
        with self.assertRaises(TypeError):
            decode_grayscale(25)
        altered = json.loads(json.dumps(self.config))
        altered["augmentation"]["horizontal_flip"] = True
        with self.assertRaises(ValueError):
            validate_config(altered)

    def test_training_is_stochastic_but_seed_reproducible(self):
        p1 = PneumoPreprocessor(self.config, mode="train", rng=random.Random(77))
        p2 = PneumoPreprocessor(self.config, mode="train", rng=random.Random(77))
        a, meta = p1(self.fixture, return_metadata=True)
        b = p2(self.fixture)
        self.assertTrue(torch.equal(a, b))
        c = p1(self.fixture)
        self.assertFalse(torch.equal(a, c))
        self.assertLessEqual(abs(meta["augmentation"]["rotation_deg"]), 5.)
        self.assertTrue(meta["augmentation"]["applied"])

    def test_eval_never_applies_augmentation(self):
        p = PneumoPreprocessor(self.config, mode="eval")
        _, meta = p(self.fixture, return_metadata=True)
        self.assertFalse(meta["augmentation"]["applied"])
        self.assertEqual(meta["augmentation"]["rotation_deg"], 0.)

    def test_positive_weight_uses_only_supplied_subset(self):
        w = bce_pos_weight_from_approved_train_labels([0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(float(w), 0.5)
        with self.assertRaises(ValueError):
            bce_pos_weight_from_approved_train_labels([1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
