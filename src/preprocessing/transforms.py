"""PneumoLens: portable, deterministic pediatric chest X-ray preprocessing.

Research/educational use only. The source and configs travel together in the
Phase 3 output ZIP and should be reused by Kaggle, training and Streamlit.
No normalization or augmentation is fitted on test or validation images.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


CONFIG_VERSION = "pneumolens-preprocess-v1"


def load_config(config_path: str | Path) -> dict:
    """Load the sole preprocessing configuration from JSON."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != CONFIG_VERSION:
        raise ValueError(f"Expected config version {CONFIG_VERSION!r}")
    if config.get("task") != "binary_normal_pneumonia":
        raise ValueError("Unexpected task. Do not silently change class mapping.")
    if config.get("class_to_index") != {"NORMAL": 0, "PNEUMONIA": 1}:
        raise ValueError("Class mapping must be NORMAL:0, PNEUMONIA:1")
    image_size = config.get("image_size")
    if type(image_size) is not int or not 64 <= image_size <= 1024:
        raise ValueError("image_size must be an integer in [64, 1024].")
    if config.get("convert_mode") != "L_then_RGB_replication":
        raise ValueError("Only deterministic luminance-to-3-channel conversion is supported.")
    if config.get("resize_interpolation") != "bilinear" or config.get("pad_value") != 0:
        raise ValueError("Unexpected interpolation or padding value.")
    norm = config.get("normalization", {})
    if norm.get("policy") != "imagenet1k_pretrained_fixed":
        raise ValueError("Normalization policy must explicitly be ImageNet fixed statistics.")
    for key in ("mean", "std"):
        vals = norm.get(key)
        if not isinstance(vals, list) or len(vals) != 3:
            raise ValueError(f"Normalization {key} must contain 3 values.")
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in vals):
            raise ValueError(f"Normalization {key} values must be finite numbers.")
    if any(float(v) <= 0 for v in norm["std"]):
        raise ValueError("Normalization std must be positive.")
    aug = config.get("augmentation", {})
    for key in ("max_degrees", "max_translate_fraction", "max_brightness_delta", "max_contrast_delta"):
        v = aug.get(key)
        max_allowed = 10.0 if key == "max_degrees" else 0.25
        if not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= float(v) <= max_allowed:
            raise ValueError(f"Invalid augmentation setting: {key}")
    if any(aug.get(key, False) for key in ("horizontal_flip", "vertical_flip", "random_crop")):
        raise ValueError("Flips and random crops are not permitted in this preprocessing version.")


def decode_grayscale(source: Image.Image | bytes | bytearray | str | Path) -> Image.Image:
    """Load bytes/path/PIL into a detached, EXIF-oriented, 8-bit grayscale image.

    Different JPEG formats are deliberately normalized to the same channel
    semantics. No per-image contrast equalization or enhancement is applied.
    """
    if isinstance(source, Image.Image):
        original = source
        original.load()
        corrected = ImageOps.exif_transpose(original)
        result = corrected.convert("L")
    elif isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as original:
            corrected = ImageOps.exif_transpose(original)
            result = corrected.convert("L")
    elif isinstance(source, (str, Path)):
        with Image.open(source) as original:
            corrected = ImageOps.exif_transpose(original)
            result = corrected.convert("L")
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")
    if result.width < 1 or result.height < 1:
        raise ValueError("The image has zero dimensions.")
    return result.copy()


def letterbox_grayscale(image: Image.Image, size: int, fill_value: int = 0):
    """Bilinear resize without geometric distortion; center-pad to square.

    Uses integer pixel sizing with explicit half-up rounding (no model crop).
    """
    if image.mode != "L":
        raise ValueError("letterbox_grayscale expects a decoded L-mode image.")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Non-positive width or height.")
    scale = min(size / w, size / h)
    new_w = min(size, max(1, int(w * scale + 0.5)))
    new_h = min(size, max(1, int(h * scale + 0.5)))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas = Image.new("L", (size, size), color=int(fill_value))
    canvas.paste(resized, (left, top))
    meta = {
        "original_width": w, "original_height": h,
        "resized_width": new_w, "resized_height": new_h,
        "pad_left": left, "pad_top": top,
        "pad_right": size - new_w - left,
        "pad_bottom": size - new_h - top,
        "content_fraction": (new_w * new_h) / (size * size),
        "padding_fraction": 1.0 - (new_w * new_h) / (size * size),
        "resize_factor": scale,
    }
    return canvas, meta


class PneumoPreprocessor:
    """Single implementation shared by Kaggle and local inference.

    mode='eval' is deterministic; mode='train' has mild randomness. A
    per-worker seeded `random.Random` instance can be passed during training.
    """
    def __init__(self, config: Mapping[str, Any], mode: str = "eval", rng=None):
        validate_config(config)
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")
        self.config = dict(config)
        self.mode = mode
        self.rng = rng if rng is not None else random
        self.size = int(config["image_size"])
        norm = config["normalization"]
        self.mean = np.array(norm["mean"], dtype=np.float32)[:, None, None]
        self.std = np.array(norm["std"], dtype=np.float32)[:, None, None]

    def __call__(self, source, *, return_metadata: bool = False):
        image = decode_grayscale(source)
        aug_info = {"applied": False, "rotation_deg": 0., "translate_x_px": 0,
                    "translate_y_px": 0, "brightness_factor": 1., "contrast_factor": 1.}
        if self.mode == "train":
            aug = self.config["augmentation"]
            brightness = 1 + self.rng.uniform(-float(aug["max_brightness_delta"]),
                                                 float(aug["max_brightness_delta"]))
            contrast = 1 + self.rng.uniform(-float(aug["max_contrast_delta"]),
                                               float(aug["max_contrast_delta"]))
            # Apply photometric jitter before padding to keep the padding constant.
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            aug_info.update({"applied": True, "brightness_factor": brightness,
                             "contrast_factor": contrast})

        image, geometry = letterbox_grayscale(image, self.size,
                                               fill_value=int(self.config["pad_value"]))
        if self.mode == "train":
            degrees = float(aug["max_degrees"])
            translate_px = self.size * float(aug["max_translate_fraction"])
            angle = self.rng.uniform(-degrees, degrees)
            dx = int(round(self.rng.uniform(-translate_px, translate_px)))
            dy = int(round(self.rng.uniform(-translate_px, translate_px)))
            # No flips, crops, elastic warps, histogram enhancement, or perspective changes.
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR,
                                 expand=False, translate=(dx, dy),
                                 fillcolor=int(self.config["pad_value"]))
            aug_info.update({"rotation_deg": angle,
                             "translate_x_px": dx, "translate_y_px": dy})

        arr = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        chw = np.repeat(arr[None, :, :], repeats=3, axis=0)
        normalized = np.ascontiguousarray((chw - self.mean) / self.std,
                                           dtype=np.float32)
        result = torch.from_numpy(normalized)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Non-finite tensor produced from input image.")
        if return_metadata:
            return result, {**geometry, "augmentation": aug_info,
                            "mode": self.mode,
                            "normalization": self.config["normalization"]["policy"]}
        return result


def denormalize_grayscale(tensor: torch.Tensor, config: Mapping[str, Any]) -> np.ndarray:
    """8-bit grayscale display of normalized channel-0 (visualization only)."""
    mean = float(config["normalization"]["mean"][0])
    std = float(config["normalization"]["std"][0])
    return np.clip((tensor[0].detach().cpu().numpy() * std + mean) * 255.,
                   0, 255).round().astype(np.uint8)


def bce_pos_weight_from_approved_train_labels(labels):
    """Calculate BCE pos_weight ONLY after a leakage-reviewed training split exists.

    `labels` must represent the *approved training subset* and use NORMAL=0,
    PNEUMONIA=1. Do not pass original train/test combined labels.
    """
    values = np.asarray(list(labels))
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expect binary labels encoded as 0/1.")
    n_negative = int((values == 0).sum())
    n_positive = int((values == 1).sum())
    if n_negative == 0 or n_positive == 0:
        raise ValueError("Both training classes are required for pos_weight.")
    return torch.tensor(float(n_negative / n_positive), dtype=torch.float32)
