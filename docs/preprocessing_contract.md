# PneumoLens — preprocessing contract

## Installation and usage

Use `src/preprocessing/transforms.py` and `configs/preprocessing.json` from this Phase 3 ZIP
in your GitHub repository. Kaggle training notebooks should import exactly these files
from the repository source, not copy/paste alternative transformations.

```python
from src.preprocessing.transforms import load_config, PneumoPreprocessor
config = load_config('configs/preprocessing.json')
train_transform = PneumoPreprocessor(config, mode='train')  # ONLY approved training subset
inference_transform = PneumoPreprocessor(config, mode='eval')  # validation/test/Streamlit
input_tensor = inference_transform('example_xray.jpg')   # shape [3,224,224]
# For a single image in PyTorch: input_tensor.unsqueeze(0)
```

Do not use `train_transform` on any validation/test or app image. For model training
with multiple DataLoader workers, seed Python's `random` in each worker as well as
the torch worker seeds; always record seed and versions for reproducible experiments.

The pretrained backbones use fixed ImageNet-1K mean/std after luminance conversion
and three-channel replication. Do not replace with Phase 2 raw thumbnail statistics.
Original grayscale should remain available separately for human viewing and Grad-CAM.

## Test locally

```bash
python -m unittest discover -s tests -p "test_preprocessing.py" -v
```

For complete numerical parity, compare output tensors with the included synthetic
reference (`tests/fixtures/reference_eval_tensor.npy`). Pillow version differences
may affect interpolation at the sub-pixel level; record the same Pillow version in
training, testing and inference and explicitly rerun regression tests after upgrades.

## Data integrity limitations

- No validation split has been created or approved.
- Original test images were not used to fit preprocessing decisions.
- Exact duplicate review, cross-split visual similarity and patient-level independence
  remain open from the Phase 1 audit.
- Raw size, color mode and padding correlate with labels; evaluate shortcut behavior.
- Do not commit patient image galleries, private file-level profiles or the dataset.
- Research and educational prototype only; no diagnostic or clinical use.
