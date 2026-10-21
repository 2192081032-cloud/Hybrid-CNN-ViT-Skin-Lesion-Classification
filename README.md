# Hybrid-CNN-ViT-Skin-Lesion-Classification

Beginner-friendly PyTorch preprocessing for seven-class HAM10000 research. No model training is included.

## Files

- `ham10000/data.py`: read existing CSVs, resolve images, validate labels and lesion separation.
- `ham10000/transforms.py`: 224×224 resizing, training-only augmentation, ImageNet normalization.
- `ham10000/dataset.py`: custom Dataset and DataLoaders (default batch size 16).
- `notebooks/ham10000_preprocessing.ipynb`: Kaggle walkthrough and batch checks.
- `tests/test_preprocessing.py`: synthetic-data tests; no dataset images are stored in Git.

## Run in Kaggle

1. Attach `kmader/skin-cancer-mnist-ham10000`. The default data directory is `/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000`; update `DATA_ROOT` if Kaggle mounts it elsewhere.
2. Keep your existing `train.csv`, `val.csv`, and `test.csv` in `/kaggle/working/splits`. This pipeline never rewrites or regenerates them. Kaggle working files must be restored if you start a new session.
3. Upload/open the notebook and copy or attach this repository. Set its `PROJECT_ROOT` to the folder containing `ham10000/`. An attached repository can remain read-only.
4. Run all cells. Kaggle usually includes PyTorch, torchvision, pandas, NumPy, and Pillow. If imports are missing, enable Internet and install `requirements.txt` using `%pip install -r /path/to/requirements.txt`. Keep PyTorch and torchvision compatible; avoid replacing Kaggle's working GPU packages unnecessarily.
5. Read the class counts and confirm each split produces a batch. No GPU, credentials, or model weights are required. Use `num_workers=0` for worker troubleshooting.

### CSV contract and checks

Every CSV needs `image_id` (for example `ISIC_0024306`, without a file extension). Supply `dx` (or `label`) and `lesion_id`, or let the pipeline join these from the dataset's `HAM10000_metadata.csv`. If both metadata and CSV values exist, they must agree. Labels must be exactly:

| Index | Class |
| --- | --- |
| 0 | akiec |
| 1 | bcc |
| 2 | bkl |
| 3 | df |
| 4 | mel |
| 5 | nv |
| 6 | vasc |

Images are searched recursively, covering the dataset's two image directories. Optional `image_path` values must match the image ID and can be absolute or relative to `DATA_ROOT`. Duplicate/ambiguous IDs, missing or unreadable images, invalid labels, inconsistent lesion diagnoses, and image/lesion overlap across any two splits raise errors. Multiple images of a lesion within one split are allowed. Validation preserves row order and split membership. Counts include zero for absent classes; no balancing or resampling is performed. Leakage checks use IDs and resolved paths, not visual duplicate detection.

All images are converted to RGB and resized to 224×224. Training adds horizontal/vertical flips, ±20° rotation, and mild brightness/contrast/saturation jitter. Validation and test have no random augmentation. Tensors use ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Only training is shuffled; incomplete final batches are retained. Augmentation choices should be reviewed for your research protocol.

## Use from Python

Run from the repository root:

```python
from ham10000.dataset import create_dataloaders

loaders, report = create_dataloaders(num_workers=2)  # validates all three splits
images, targets = next(iter(loaders['train']))
print(images.shape, targets.shape)  # [N, 3, 224, 224], [N]; N <= 16
```

## Local development and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
# CPU-only alternative; omit this line when using a working CUDA installation.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests generate temporary images and CSVs, check normalization and loader behavior, and inject invalid inputs to exercise validation failures. Real Kaggle data is not included; run the notebook there to validate your actual splits. Never add dataset images or research split files to Git.
