"""Dataset and reproducible DataLoaders for already validated split frames."""
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data import validate_splits, DEFAULT_DATA_ROOT, DEFAULT_SPLIT_ROOT
from .transforms import build_transform


class HAM10000Dataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = build_transform(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor(int(row.target), dtype=torch.long)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def create_dataloaders(data_root=DEFAULT_DATA_ROOT, split_root=DEFAULT_SPLIT_ROOT,
                       batch_size=16, num_workers=2, seed=42):
    """Validate all splits first; shuffle and augment training only.

    No resampling, split generation, or dropped final batch. Use num_workers=0
    to debug notebook worker errors. Same seed reproduces a fresh loader run.
    """
    frames, report = validate_splits(data_root, split_root)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loaders = {}
    for name, frame in frames.items():
        loaders[name] = DataLoader(
            HAM10000Dataset(frame, training=name == "train"),
            batch_size=batch_size, shuffle=name == "train", num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    return loaders, report
