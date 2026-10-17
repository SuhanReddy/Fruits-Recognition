"""Data loading utilities for the Fruits-360 dataset.

Expected folder layout (one sub-folder per fruit class):

    <data_dir>/
        Training/
            Apple Braeburn/ *.jpg
            Banana/         *.jpg
            ...
        Test/
            Apple Braeburn/ *.jpg
            ...
"""
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int, train: bool):
    """Training gets light augmentation; validation/test/inference does not."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _get_dir(data_dir, name: str) -> Path:
    path = Path(data_dir) / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Could not find '{path}'. Point --data_dir to the folder that "
            f"directly contains the 'Training' and 'Test' folders."
        )
    return path


def _align_test_labels(test_set, class_names):
    train_class_to_idx = {name: index for index, name in enumerate(class_names)}
    target_map = {
        test_index: train_class_to_idx[class_name]
        for test_index, class_name in enumerate(test_set.classes)
        if class_name in train_class_to_idx
    }
    test_set.target_transform = target_map.__getitem__
    valid_indices = [
        index for index, target in enumerate(test_set.targets)
        if target in target_map
    ]
    if not valid_indices:
        raise ValueError("Test/ has no images for classes present in Training/.")

    unknown_classes = set(test_set.classes) - train_class_to_idx.keys()
    if unknown_classes:
        skipped_count = len(test_set.targets) - len(valid_indices)
        warnings.warn(
            f"Skipping {skipped_count} Test/ image(s) from classes absent in Training/: "
            f"{sorted(unknown_classes)}",
            UserWarning,
        )

    return Subset(test_set, valid_indices)


def build_dataloaders(data_dir, img_size=100, batch_size=64, val_split=0.1,
                      num_workers=2, seed=42):
    """Return train/val/test loaders and the list of class names."""
    train_dir = _get_dir(data_dir, "Training")
    test_dir = _get_dir(data_dir, "Test")

    # Two ImageFolder copies so the validation subset does NOT get augmentation.
    train_full = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, True))
    val_full = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, False))
    test_set = datasets.ImageFolder(test_dir, transform=get_transforms(img_size, False))
    test_set = _align_test_labels(test_set, train_full.classes)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    n_val = int(len(perm) * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, train_full.classes


def build_test_loader(data_dir, img_size=100, batch_size=64, num_workers=2,
                      class_names=None):
    """Return only the test loader and class names (used by evaluate.py)."""
    test_dir = _get_dir(data_dir, "Test")
    test_set = datasets.ImageFolder(test_dir, transform=get_transforms(img_size, False))
    test_classes = test_set.classes
    if class_names is not None:
        test_set = _align_test_labels(test_set, class_names)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, test_classes
