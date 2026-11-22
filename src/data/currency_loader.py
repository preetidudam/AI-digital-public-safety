"""
Data loader and robust preprocessing transforms for Indian Currency Verification.
Implements:
1. AspectPreservingPadResize: Eliminates aspect ratio distortion shortcuts.
2. Random90Rotation: Prevents orientation/portrait vs landscape cheats.
3. Color & background regularization: Guards against dataset-specific texture/watermark artifacts.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CURRENCY_DATASET_DIR, CURRENCY_IMAGE_SIZE, CURRENCY_CLASSES

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AspectPreservingPadResize:
    """
    Resizes image maintaining original aspect ratio and pads to target_size
    with neutral grey background (128, 128, 128).
    Eliminates aspect-ratio squashing artifacts between vertical & horizontal notes.
    """
    def __init__(self, target_size: Tuple[int, int] = CURRENCY_IMAGE_SIZE):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded_img = Image.new("RGB", self.target_size, (128, 128, 128))
        padded_img.paste(resized_img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return padded_img


class Random90Rotation:
    """
    Randomly rotates image by 0, 90, 180, or 270 degrees.
    Ensures absolute orientation invariance across portrait and landscape captures.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        return img.rotate(angle, expand=True)


def get_currency_transforms(image_size: Tuple[int, int] = CURRENCY_IMAGE_SIZE):
    """
    Returns robust training and validation/inference image transformations.
    Guarantees geometry preservation and breaks background/orientation artifacts.
    """
    train_transform = transforms.Compose([
        Random90Rotation(),
        AspectPreservingPadResize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.06),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random")
    ])

    val_transform = transforms.Compose([
        AspectPreservingPadResize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def check_dataset_availability(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Checks if currency dataset exists and counts images in train/val/external_test splits.
    """
    if data_dir is None:
        data_dir = Path(CURRENCY_DATASET_DIR)
    else:
        data_dir = Path(data_dir)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    stats = {
        "dataset_dir": str(data_dir),
        "exists": data_dir.exists(),
        "train_genuine": 0,
        "train_counterfeit": 0,
        "val_genuine": 0,
        "val_counterfeit": 0,
        "ext_genuine": 0,
        "ext_counterfeit": 0,
        "total_images": 0,
        "is_ready": False
    }

    if not data_dir.exists():
        return stats

    for split in ["train", "val"]:
        for cls_name in ["genuine", "counterfeit"]:
            folder = data_dir / split / cls_name
            if folder.exists():
                count = sum(1 for f in folder.iterdir() if f.suffix.lower() in valid_exts)
                stats[f"{split}_{cls_name}"] = count
                stats["total_images"] += count

    for cls_name in ["genuine", "counterfeit"]:
        folder = data_dir / "external_test" / cls_name
        if folder.exists():
            count = sum(1 for f in folder.iterdir() if f.suffix.lower() in valid_exts)
            stats[f"ext_{cls_name}"] = count

    stats["is_ready"] = (
        (stats["train_genuine"] + stats["train_counterfeit"]) > 0
    )
    return stats


def get_currency_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Builds PyTorch DataLoaders for train and validation splits.
    """
    if data_dir is None:
        data_dir = Path(CURRENCY_DATASET_DIR)
    else:
        data_dir = Path(data_dir)

    stats = check_dataset_availability(data_dir)
    if not stats["is_ready"]:
        raise FileNotFoundError(
            f"Currency dataset not found or empty in '{data_dir}'."
        )

    train_transform, val_transform = get_currency_transforms()

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)

    if (stats["val_genuine"] + stats["val_counterfeit"]) > 0:
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    else:
        val_dataset = datasets.ImageFolder(str(train_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, stats
