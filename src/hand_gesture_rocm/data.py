"""Dataset and transform helpers for ImageFolder-style gesture datasets."""

from __future__ import annotations

from pathlib import Path

from .constants import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(train: bool = False, grayscale_to_rgb: bool = True):
    """Build 224x224 ImageNet-normalized transforms.

    Augmentation is intentionally not enabled in Phase 1. It can be added later
    as an explicit experiment option.
    """
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build image transforms.") from exc

    steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if grayscale_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_image_folder(data_dir: str | Path, train: bool = False, grayscale_to_rgb: bool = True):
    """Create an ImageFolder dataset and validate the fixed class order."""
    try:
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load ImageFolder datasets.") from exc

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    dataset = ImageFolder(path, transform=build_transforms(train=train, grayscale_to_rgb=grayscale_to_rgb))
    validate_class_order(dataset.classes)
    return dataset


def validate_class_order(classes: list[str] | tuple[str, ...]) -> None:
    """Fail fast if ImageFolder classes differ from the thesis class order."""
    observed = tuple(classes)
    expected = CLASS_NAMES
    if observed != expected:
        raise ValueError(
            "Invalid class order. "
            f"Expected {list(expected)}, got {list(observed)}. "
            "Use one folder per class with the fixed class names."
        )
