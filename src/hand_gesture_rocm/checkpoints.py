"""Checkpoint loading helpers shared by evaluation, export and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import CLASS_NAMES, NUM_CLASSES


def load_checkpoint(path: str | Path, map_location: str | object = "cpu") -> dict[str, Any]:
    """Load a training checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load checkpoints.") from exc

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | object = "cpu",
    architecture: str | None = None,
    training_mode: str | None = None,
) -> tuple[object, dict[str, Any]]:
    """Rebuild a model and load weights from a checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    architecture = architecture or checkpoint.get("architecture")
    training_mode = training_mode or checkpoint.get("training_mode", "full_finetuning")
    pretrained = bool(checkpoint.get("pretrained", False))
    num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))

    if not architecture:
        raise ValueError("Checkpoint does not include architecture. Pass --architecture.")

    from .models import build_model

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False if "model_state_dict" in checkpoint else pretrained,
        training_mode=training_mode,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def checkpoint_class_names(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    """Return checkpoint class names and validate the expected thesis order."""
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order differs from expected order: {class_names}")
    return class_names
