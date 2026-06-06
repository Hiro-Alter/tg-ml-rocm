"""Model factory for the two architectures used in the experiments."""

from __future__ import annotations

import torch.nn as nn

from .constants import NUM_CLASSES, SUPPORTED_MODELS, TRAINING_MODES


def build_model(
    architecture: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    training_mode: str = "classifier_only",
) -> nn.Module:
    """Build a torchvision model with a 10-class classifier head."""
    if architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported architecture: {architecture}. Options: {SUPPORTED_MODELS}")
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unsupported training mode: {training_mode}. Options: {TRAINING_MODES}")

    try:
        from torchvision import models as tv_models
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build models.") from exc

    if architecture == "resnet18":
        model = _build_resnet18(tv_models, pretrained)
        if training_mode == "classifier_only":
            _freeze_all(model)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = _build_mobilenetv3_small(tv_models, pretrained)
        if training_mode == "classifier_only":
            _freeze_all(model)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    if training_mode == "full_finetuning":
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def _build_resnet18(tv_models, pretrained: bool) -> nn.Module:
    if hasattr(tv_models, "ResNet18_Weights"):
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        return tv_models.resnet18(weights=weights)
    return tv_models.resnet18(pretrained=pretrained)


def _build_mobilenetv3_small(tv_models, pretrained: bool) -> nn.Module:
    if hasattr(tv_models, "MobileNet_V3_Small_Weights"):
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        return tv_models.mobilenet_v3_small(weights=weights)
    return tv_models.mobilenet_v3_small(pretrained=pretrained)


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
