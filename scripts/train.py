#!/usr/bin/env python3
"""Train one configured hand gesture classification experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.config import get_nested, load_config
from hand_gesture_rocm.constants import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from hand_gesture_rocm.metrics import classification_summary, empty_confusion_matrix, update_confusion_matrix


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_precision_macro",
    "train_recall_macro",
    "train_f1_macro",
    "val_loss",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "epoch_seconds",
    "is_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured PyTorch gesture classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}") from exc

    from hand_gesture_rocm.data import build_image_folder
    from hand_gesture_rocm.models import build_model, count_parameters

    seed = int(get_nested(config, "experiment.seed", 42))
    set_seed(seed, torch, np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = str(get_nested(config, "experiment.name", "experiment"))
    run_dir = Path(get_nested(config, "outputs.run_dir", f"runs/{experiment_name}"))
    checkpoint_dir = Path(get_nested(config, "outputs.checkpoint_dir", f"checkpoints/{experiment_name}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dir = get_nested(config, "data.train_dir")
    val_dir = get_nested(config, "data.val_dir")
    if not train_dir or not val_dir:
        raise ValueError("Config must define data.train_dir and data.val_dir.")

    batch_size = int(get_nested(config, "data.batch_size", 32))
    num_workers = int(get_nested(config, "data.num_workers", 4))
    grayscale_to_rgb = bool(get_nested(config, "data.grayscale_to_rgb", True))

    train_dataset = build_image_folder(train_dir, train=True, grayscale_to_rgb=grayscale_to_rgb)
    val_dataset = build_image_folder(val_dir, train=False, grayscale_to_rgb=grayscale_to_rgb)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    architecture = str(get_nested(config, "model.architecture", "resnet18"))
    pretrained = bool(get_nested(config, "model.pretrained", True))
    training_mode = str(get_nested(config, "model.training_mode", "classifier_only"))
    model = build_model(
        architecture=architecture,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        training_mode=training_mode,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    epochs = int(get_nested(config, "training.epochs", 30))
    early_config = get_nested(config, "training.early_stopping", {}) or {}
    monitor_name = str(early_config.get("monitor", "val_loss"))
    monitor_mode = str(early_config.get("mode", infer_monitor_mode(monitor_name))).lower()
    patience = int(early_config.get("patience", epochs))
    min_delta = float(early_config.get("min_delta", 0.0))
    scheduler = build_scheduler(config, optimizer, epochs, monitor_mode)

    history_path = run_dir / "training_history.csv"
    metrics_path = run_dir / "metrics.json"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    last_checkpoint_path = checkpoint_dir / "last_model.pth"

    best_score: float | None = None
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Model: {architecture} ({training_mode})")
    print(f"Parameters: {count_parameters(model):,} total / {count_parameters(model, trainable_only=True):,} trainable")
    print(f"Outputs: {run_dir} and {checkpoint_dir}")

    stopped_early = False
    final_epoch = 0
    final_validation: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_learning_rate = current_learning_rate(optimizer)
        train_stats = run_epoch(model, train_loader, criterion, device, torch, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, torch, optimizer=None)
        final_epoch = epoch
        final_validation = val_stats

        monitor_value = resolve_monitor_value(monitor_name, val_stats)
        is_best = best_score is None or is_improvement(monitor_value, best_score, monitor_mode, min_delta)
        if is_best:
            best_score = monitor_value
            best_epoch = epoch
            best_validation = val_stats
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start
        row = build_history_row(epoch, epoch_learning_rate, train_stats, val_stats, epoch_seconds, is_best)
        history.append(row)
        write_history_csv(history_path, history)
        scheduler_step(scheduler, monitor_value)

        checkpoint_payload = build_checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            architecture=architecture,
            training_mode=training_mode,
            pretrained=pretrained,
            validation_stats=val_stats,
            best_epoch=best_epoch,
            best_score=best_score,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
        )
        torch.save(checkpoint_payload, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint_payload, best_checkpoint_path)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1_macro']:.4f} "
            f"lr={row['learning_rate']:.6g}"
        )

        if epochs_without_improvement >= patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch} on {monitor_name}.")
            break

    metrics = build_metrics_json(
        config=config,
        experiment_name=experiment_name,
        architecture=architecture,
        training_mode=training_mode,
        pretrained=pretrained,
        model=model,
        best_epoch=best_epoch,
        best_score=best_score,
        best_validation=best_validation,
        final_epoch=final_epoch,
        final_validation=final_validation,
        monitor_name=monitor_name,
        monitor_mode=monitor_mode,
        stopped_early=stopped_early,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )
    write_json(metrics_path, metrics)
    print(f"Saved history: {history_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved checkpoints: {best_checkpoint_path}, {last_checkpoint_path}")
    return 0


def set_seed(seed: int, torch_module, numpy_module) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_optimizer(config: dict[str, Any], model):
    import torch

    optimizer_name = str(get_nested(config, "training.optimizer", "AdamW")).lower()
    learning_rate = float(get_nested(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_nested(config, "training.weight_decay", 1e-4))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters found for this model/training mode.")

    if optimizer_name != "adamw":
        raise ValueError("Only AdamW is implemented in Phase 2.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(config: dict[str, Any], optimizer, epochs: int, monitor_mode: str):
    import torch

    scheduler_name = str(get_nested(config, "training.scheduler", "ReduceLROnPlateau"))
    normalized = scheduler_name.lower().replace("_", "").replace("-", "")
    if normalized in {"none", "null"}:
        return None
    if normalized in {"reducelronplateau", "reduceonplateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=monitor_mode, factor=0.1, patience=3)
    if normalized in {"cosineannealinglr", "cosine"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_epoch(model, loader, criterion, device, torch_module, optimizer=None) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    matrix = empty_confusion_matrix(NUM_CLASSES)
    context = torch_module.enable_grad() if is_training else torch_module.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            predictions = outputs.argmax(dim=1)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            update_confusion_matrix(
                matrix,
                targets.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            )

    summary = classification_summary(matrix)
    summary["loss"] = total_loss / max(total_samples, 1)
    return summary


def build_history_row(
    epoch: int,
    learning_rate: float,
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    epoch_seconds: float,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": float(learning_rate),
        "train_loss": float(train_stats["loss"]),
        "train_accuracy": float(train_stats["accuracy"]),
        "train_precision_macro": float(train_stats["precision_macro"]),
        "train_recall_macro": float(train_stats["recall_macro"]),
        "train_f1_macro": float(train_stats["f1_macro"]),
        "val_loss": float(val_stats["loss"]),
        "val_accuracy": float(val_stats["accuracy"]),
        "val_precision_macro": float(val_stats["precision_macro"]),
        "val_recall_macro": float(val_stats["recall_macro"]),
        "val_f1_macro": float(val_stats["f1_macro"]),
        "epoch_seconds": float(epoch_seconds),
        "is_best": bool(is_best),
    }


def write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_checkpoint_payload(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    epoch: int,
    architecture: str,
    training_mode: str,
    pretrained: bool,
    validation_stats: dict[str, Any],
    best_epoch: int,
    best_score: float | None,
    monitor_name: str,
    monitor_mode: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "architecture": architecture,
        "training_mode": training_mode,
        "pretrained": pretrained,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "validation_metrics": validation_stats,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "monitor_name": monitor_name,
        "monitor_mode": monitor_mode,
        "config": config,
    }


def build_metrics_json(
    config: dict[str, Any],
    experiment_name: str,
    architecture: str,
    training_mode: str,
    pretrained: bool,
    model,
    best_epoch: int,
    best_score: float | None,
    best_validation: dict[str, Any] | None,
    final_epoch: int,
    final_validation: dict[str, Any] | None,
    monitor_name: str,
    monitor_mode: str,
    stopped_early: bool,
    run_dir: Path,
    checkpoint_dir: Path,
    history_path: Path,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
) -> dict[str, Any]:
    from hand_gesture_rocm.models import count_parameters

    return {
        "experiment": experiment_name,
        "model": {
            "architecture": architecture,
            "training_mode": training_mode,
            "pretrained": pretrained,
            "num_classes": NUM_CLASSES,
            "parameters": count_parameters(model),
            "trainable_parameters": count_parameters(model, trainable_only=True),
        },
        "data": {
            "class_names": list(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "train_dir": get_nested(config, "data.train_dir"),
            "val_dir": get_nested(config, "data.val_dir"),
            "batch_size": get_nested(config, "data.batch_size"),
        },
        "early_stopping": {
            "monitor": monitor_name,
            "mode": monitor_mode,
            "stopped_early": stopped_early,
        },
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation_metrics": best_validation,
        "final_epoch": final_epoch,
        "final_validation_metrics": final_validation,
        "outputs": {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "history_path": str(history_path),
            "metrics_path": str(run_dir / "metrics.json"),
            "best_checkpoint_path": str(best_checkpoint_path),
            "last_checkpoint_path": str(last_checkpoint_path),
        },
    }


def scheduler_step(scheduler, monitor_value: float) -> None:
    if scheduler is None:
        return
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(monitor_value)
    else:
        scheduler.step()


def current_learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def resolve_monitor_value(monitor_name: str, val_stats: dict[str, Any]) -> float:
    normalized = monitor_name.lower().replace("-", "_")
    mapping = {
        "val_loss": "loss",
        "validation_loss": "loss",
        "val_accuracy": "accuracy",
        "validation_accuracy": "accuracy",
        "val_precision_macro": "precision_macro",
        "val_recall_macro": "recall_macro",
        "val_f1": "f1_macro",
        "val_f1_macro": "f1_macro",
        "validation_f1": "f1_macro",
        "validation_f1_macro": "f1_macro",
    }
    key = mapping.get(normalized)
    if key is None:
        raise ValueError(f"Unsupported early stopping monitor: {monitor_name}")
    return float(val_stats[key])


def infer_monitor_mode(monitor_name: str) -> str:
    return "min" if "loss" in monitor_name.lower() else "max"


def is_improvement(value: float, best_value: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return value < best_value - min_delta
    if mode == "max":
        return value > best_value + min_delta
    raise ValueError(f"Unsupported monitor mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
