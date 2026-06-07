#!/usr/bin/env python3
"""Train one configured hand gesture classification experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
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
    "train_seconds",
    "val_seconds",
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
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}") from exc

    from hand_gesture_rocm.data import build_image_folder
    from hand_gesture_rocm.models import apply_training_mode, build_model, count_parameters

    seed = int(get_nested(config, "experiment.seed", 42))
    deterministic = bool(get_nested(config, "experiment.deterministic", True))
    set_seed(seed, torch, np, deterministic=deterministic)

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
    pin_memory = bool(get_nested(config, "data.pin_memory", device.type == "cuda"))
    persistent_workers = bool(get_nested(config, "data.persistent_workers", num_workers > 0))
    prefetch_factor = get_nested(config, "data.prefetch_factor", 2 if num_workers > 0 else None)
    grayscale_to_rgb = bool(get_nested(config, "data.grayscale_to_rgb", True))

    train_dataset = build_image_folder(train_dir, train=True, grayscale_to_rgb=grayscale_to_rgb)
    val_dataset = build_image_folder(val_dir, train=False, grayscale_to_rgb=grayscale_to_rgb)

    default_limit = get_nested(config, "data.max_samples_per_class")
    train_limit = get_nested(config, "data.max_train_samples_per_class", default_limit)
    val_limit = get_nested(config, "data.max_val_samples_per_class", default_limit)
    train_dataset = limit_samples_per_class(train_dataset, train_limit, seed, Subset)
    val_dataset = limit_samples_per_class(val_dataset, val_limit, seed + 1, Subset)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    architecture = str(get_nested(config, "model.architecture", "resnet18"))
    pretrained = bool(get_nested(config, "model.pretrained", True))
    training_mode = str(get_nested(config, "model.training_mode", "classifier_only"))
    training_stages = build_training_stages(config, training_mode)
    initial_training_mode = str(training_stages[0]["training_mode"])
    model = build_model(
        architecture=architecture,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        training_mode=initial_training_mode,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    history_path = run_dir / "training_history.csv"
    metrics_path = run_dir / "metrics.json"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    last_checkpoint_path = checkpoint_dir / "last_model.pth"

    best_score: float | None = None
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Model: {architecture} ({initial_training_mode})")
    print(f"Protocol: {format_training_protocol(training_stages)}")
    print(f"Samples: {len(train_dataset):,} train / {len(val_dataset):,} val")
    print(
        "DataLoader: "
        f"batch_size={batch_size} num_workers={num_workers} pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers if num_workers > 0 else False} "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else None}"
    )
    print(
        "Torch performance: "
        f"deterministic={deterministic} "
        f"cudnn_benchmark={getattr(torch.backends.cudnn, 'benchmark', None)} "
        f"cudnn_deterministic={getattr(torch.backends.cudnn, 'deterministic', None)}"
    )
    print(f"Parameters: {count_parameters(model):,} total / {count_parameters(model, trainable_only=True):,} trainable")
    print(f"Outputs: {run_dir} and {checkpoint_dir}")

    stopped_early = False
    stopped_stages: list[str] = []
    final_epoch = 0
    final_validation: dict[str, Any] | None = None
    final_training_mode = initial_training_mode
    final_monitor_name = "val_loss"
    final_monitor_mode = "min"

    for stage_index, stage in enumerate(training_stages, start=1):
        stage_name = str(stage["name"])
        stage_training_mode = str(stage["training_mode"])
        stage_config = config_for_stage(config, stage)
        stage_epochs = int(stage["epochs"])
        apply_training_mode(model, architecture, stage_training_mode)
        optimizer = build_optimizer(stage_config, model)
        amp_config = build_amp_config(stage_config, device, torch)
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=amp_config["enabled"] and amp_config["grad_scaler"],
        )
        early_config = get_nested(stage_config, "training.early_stopping", {}) or {}
        monitor_name = str(early_config.get("monitor", "val_loss"))
        monitor_mode = str(early_config.get("mode", infer_monitor_mode(monitor_name))).lower()
        patience = int(early_config.get("patience", stage_epochs))
        min_delta = float(early_config.get("min_delta", 0.0))
        scheduler = build_scheduler(stage_config, optimizer, stage_epochs, monitor_mode)
        stage_best_score: float | None = None
        stage_epochs_without_improvement = 0
        final_training_mode = stage_training_mode
        final_monitor_name = monitor_name
        final_monitor_mode = monitor_mode

        print(
            f"Stage {stage_index}/{len(training_stages)}: {stage_name} "
            f"mode={stage_training_mode} epochs={stage_epochs} "
            f"lr={current_learning_rate(optimizer):.6g} "
            f"trainable={count_parameters(model, trainable_only=True):,}"
        )
        print(
            "Mixed precision: "
            f"enabled={amp_config['enabled']} dtype={amp_config['dtype_name']} "
            f"grad_scaler={amp_config['grad_scaler'] and scaler.is_enabled()}"
        )

        for stage_epoch in range(1, stage_epochs + 1):
            final_epoch += 1
            epoch_start = time.perf_counter()
            epoch_learning_rate = current_learning_rate(optimizer)
            train_start = time.perf_counter()
            train_stats = run_epoch(model, train_loader, criterion, device, torch, optimizer, amp_config, scaler)
            train_seconds = time.perf_counter() - train_start
            val_start = time.perf_counter()
            val_stats = run_epoch(model, val_loader, criterion, device, torch, optimizer=None, amp_config=amp_config)
            val_seconds = time.perf_counter() - val_start
            final_validation = val_stats

            monitor_value = resolve_monitor_value(monitor_name, val_stats)
            is_stage_improvement = stage_best_score is None or is_improvement(
                monitor_value,
                stage_best_score,
                monitor_mode,
                min_delta,
            )
            if is_stage_improvement:
                stage_best_score = monitor_value
                stage_epochs_without_improvement = 0
            else:
                stage_epochs_without_improvement += 1

            is_best_eligible = len(training_stages) == 1 or stage_index == len(training_stages)
            is_best = is_best_eligible and (
                best_score is None or is_improvement(monitor_value, best_score, monitor_mode, min_delta)
            )
            if is_best:
                best_score = monitor_value
                best_epoch = final_epoch
                best_validation = val_stats

            epoch_seconds = time.perf_counter() - epoch_start
            row = build_history_row(
                final_epoch,
                epoch_learning_rate,
                train_stats,
                val_stats,
                train_seconds,
                val_seconds,
                epoch_seconds,
                is_best,
            )
            history.append(row)
            write_history_csv(history_path, history)
            scheduler_step(scheduler, monitor_value)

            checkpoint_payload = build_checkpoint_payload(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=final_epoch,
                architecture=architecture,
                training_mode=stage_training_mode,
                pretrained=pretrained,
                validation_stats=val_stats,
                best_epoch=best_epoch,
                best_score=best_score,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                training_protocol=training_stages,
            )
            torch.save(checkpoint_payload, last_checkpoint_path)
            if is_best:
                torch.save(checkpoint_payload, best_checkpoint_path)

            print(
                f"Epoch {final_epoch:03d} "
                f"stage={stage_name} {stage_epoch:03d}/{stage_epochs:03d} "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1_macro']:.4f} "
                f"lr={row['learning_rate']:.6g} "
                f"time={row['epoch_seconds']:.2f}s train={row['train_seconds']:.2f}s val={row['val_seconds']:.2f}s"
            )

            if stage_epochs_without_improvement >= patience:
                stopped_early = True
                stopped_stages.append(stage_name)
                print(f"Early stopping stage {stage_name} at epoch {stage_epoch} on {monitor_name}.")
                break

    metrics = build_metrics_json(
        config=config,
        experiment_name=experiment_name,
        architecture=architecture,
        training_mode=final_training_mode,
        pretrained=pretrained,
        model=model,
        best_epoch=best_epoch,
        best_score=best_score,
        best_validation=best_validation,
        final_epoch=final_epoch,
        final_validation=final_validation,
        monitor_name=final_monitor_name,
        monitor_mode=final_monitor_mode,
        stopped_early=stopped_early,
        stopped_stages=stopped_stages,
        training_protocol=training_stages,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
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


def set_seed(seed: int, torch_module, numpy_module, deterministic: bool = True) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = not deterministic
        torch_module.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def limit_samples_per_class(dataset, max_samples_per_class: Any, seed: int, subset_cls):
    if max_samples_per_class is None:
        return dataset

    limit = int(max_samples_per_class)
    if limit <= 0:
        raise ValueError("data.max_samples_per_class must be greater than zero.")

    indices_by_class = {class_index: [] for class_index in range(NUM_CLASSES)}
    for index, target in enumerate(dataset.targets):
        indices_by_class[target].append(index)

    rng = random.Random(seed)
    indices: list[int] = []
    for class_index in range(NUM_CLASSES):
        class_indices = indices_by_class[class_index]
        if len(class_indices) < limit:
            raise ValueError(
                f"Class {CLASS_NAMES[class_index]} has {len(class_indices)} samples, "
                f"cannot select {limit}."
            )
        rng.shuffle(class_indices)
        indices.extend(class_indices[:limit])
    rng.shuffle(indices)
    return subset_cls(dataset, indices)


def build_training_stages(config: dict[str, Any], default_training_mode: str) -> list[dict[str, Any]]:
    raw_stages = get_nested(config, "training.stages")
    base_training = copy.deepcopy(get_nested(config, "training", {}) or {})
    base_training.pop("stages", None)

    if not raw_stages:
        stage = copy.deepcopy(base_training)
        stage.setdefault("name", default_training_mode)
        stage.setdefault("training_mode", default_training_mode)
        stage.setdefault("epochs", get_nested(config, "training.epochs", 30))
        return [normalize_training_stage(stage, base_training, 1)]

    if not isinstance(raw_stages, list):
        raise ValueError("training.stages must be a list when defined.")

    stages: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_stages, start=1):
        if not isinstance(raw_stage, dict):
            raise ValueError("Each training stage must be a mapping.")
        stage = copy.deepcopy(base_training)
        stage.update(copy.deepcopy(raw_stage))
        stages.append(normalize_training_stage(stage, base_training, index))
    return stages


def normalize_training_stage(stage: dict[str, Any], base_training: dict[str, Any], index: int) -> dict[str, Any]:
    training_mode = str(stage.get("training_mode", stage.get("name", "classifier_only")))
    stage.setdefault("name", training_mode)
    stage["training_mode"] = training_mode
    stage["epochs"] = int(stage.get("epochs", base_training.get("epochs", 30)))
    if stage["epochs"] <= 0:
        raise ValueError(f"Stage {index} must define epochs > 0.")

    if "learning_rate_multiplier" in stage:
        base_learning_rate = float(stage.get("learning_rate", base_training.get("learning_rate", 1e-3)))
        stage["learning_rate"] = base_learning_rate * float(stage["learning_rate_multiplier"])
    else:
        stage.setdefault("learning_rate", base_training.get("learning_rate", 1e-3))

    stage.setdefault("optimizer", base_training.get("optimizer", "AdamW"))
    stage.setdefault("weight_decay", base_training.get("weight_decay", 1e-4))
    stage.setdefault("scheduler", base_training.get("scheduler", "ReduceLROnPlateau"))
    stage.setdefault("early_stopping", copy.deepcopy(base_training.get("early_stopping", {})))
    stage.setdefault("mixed_precision", copy.deepcopy(base_training.get("mixed_precision", False)))
    return stage


def config_for_stage(config: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    stage_config = copy.deepcopy(config)
    stage_training = copy.deepcopy(stage)
    stage_training.pop("name", None)
    stage_training.pop("training_mode", None)
    stage_config["training"] = stage_training
    return stage_config


def format_training_protocol(stages: list[dict[str, Any]]) -> str:
    return " -> ".join(
        f"{stage['name']}[{stage['training_mode']}, {stage['epochs']}e, lr={float(stage['learning_rate']):.6g}]"
        for stage in stages
    )


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


def build_amp_config(config: dict[str, Any], device, torch_module) -> dict[str, Any]:
    raw_config = get_nested(config, "training.mixed_precision", False)
    if isinstance(raw_config, dict):
        requested = bool(raw_config.get("enabled", False))
        dtype_name = str(raw_config.get("dtype", "float16")).lower()
        grad_scaler = bool(raw_config.get("grad_scaler", dtype_name in {"float16", "fp16", "half"}))
    else:
        requested = bool(raw_config)
        dtype_name = "float16"
        grad_scaler = True

    dtype_mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    if dtype_name not in dtype_mapping:
        raise ValueError(f"Unsupported mixed precision dtype: {dtype_name}")

    available = device.type == "cuda" and torch_module.amp.autocast_mode.is_autocast_available(device.type)
    enabled = requested and available
    return {
        "requested": requested,
        "enabled": enabled,
        "available": available,
        "dtype": dtype_mapping[dtype_name],
        "dtype_name": dtype_name,
        "grad_scaler": grad_scaler and dtype_mapping[dtype_name] == torch_module.float16,
    }


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


def run_epoch(
    model,
    loader,
    criterion,
    device,
    torch_module,
    optimizer=None,
    amp_config: dict[str, Any] | None = None,
    scaler=None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    amp_config = amp_config or {"enabled": False, "dtype": None}

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

            autocast_context = (
                torch_module.amp.autocast(device_type=device.type, dtype=amp_config["dtype"], enabled=True)
                if amp_config.get("enabled")
                else nullcontext()
            )
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, targets)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
    train_seconds: float,
    val_seconds: float,
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
        "train_seconds": float(train_seconds),
        "val_seconds": float(val_seconds),
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
    stage_name: str | None = None,
    stage_epoch: int | None = None,
    training_protocol: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_epoch": epoch,
        "stage_name": stage_name,
        "stage_epoch": stage_epoch,
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
        "training_protocol": training_protocol or [],
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
    stopped_stages: list[str],
    training_protocol: list[dict[str, Any]],
    train_samples: int,
    val_samples: int,
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
            "num_workers": get_nested(config, "data.num_workers"),
            "pin_memory": get_nested(config, "data.pin_memory"),
            "persistent_workers": get_nested(config, "data.persistent_workers"),
            "prefetch_factor": get_nested(config, "data.prefetch_factor"),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_samples_per_class": get_nested(config, "data.max_samples_per_class"),
            "max_train_samples_per_class": get_nested(config, "data.max_train_samples_per_class"),
            "max_val_samples_per_class": get_nested(config, "data.max_val_samples_per_class"),
            "subset_seed": get_nested(config, "experiment.seed"),
            "subset_strategy": "stratified_random_per_class" if get_nested(config, "data.max_samples_per_class") is not None or get_nested(config, "data.max_train_samples_per_class") is not None or get_nested(config, "data.max_val_samples_per_class") is not None else "full_dataset",
        },
        "performance": {
            "deterministic": get_nested(config, "experiment.deterministic", True),
            "mixed_precision": get_nested(config, "training.mixed_precision", False),
        },
        "training_protocol": training_protocol,
        "early_stopping": {
            "monitor": monitor_name,
            "mode": monitor_mode,
            "stopped_early": stopped_early,
            "stopped_stages": stopped_stages,
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
