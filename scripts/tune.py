#!/usr/bin/env python3
"""Run a small hyperparameter grid by delegating each trial to train.py."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.config import get_nested, load_config


SUMMARY_FIELDS = [
    "trial",
    "status",
    "return_code",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "epochs",
    "monitor",
    "monitor_mode",
    "selection_metric",
    "selection_value",
    "best_epoch",
    "best_score",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_f1_macro",
    "final_epoch",
    "stopped_early",
    "run_dir",
    "checkpoint_dir",
    "config_path",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run limited tuning trials from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a tuning YAML config.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap for grid trials.")
    parser.add_argument("--dry-run", action="store_true", help="Write trial configs without training.")
    parser.add_argument("--rerun-existing", action="store_true", help="Run trials even if metrics.json exists.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed trial.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    run_dir = Path(get_nested(config, "outputs.run_dir", "runs/tuning"))
    checkpoint_dir = Path(get_nested(config, "outputs.checkpoint_dir", "checkpoints/tuning"))
    config_dir = run_dir / "trial_configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(config, run_dir, checkpoint_dir)
    if args.max_trials is not None:
        trials = trials[: max(args.max_trials, 0)]
    if not trials:
        raise ValueError("No tuning trials were generated.")

    selection_metric = str(get_nested(config, "tuning.selection_metric", trial_monitor(config)))
    selection_mode = str(get_nested(config, "tuning.selection_mode", infer_metric_mode(selection_metric))).lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    summaries: list[dict[str, Any]] = []
    summary_csv = run_dir / "tuning_summary.csv"
    summary_json = run_dir / "tuning_summary.json"
    best_json = run_dir / "best_trial.json"

    print(f"Tuning config: {args.config}")
    print(f"Trials: {len(trials)}")
    print(f"Outputs: {run_dir}")
    print(f"Selection: {selection_metric} ({selection_mode})")

    for trial in trials:
        trial_config_path = config_dir / f"trial_{trial['trial']:03d}.yaml"
        write_yaml(trial_config_path, trial["config"])

        metrics_path = Path(trial["config"]["outputs"]["run_dir"]) / "metrics.json"
        summary = build_planned_summary(trial, trial_config_path, metrics_path, selection_metric)

        if args.dry_run:
            summary["status"] = "planned"
        elif metrics_path.exists() and not args.rerun_existing:
            summary.update(summary_from_metrics(metrics_path, selection_metric))
            summary["status"] = "skipped_existing"
            summary["return_code"] = 0
        else:
            print(f"\nTrial {trial['trial']:03d}: lr={trial['learning_rate']} wd={trial['weight_decay']}")
            return_code = run_trial(trial_config_path)
            summary["return_code"] = return_code
            if return_code == 0 and metrics_path.exists():
                summary.update(summary_from_metrics(metrics_path, selection_metric))
                summary["status"] = "completed"
            else:
                summary["status"] = "failed"

        summaries.append(summary)
        write_outputs(summary_csv, summary_json, best_json, summaries, selection_mode)

        if summary["status"] == "failed" and args.stop_on_failure:
            return 1

    if args.dry_run:
        print(f"Dry run complete. Trial configs: {config_dir}")
    else:
        print(f"Tuning summary: {summary_csv}")
    return 0 if all(row["status"] != "failed" for row in summaries) else 1


def build_trials(config: dict[str, Any], run_dir: Path, checkpoint_dir: Path) -> list[dict[str, Any]]:
    learning_rates = as_list(get_nested(config, "tuning.learning_rate", [get_nested(config, "training.learning_rate", 1e-3)]))
    weight_decays = as_list(get_nested(config, "tuning.weight_decay", [get_nested(config, "training.weight_decay", 1e-4)]))
    optimizer = str(get_nested(config, "tuning.optimizer", get_nested(config, "training.optimizer", "AdamW")))
    scheduler = str(get_nested(config, "tuning.scheduler", get_nested(config, "training.scheduler", "ReduceLROnPlateau")))
    epochs = int(get_nested(config, "tuning.epochs_per_trial", get_nested(config, "training.epochs", 20)))
    early_stopping = copy.deepcopy(get_nested(config, "tuning.early_stopping", get_nested(config, "training.early_stopping", {})) or {})
    base_name = str(get_nested(config, "experiment.name", "tuning"))

    trials: list[dict[str, Any]] = []
    for index, (learning_rate, weight_decay) in enumerate(product(learning_rates, weight_decays), start=1):
        trial_name = f"{base_name}_trial_{index:03d}_lr{slug_float(learning_rate)}_wd{slug_float(weight_decay)}"
        trial_config = copy.deepcopy(config)
        trial_config.pop("tuning", None)
        trial_config.setdefault("experiment", {})["name"] = trial_name
        trial_config["training"] = {
            "epochs": epochs,
            "optimizer": optimizer,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": scheduler,
            "early_stopping": early_stopping,
        }
        trial_config["outputs"] = {
            "run_dir": str(run_dir / trial_name),
            "checkpoint_dir": str(checkpoint_dir / trial_name),
        }
        trials.append(
            {
                "trial": index,
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epochs": epochs,
                "monitor": str(early_stopping.get("monitor", "val_loss")),
                "monitor_mode": str(early_stopping.get("mode", infer_metric_mode(str(early_stopping.get("monitor", "val_loss"))))),
                "config": trial_config,
            }
        )
    return trials


def build_planned_summary(
    trial: dict[str, Any],
    config_path: Path,
    metrics_path: Path,
    selection_metric: str,
) -> dict[str, Any]:
    return {
        "trial": trial["trial"],
        "status": "pending",
        "return_code": "",
        "learning_rate": trial["learning_rate"],
        "weight_decay": trial["weight_decay"],
        "optimizer": trial["optimizer"],
        "scheduler": trial["scheduler"],
        "epochs": trial["epochs"],
        "monitor": trial["monitor"],
        "monitor_mode": trial["monitor_mode"],
        "selection_metric": selection_metric,
        "selection_value": "",
        "best_epoch": "",
        "best_score": "",
        "best_val_loss": "",
        "best_val_accuracy": "",
        "best_val_f1_macro": "",
        "final_epoch": "",
        "stopped_early": "",
        "run_dir": trial["config"]["outputs"]["run_dir"],
        "checkpoint_dir": trial["config"]["outputs"]["checkpoint_dir"],
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
    }


def summary_from_metrics(metrics_path: Path, selection_metric: str) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    best_validation = metrics.get("best_validation_metrics") or {}
    early_stopping = metrics.get("early_stopping") or {}
    return {
        "selection_value": metric_value(best_validation, selection_metric),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_score": metrics.get("best_score", ""),
        "best_val_loss": best_validation.get("loss", ""),
        "best_val_accuracy": best_validation.get("accuracy", ""),
        "best_val_f1_macro": best_validation.get("f1_macro", ""),
        "final_epoch": metrics.get("final_epoch", ""),
        "stopped_early": early_stopping.get("stopped_early", ""),
    }


def write_outputs(
    summary_csv: Path,
    summary_json: Path,
    best_json: Path,
    summaries: list[dict[str, Any]],
    selection_mode: str,
) -> None:
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    best = best_summary(summaries, selection_mode)
    if best is not None:
        with best_json.open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
    elif best_json.exists():
        best_json.unlink()


def best_summary(summaries: list[dict[str, Any]], selection_mode: str) -> dict[str, Any] | None:
    completed = [row for row in summaries if row["status"] in {"completed", "skipped_existing"} and row["selection_value"] != ""]
    if not completed:
        return None
    key = lambda row: float(row["selection_value"])
    if selection_mode == "min":
        return min(completed, key=key)
    if selection_mode == "max":
        return max(completed, key=key)
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def run_trial(config_path: Path) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", str(config_path)]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write tuning trial configs.") from exc

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def metric_value(best_validation: dict[str, Any], metric_name: str) -> float | str:
    key = normalize_metric(metric_name)
    return best_validation.get(key, "")


def normalize_metric(metric_name: str) -> str:
    normalized = metric_name.lower().replace("-", "_")
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
    if normalized not in mapping:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return mapping[normalized]


def trial_monitor(config: dict[str, Any]) -> str:
    return str(get_nested(config, "tuning.early_stopping.monitor", get_nested(config, "training.early_stopping.monitor", "val_loss")))


def infer_metric_mode(metric_name: str) -> str:
    return "min" if "loss" in metric_name.lower() else "max"


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def slug_float(value: Any) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    sys.exit(main())
