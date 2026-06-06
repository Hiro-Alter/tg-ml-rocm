#!/usr/bin/env python3
"""Run short training performance diagnostics with resource sampling."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.config import get_nested, load_config


VARIANT_ENVS = {
    "baseline": {},
    "amp_fp16": {},
    "miopen_search": {
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_FIND_ENFORCE": "SEARCH",
    },
    "miopen_amp_fp16": {
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_FIND_ENFORCE": "SEARCH",
    },
    "tunableop": {
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
    },
    "miopen_tunableop": {
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_FIND_ENFORCE": "SEARCH",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
    },
}

RESOURCE_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "cpu_percent",
    "ram_used_percent",
    "ram_used_mb",
    "process_tree_rss_mb",
    "disk_read_mb_s",
    "disk_write_mb_s",
    "gpu_use_percent",
    "vram_used_percent",
    "vram_used_mb",
    "vram_total_mb",
    "gpu_power_w",
    "gpu_temp_edge_c",
]

SUMMARY_FIELDS = [
    "trial",
    "status",
    "return_code",
    "variant",
    "mixed_precision",
    "batch_size",
    "num_workers",
    "timed_epochs",
    "timed_epoch_seconds_mean",
    "timed_train_seconds_mean",
    "timed_val_seconds_mean",
    "timed_train_img_s_mean",
    "timed_epoch_img_s_mean",
    "gpu_use_avg",
    "gpu_use_max",
    "vram_used_mb_max",
    "cpu_percent_avg",
    "cpu_percent_max",
    "ram_used_percent_max",
    "process_tree_rss_mb_max",
    "disk_read_mb_s_avg",
    "disk_write_mb_s_avg",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_f1_macro",
    "run_dir",
    "checkpoint_dir",
    "config_path",
    "resource_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark short train.py runs across DataLoader and ROCm settings.")
    parser.add_argument("--config", required=True, help="Base YAML config for diagnostics.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--num-workers", nargs="+", type=int, default=[0, 2, 4, 6, 8])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANT_ENVS),
        default=["baseline", "miopen_search", "amp_fp16", "miopen_amp_fp16"],
    )
    parser.add_argument("--timed-epochs", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write trial configs without training.")
    parser.add_argument("--rerun-existing", action="store_true", help="Run trials even if metrics.json exists.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed trial.")
    parser.add_argument("--yes", action="store_true", help="Allow running more than 15 trials.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(get_nested(config, "outputs.run_dir", "runs/perf_diagnostics"))
    checkpoint_dir = Path(get_nested(config, "outputs.checkpoint_dir", "checkpoints/perf_diagnostics"))
    config_dir = run_dir / "trial_configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(config, args, run_dir, checkpoint_dir)
    if args.max_trials is not None:
        trials = trials[: max(args.max_trials, 0)]
    if not trials:
        raise ValueError("No diagnostic trials were generated.")
    if not args.dry_run and len(trials) > 15 and not args.yes:
        raise SystemExit(f"Refusing to run {len(trials)} trials without --yes.")

    summary_csv = run_dir / "perf_summary.csv"
    summary_json = run_dir / "perf_summary.json"
    summaries: list[dict[str, Any]] = []

    print(f"Diagnostics config: {args.config}")
    print(f"Trials: {len(trials)}")
    print(f"Outputs: {run_dir}")
    print(f"Timed epochs: {','.join(str(epoch) for epoch in args.timed_epochs)}")

    for trial in trials:
        config_path = config_dir / f"trial_{trial['trial']:03d}_{trial['name']}.yaml"
        write_yaml(config_path, trial["config"])
        metrics_path = Path(trial["run_dir"]) / "metrics.json"
        resource_path = Path(trial["run_dir"]) / "resource_samples.csv"
        summary = planned_summary(trial, config_path, resource_path, args.timed_epochs)

        if args.dry_run:
            summary["status"] = "planned"
        elif metrics_path.exists() and not args.rerun_existing:
            summary.update(completed_summary(trial, metrics_path, resource_path, args.timed_epochs))
            summary["status"] = "skipped_existing"
            summary["return_code"] = 0
        else:
            print(
                f"\nTrial {trial['trial']:03d}: "
                f"variant={trial['variant']} bs={trial['batch_size']} workers={trial['num_workers']}"
            )
            return_code = run_trial(config_path, trial, resource_path, args.sample_interval)
            summary["return_code"] = return_code
            if return_code == 0 and metrics_path.exists():
                summary.update(completed_summary(trial, metrics_path, resource_path, args.timed_epochs))
                summary["status"] = "completed"
            else:
                summary["status"] = "failed"

        summaries.append(summary)
        write_summary(summary_csv, summary_json, summaries)

        if summary["status"] == "failed" and args.stop_on_failure:
            return 1

    if args.dry_run:
        print(f"Dry run complete. Trial configs: {config_dir}")
    else:
        print(f"Diagnostics summary: {summary_csv}")
    return 0 if all(row["status"] != "failed" for row in summaries) else 1


def build_trials(config: dict[str, Any], args: argparse.Namespace, run_dir: Path, checkpoint_dir: Path) -> list[dict[str, Any]]:
    base_name = str(get_nested(config, "experiment.name", "perf"))
    trials: list[dict[str, Any]] = []
    for index, (variant, batch_size, num_workers) in enumerate(
        product(args.variants, args.batch_sizes, args.num_workers),
        start=1,
    ):
        trial_config = copy.deepcopy(config)
        trial_name = f"{base_name}_{variant}_bs{batch_size}_nw{num_workers}"
        trial_run_dir = run_dir / variant / f"bs{batch_size}_nw{num_workers}"
        trial_checkpoint_dir = checkpoint_dir / variant / f"bs{batch_size}_nw{num_workers}"

        trial_config.setdefault("experiment", {})["name"] = trial_name
        trial_config.setdefault("data", {})["batch_size"] = int(batch_size)
        trial_config.setdefault("data", {})["num_workers"] = int(num_workers)
        trial_config["data"]["persistent_workers"] = bool(num_workers > 0)
        trial_config["data"]["prefetch_factor"] = 2 if num_workers > 0 else None
        trial_config.setdefault("training", {})["mixed_precision"] = mixed_precision_config(variant)
        trial_config["outputs"] = {
            "run_dir": str(trial_run_dir),
            "checkpoint_dir": str(trial_checkpoint_dir),
        }

        env = dict(VARIANT_ENVS[variant])
        if "PYTORCH_TUNABLEOP_ENABLED" in env:
            env["PYTORCH_TUNABLEOP_FILENAME"] = str(trial_run_dir / "tunableop_results.csv")

        trials.append(
            {
                "trial": index,
                "name": f"{variant}_bs{batch_size}_nw{num_workers}",
                "variant": variant,
                "mixed_precision": "fp16" if variant.endswith("amp_fp16") else "fp32",
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "run_dir": str(trial_run_dir),
                "checkpoint_dir": str(trial_checkpoint_dir),
                "config": trial_config,
                "env": env,
            }
        )
    return trials


def mixed_precision_config(variant: str) -> dict[str, Any]:
    return {
        "enabled": variant.endswith("amp_fp16"),
        "dtype": "float16",
        "grad_scaler": True,
    }


def run_trial(config_path: Path, trial: dict[str, Any], resource_path: Path, sample_interval: float) -> int:
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(trial["env"])
    command = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", str(config_path)]
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
    monitor_process(process, resource_path, sample_interval)
    return int(process.returncode or 0)


def monitor_process(process: subprocess.Popen, resource_path: Path, sample_interval: float) -> None:
    sampler = ResourceSampler(process.pid)
    with resource_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOURCE_FIELDS)
        writer.writeheader()
        while True:
            writer.writerow(sampler.sample())
            handle.flush()
            if process.poll() is not None:
                break
            time.sleep(max(sample_interval, 0.2))


class ResourceSampler:
    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self.start_time = time.time()
        self.prev_cpu = read_cpu_times()
        self.prev_disk = read_disk_bytes()
        self.prev_time = self.start_time
        self.first_sample = True

    def sample(self) -> dict[str, Any]:
        now = time.time()
        cpu = read_cpu_times()
        disk = read_disk_bytes()
        elapsed_delta = max(now - self.prev_time, 1e-6)
        if self.first_sample:
            cpu_percent = 0.0
            disk_read_mb_s = 0.0
            disk_write_mb_s = 0.0
            self.first_sample = False
        else:
            cpu_percent = compute_cpu_percent(self.prev_cpu, cpu)
            disk_read_mb_s = (disk["read_bytes"] - self.prev_disk["read_bytes"]) / elapsed_delta / 1024 / 1024
            disk_write_mb_s = (disk["write_bytes"] - self.prev_disk["write_bytes"]) / elapsed_delta / 1024 / 1024
        mem = read_meminfo()
        gpu = read_rocm_smi()
        rss_mb = process_tree_rss_mb(self.root_pid)

        self.prev_cpu = cpu
        self.prev_disk = disk
        self.prev_time = now

        return {
            "timestamp": int(now),
            "elapsed_seconds": round(now - self.start_time, 3),
            "cpu_percent": round(cpu_percent, 3),
            "ram_used_percent": round(mem["used_percent"], 3),
            "ram_used_mb": round(mem["used_mb"], 3),
            "process_tree_rss_mb": round(rss_mb, 3),
            "disk_read_mb_s": round(max(disk_read_mb_s, 0.0), 3),
            "disk_write_mb_s": round(max(disk_write_mb_s, 0.0), 3),
            "gpu_use_percent": blank_if_none(gpu.get("gpu_use_percent")),
            "vram_used_percent": blank_if_none(gpu.get("vram_used_percent")),
            "vram_used_mb": blank_if_none(gpu.get("vram_used_mb")),
            "vram_total_mb": blank_if_none(gpu.get("vram_total_mb")),
            "gpu_power_w": blank_if_none(gpu.get("gpu_power_w")),
            "gpu_temp_edge_c": blank_if_none(gpu.get("gpu_temp_edge_c")),
        }


def read_cpu_times() -> dict[str, int]:
    with Path("/proc/stat").open("r", encoding="utf-8") as handle:
        parts = handle.readline().split()[1:]
    values = [int(value) for value in parts]
    idle = values[3] + values[4]
    return {"total": sum(values), "idle": idle}


def compute_cpu_percent(previous: dict[str, int], current: dict[str, int]) -> float:
    total_delta = current["total"] - previous["total"]
    idle_delta = current["idle"] - previous["idle"]
    if total_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def read_meminfo() -> dict[str, float]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0])
    total_kb = values["MemTotal"]
    available_kb = values.get("MemAvailable", 0)
    used_kb = total_kb - available_kb
    return {
        "used_mb": used_kb / 1024,
        "used_percent": 100.0 * used_kb / max(total_kb, 1),
    }


def read_disk_bytes() -> dict[str, int]:
    read_sectors = 0
    write_sectors = 0
    with Path("/proc/diskstats").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if should_ignore_disk(name):
                continue
            read_sectors += int(parts[5])
            write_sectors += int(parts[9])
    return {
        "read_bytes": read_sectors * 512,
        "write_bytes": write_sectors * 512,
    }


def should_ignore_disk(name: str) -> bool:
    if name.startswith(("loop", "ram", "zram")):
        return True
    if re.match(r"^(sd[a-z]|vd[a-z]|xvd[a-z]|nvme\d+n\d+|hd[a-z]|dm-\d+)$", name):
        return False
    return True


def read_rocm_smi() -> dict[str, float]:
    command = [
        "rocm-smi",
        "--showuse",
        "--showmemuse",
        "--showmeminfo",
        "vram",
        "--showtemp",
        "--showpower",
        "--json",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
        if completed.returncode != 0:
            return {}
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    if not payload:
        return {}
    card = next(iter(payload.values()))
    vram_used_b = parse_number(card.get("VRAM Total Used Memory (B)"))
    vram_total_b = parse_number(card.get("VRAM Total Memory (B)"))
    return {
        "gpu_use_percent": parse_number(card.get("GPU use (%)")),
        "vram_used_percent": parse_number(card.get("GPU Memory Allocated (VRAM%)")),
        "vram_used_mb": round(vram_used_b / 1024 / 1024, 3) if vram_used_b is not None else "",
        "vram_total_mb": round(vram_total_b / 1024 / 1024, 3) if vram_total_b is not None else "",
        "gpu_power_w": parse_number(card.get("Average Graphics Package Power (W)")),
        "gpu_temp_edge_c": parse_number(card.get("Temperature (Sensor edge) (C)")),
    }


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def process_tree_rss_mb(root_pid: int) -> float:
    pids = process_tree_pids(root_pid)
    rss_kb = 0
    for pid in pids:
        status_path = Path("/proc") / str(pid) / "status"
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss_kb += int(line.split()[1])
                        break
        except OSError:
            continue
    return rss_kb / 1024


def process_tree_pids(root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            ppid = read_ppid(pid)
        except OSError:
            continue
        children.setdefault(ppid, []).append(pid)

    result: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in result:
            continue
        result.add(pid)
        stack.extend(children.get(pid, []))
    return result


def read_ppid(pid: int) -> int:
    with (Path("/proc") / str(pid) / "status").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("PPid:"):
                return int(line.split()[1])
    return 0


def completed_summary(
    trial: dict[str, Any],
    metrics_path: Path,
    resource_path: Path,
    timed_epochs: list[int],
) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    history_path = Path(metrics["outputs"]["history_path"])
    train_samples = int(metrics["data"]["train_samples"])
    val_samples = int(metrics["data"]["val_samples"])
    history = summarize_history(history_path, timed_epochs, train_samples, val_samples)
    resources = summarize_resources(resource_path)
    best_validation = metrics.get("best_validation_metrics") or {}
    return {
        **history,
        **resources,
        "best_val_loss": best_validation.get("loss", ""),
        "best_val_accuracy": best_validation.get("accuracy", ""),
        "best_val_f1_macro": best_validation.get("f1_macro", ""),
    }


def summarize_history(
    history_path: Path,
    timed_epochs: list[int],
    train_samples: int,
    val_samples: int,
) -> dict[str, Any]:
    rows = read_csv(history_path)
    selected = [row for row in rows if int(row["epoch"]) in set(timed_epochs)]
    if not selected:
        selected = rows
    epoch_seconds = mean_float(selected, "epoch_seconds")
    train_seconds = mean_float(selected, "train_seconds")
    val_seconds = mean_float(selected, "val_seconds")
    return {
        "timed_epoch_seconds_mean": epoch_seconds,
        "timed_train_seconds_mean": train_seconds,
        "timed_val_seconds_mean": val_seconds,
        "timed_train_img_s_mean": safe_div(train_samples, train_seconds),
        "timed_epoch_img_s_mean": safe_div(train_samples + val_samples, epoch_seconds),
    }


def summarize_resources(resource_path: Path) -> dict[str, Any]:
    if not resource_path.exists():
        return {
            "gpu_use_avg": "",
            "gpu_use_max": "",
            "vram_used_mb_max": "",
            "cpu_percent_avg": "",
            "cpu_percent_max": "",
            "ram_used_percent_max": "",
            "process_tree_rss_mb_max": "",
            "disk_read_mb_s_avg": "",
            "disk_write_mb_s_avg": "",
        }
    rows = read_csv(resource_path)
    return {
        "gpu_use_avg": mean_float(rows, "gpu_use_percent"),
        "gpu_use_max": max_float(rows, "gpu_use_percent"),
        "vram_used_mb_max": max_float(rows, "vram_used_mb"),
        "cpu_percent_avg": mean_float(rows, "cpu_percent"),
        "cpu_percent_max": max_float(rows, "cpu_percent"),
        "ram_used_percent_max": max_float(rows, "ram_used_percent"),
        "process_tree_rss_mb_max": max_float(rows, "process_tree_rss_mb"),
        "disk_read_mb_s_avg": mean_float(rows, "disk_read_mb_s"),
        "disk_write_mb_s_avg": mean_float(rows, "disk_write_mb_s"),
    }


def planned_summary(
    trial: dict[str, Any],
    config_path: Path,
    resource_path: Path,
    timed_epochs: list[int],
) -> dict[str, Any]:
    return {
        "trial": trial["trial"],
        "status": "pending",
        "return_code": "",
        "variant": trial["variant"],
        "mixed_precision": trial["mixed_precision"],
        "batch_size": trial["batch_size"],
        "num_workers": trial["num_workers"],
        "timed_epochs": ",".join(str(epoch) for epoch in timed_epochs),
        "timed_epoch_seconds_mean": "",
        "timed_train_seconds_mean": "",
        "timed_val_seconds_mean": "",
        "timed_train_img_s_mean": "",
        "timed_epoch_img_s_mean": "",
        "gpu_use_avg": "",
        "gpu_use_max": "",
        "vram_used_mb_max": "",
        "cpu_percent_avg": "",
        "cpu_percent_max": "",
        "ram_used_percent_max": "",
        "process_tree_rss_mb_max": "",
        "disk_read_mb_s_avg": "",
        "disk_write_mb_s_avg": "",
        "best_val_loss": "",
        "best_val_accuracy": "",
        "best_val_f1_macro": "",
        "run_dir": trial["run_dir"],
        "checkpoint_dir": trial["checkpoint_dir"],
        "config_path": str(config_path),
        "resource_path": str(resource_path),
    }


def mean_float(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return round(sum(values) / len(values), 6)


def max_float(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return round(max(values), 6)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def safe_div(numerator: float, denominator: float | str) -> float | str:
    if denominator == "" or float(denominator) <= 0:
        return ""
    return round(float(numerator) / float(denominator), 6)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(summary_csv: Path, summary_json: Path, summaries: list[dict[str, Any]]) -> None:
    merged = merge_existing_summaries(summary_csv, summaries)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)


def merge_existing_summaries(summary_csv: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    if summary_csv.exists():
        for row in read_csv(summary_csv):
            merged[summary_key(row)] = normalize_summary_row(row)
    for row in summaries:
        merged[summary_key(row)] = normalize_summary_row(row)
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row["variant"]),
            int(row["batch_size"]),
            int(row["num_workers"]),
        ),
    )


def summary_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["variant"]), str(row["batch_size"]), str(row["num_workers"]))


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write diagnostic trial configs.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


if __name__ == "__main__":
    sys.exit(main())
