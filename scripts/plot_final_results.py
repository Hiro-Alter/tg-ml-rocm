#!/usr/bin/env python3
"""Generate final Spanish-labeled plots for the project report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


MODELS = {
    "ResNet18": {
        "slug": "resnet18",
        "run_dir": Path("runs/resnet18_two_stage"),
        "color": "#1f77b4",
    },
    "MobileNetV3 Small": {
        "slug": "mobilenetv3_small",
        "run_dir": Path("runs/mobilenetv3_two_stage"),
        "color": "#d62728",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera graficas finales del proyecto.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directorio base de runs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/final_results"),
        help="Directorio de salida para las figuras.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    model_data = load_model_data(args.runs_dir)
    saved_paths = []

    for model_name, data in model_data.items():
        saved_paths.append(
            plot_training_curves(
                model_name,
                data["history"],
                output_dir / f"01_entrenamiento_{MODELS[model_name]['slug']}.png",
            )
        )

    saved_paths.append(plot_test_comparison(model_data, output_dir / "02_comparacion_test.png"))

    for model_name, data in model_data.items():
        saved_paths.append(
            plot_confusion_matrix(
                model_name,
                data["test_metrics"],
                output_dir / f"03_matriz_confusion_{MODELS[model_name]['slug']}.png",
            )
        )

    saved_paths.append(plot_inference_time(model_data, output_dir / "04_tiempo_inferencia.png"))
    saved_paths.append(
        plot_tuning_resnet18(
            args.runs_dir / "tuning_resnet18" / "tuning_summary.csv",
            output_dir / "05_tuning_resnet18.png",
        )
    )
    saved_paths.append(
        plot_rocm_dataloader(
            args.runs_dir / "perf_resnet18_rocm" / "perf_summary.csv",
            output_dir / "06_diagnostico_rocm_dataloader.png",
        )
    )

    print("Figuras generadas:")
    for path in saved_paths:
        print(f"- {path}")
    return 0


def setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def load_model_data(runs_dir: Path) -> dict[str, dict]:
    data = {}
    for model_name, config in MODELS.items():
        run_dir = runs_dir / config["run_dir"].relative_to("runs")
        history_path = run_dir / "training_history.csv"
        metrics_path = run_dir / "test_evaluation" / "metrics.json"
        data[model_name] = {
            "history": read_history(history_path),
            "test_metrics": read_json(metrics_path),
        }
    return data


def read_history(path: Path) -> list[dict[str, float | bool]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el historial: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                parsed[key] = value.lower() == "true" if key == "is_best" else float(value)
            rows.append(parsed)
        if not rows:
            raise ValueError(f"Historial vacio: {path}")
        return rows


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de metricas: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(model_name: str, rows: list[dict[str, float | bool]], output_path: Path) -> Path:
    from matplotlib import pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    best_epochs = [int(row["epoch"]) for row in rows if row.get("is_best")]
    best_epoch = best_epochs[-1] if best_epochs else None
    best_row = next((row for row in rows if int(row["epoch"]) == best_epoch), None) if best_epoch else None

    panels = (
        ("train_loss", "val_loss", "Loss", "Loss"),
        ("train_accuracy", "val_accuracy", "Accuracy", "Accuracy"),
        ("train_f1_macro", "val_f1_macro", "F1 macro", "F1 macro"),
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, (train_key, val_key, ylabel, title) in zip(axes, panels):
        val_values = [row[val_key] for row in rows]
        axis.plot(epochs, [row[train_key] for row in rows], marker="o", linewidth=1.8, label="Train")
        axis.plot(epochs, val_values, marker="o", linewidth=1.8, label="Validation")
        if best_epoch is not None:
            axis.axvline(best_epoch, color="#444444", linestyle="--", linewidth=1.0, label="Best epoch")
        if best_row is not None:
            best_value = float(best_row[val_key])
            offset = (12, -28) if best_value > 0.9 else (12, 18)
            axis.scatter([best_epoch], [best_value], color="#111111", s=28, zorder=5)
            axis.annotate(
                f"Best val: {best_value:.4f}",
                xy=(best_epoch, best_value),
                xytext=offset,
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.9},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#999999", "alpha": 0.9},
                fontsize=8,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if "accuracy" in train_key or "f1" in train_key:
            axis.set_ylim(0.75, 1.01)
        axis.legend(loc="best")

    figure.suptitle(f"Curvas de entrenamiento final - {model_name}", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_test_comparison(model_data: dict[str, dict], output_path: Path) -> Path:
    from matplotlib import pyplot as plt

    model_names = list(model_data)
    x = np.arange(len(model_names))
    width = 0.36
    accuracy = [model_data[name]["test_metrics"]["accuracy"] * 100 for name in model_names]
    f1_macro = [model_data[name]["test_metrics"]["f1_macro"] * 100 for name in model_names]
    loss = [model_data[name]["test_metrics"]["loss"] for name in model_names]
    colors = [MODELS[name]["color"] for name in model_names]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    metric_axis = axes[0]
    bars_acc = metric_axis.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2a9d8f")
    bars_f1 = metric_axis.bar(x + width / 2, f1_macro, width, label="F1 macro", color="#e9c46a")
    metric_axis.set_xticks(x, model_names)
    metric_axis.set_ylabel("Score (%)")
    metric_axis.set_title("Final test metrics")
    metric_axis.set_ylim(98.5, 100.05)
    metric_axis.legend()
    add_bar_labels(metric_axis, bars_acc, suffix="%")
    add_bar_labels(metric_axis, bars_f1, suffix="%")

    loss_axis = axes[1]
    bars_loss = loss_axis.bar(model_names, loss, color=colors)
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Test loss")
    loss_axis.set_ylim(0, max(loss) * 1.25)
    add_bar_labels(loss_axis, bars_loss, decimals=4)

    figure.suptitle("Comparación final sobre el conjunto de test", y=1.03)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_confusion_matrix(model_name: str, metrics: dict, output_path: Path) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0) * 100
    class_names = list(metrics["class_names"])

    figure, axis = plt.subplots(figsize=(10, 8.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    colorbar.set_label("% per true class")

    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"Normalized confusion matrix - {model_name}")

    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            value = normalized[row_index, column_index]
            if row_index == column_index or value >= 0.1:
                color = "white" if value >= 55 else "#222222"
                axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_inference_time(model_data: dict[str, dict], output_path: Path) -> Path:
    from matplotlib import pyplot as plt

    model_names = list(model_data)
    values = [model_data[name]["test_metrics"]["avg_inference_ms_per_image"] for name in model_names]
    colors = [MODELS[name]["color"] for name in model_names]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(model_names, values, color=colors)
    axis.set_ylabel("ms/image")
    axis.set_title("Average inference time on test")
    axis.set_ylim(0, max(values) * 1.25)
    add_bar_labels(axis, bars, decimals=3, suffix=" ms")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_tuning_resnet18(summary_path: Path, output_path: Path) -> Path:
    from matplotlib import pyplot as plt

    rows = read_tuning_summary(summary_path)
    if not rows:
        raise ValueError(f"No hay trials completados en: {summary_path}")

    best = min(rows, key=lambda row: row["best_val_loss"])
    labels = [f"T{row['trial']}\nlr={format_float(row['learning_rate'])}\nwd={format_float(row['weight_decay'])}" for row in rows]
    x = np.arange(len(rows))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    loss_values = [row["best_val_loss"] for row in rows]
    colors = ["#d62728" if row["trial"] == best["trial"] else "#1f77b4" for row in rows]
    bars = axes[0].bar(x, loss_values, color=colors)
    axes[0].set_title("Best validation loss by trial")
    axes[0].set_ylabel("val_loss")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(loss_values) * 1.25)
    add_bar_labels(axes[0], bars, decimals=4)

    accuracy = [row["best_val_accuracy"] * 100 for row in rows]
    f1_macro = [row["best_val_f1_macro"] * 100 for row in rows]
    width = 0.36
    axes[1].bar(x - width / 2, accuracy, width, label="val_accuracy", color="#2a9d8f")
    bars_f1 = axes[1].bar(x + width / 2, f1_macro, width, label="val_f1_macro", color="#e9c46a")
    axes[1].set_title("Best validation scores by trial")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(96.5, 99.0)
    axes[1].legend()
    add_bar_labels(axes[1], bars_f1, decimals=2, suffix="%")

    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)

    figure.suptitle(
        f"ResNet18 tuning - selected trial T{best['trial']} "
        f"(lr={format_float(best['learning_rate'])}, wd={format_float(best['weight_decay'])})",
        y=1.04,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_rocm_dataloader(summary_path: Path, output_path: Path) -> Path:
    from matplotlib import pyplot as plt

    rows = read_perf_summary(summary_path)
    if not rows:
        raise ValueError(f"No hay trials completados en: {summary_path}")

    batch_colors = {32: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c"}
    variant_styles = {"baseline": "-", "miopen_search": "--"}
    variant_names = {"baseline": "Baseline FP32", "miopen_search": "MIOpen search"}

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for variant in sorted({row["variant"] for row in rows}):
        for batch_size in sorted({row["batch_size"] for row in rows}):
            series = sorted(
                [row for row in rows if row["variant"] == variant and row["batch_size"] == batch_size],
                key=lambda item: item["num_workers"],
            )
            if not series:
                continue
            label = f"{variant_names.get(variant, variant)} - batch {batch_size}"
            workers = [row["num_workers"] for row in series]
            throughput = [row["timed_epoch_img_s_mean"] for row in series]
            epoch_seconds = [row["timed_epoch_seconds_mean"] for row in series]
            axes[0].plot(
                workers,
                throughput,
                marker="o",
                color=batch_colors[batch_size],
                linestyle=variant_styles.get(variant, "-"),
                label=label,
            )
            axes[1].plot(
                workers,
                epoch_seconds,
                marker="o",
                color=batch_colors[batch_size],
                linestyle=variant_styles.get(variant, "-"),
                label=label,
            )

    best = max(rows, key=lambda row: row["timed_epoch_img_s_mean"])
    axes[0].set_ylim(
        min(row["timed_epoch_img_s_mean"] for row in rows) - 1.0,
        max(row["timed_epoch_img_s_mean"] for row in rows) + 2.0,
    )
    axes[0].annotate(
        f"Best: batch {best['batch_size']}, workers {best['num_workers']}",
        xy=(best["num_workers"], best["timed_epoch_img_s_mean"]),
        xytext=(best["num_workers"] + 0.8, best["timed_epoch_img_s_mean"] + 1.1),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#999999", "alpha": 0.9},
        fontsize=9,
        ha="left",
    )

    axes[0].set_title("Throughput by configuration")
    axes[0].set_xlabel("num_workers")
    axes[0].set_ylabel("Images/s (train + val)")
    axes[1].set_title("Epoch time")
    axes[1].set_xlabel("num_workers")
    axes[1].set_ylabel("Seconds/epoch")
    for axis in axes:
        axis.set_xticks([0, 2, 4, 6, 8])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    figure.suptitle("Diagnóstico ROCm/DataLoader - ResNet18", y=1.03)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def read_perf_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el resumen de rendimiento: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "completed":
                continue
            rows.append(
                {
                    "variant": row["variant"],
                    "batch_size": int(row["batch_size"]),
                    "num_workers": int(row["num_workers"]),
                    "timed_epoch_img_s_mean": float(row["timed_epoch_img_s_mean"]),
                    "timed_epoch_seconds_mean": float(row["timed_epoch_seconds_mean"]),
                }
            )
    return rows


def read_tuning_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el resumen de tuning: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "completed":
                continue
            rows.append(
                {
                    "trial": int(row["trial"]),
                    "learning_rate": float(row["learning_rate"]),
                    "weight_decay": float(row["weight_decay"]),
                    "best_val_loss": float(row["best_val_loss"]),
                    "best_val_accuracy": float(row["best_val_accuracy"]),
                    "best_val_f1_macro": float(row["best_val_f1_macro"]),
                }
            )
    return rows


def format_float(value: float) -> str:
    return f"{value:.0e}" if value < 0.001 else f"{value:g}"


def add_bar_labels(axis, bars, decimals: int = 2, suffix: str = "") -> None:
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.{decimals}f}{suffix}"
        axis.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


if __name__ == "__main__":
    raise SystemExit(main())
