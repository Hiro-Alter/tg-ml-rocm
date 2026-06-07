#!/usr/bin/env python3
"""Export final models to portable artifacts under models/."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.checkpoints import checkpoint_class_names, load_model_from_checkpoint
from hand_gesture_rocm.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


DEFAULT_EXPORTS = (
    ("resnet18", Path("checkpoints/resnet18_two_stage/best_model.pth")),
    ("mobilenetv3_small", Path("checkpoints/mobilenetv3_two_stage/best_model.pth")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final models to portable artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--format",
        choices=("all", "torchscript", "onnx", "pth"),
        default="all",
        help="Export format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Portable exports for the final hand gesture models.",
        "input": build_input_metadata(),
        "output": {"type": "logits", "postprocess": "argmax over class_names"},
        "models": [],
    }

    for model_slug, checkpoint_path in DEFAULT_EXPORTS:
        output_dir = args.output_dir / model_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device="cpu")
        model.eval()
        class_names = checkpoint_class_names(checkpoint)
        metadata = build_model_metadata(model_slug, checkpoint_path, checkpoint, class_names)

        written = export_model(torch, model, checkpoint, class_names, output_dir, args.format, args.opset)
        write_json(output_dir / "metadata.json", metadata)
        write_labels(output_dir / "labels.txt", class_names)

        manifest["models"].append(
            {
                "name": model_slug,
                "architecture": metadata["architecture"],
                "directory": str(output_dir),
                "artifacts": [str(path) for path in written],
                "metadata": str(output_dir / "metadata.json"),
                "labels": str(output_dir / "labels.txt"),
            }
        )

    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Exported portable models to: {args.output_dir}")
    return 0


def export_model(torch, model, checkpoint: dict, class_names: tuple[str, ...], output_dir: Path, fmt: str, opset: int) -> list[Path]:
    formats = ("torchscript", "onnx", "pth") if fmt == "all" else (fmt,)
    cleanup_stale_artifacts(output_dir, formats)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    written: list[Path] = []

    with torch.no_grad():
        if "torchscript" in formats:
            path = output_dir / "model_torchscript.pt"
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(path))
            written.append(path)

        if "onnx" in formats:
            path = output_dir / "model.onnx"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using the legacy TorchScript-based ONNX export.*",
                    category=DeprecationWarning,
                )
                torch.onnx.export(
                    model,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["logits"],
                    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                    dynamo=False,
                )
            written.append(path)

        if "pth" in formats:
            path = output_dir / "model_state_dict.pth"
            torch.save(build_pth_payload(model, checkpoint, class_names), path)
            written.append(path)

    return written


def cleanup_stale_artifacts(output_dir: Path, formats: tuple[str, ...]) -> None:
    if "onnx" not in formats:
        return
    stale_external_data = output_dir / "model.onnx.data"
    if stale_external_data.exists():
        stale_external_data.unlink()


def build_input_metadata() -> dict:
    return {
        "shape": ["batch", 3, IMAGE_SIZE, IMAGE_SIZE],
        "color_order": "RGB",
        "image_size": IMAGE_SIZE,
        "value_range_before_normalization": "0.0..1.0",
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }


def build_model_metadata(model_slug: str, checkpoint_path: Path, checkpoint: dict, class_names: tuple[str, ...]) -> dict:
    return {
        "name": model_slug,
        "architecture": checkpoint.get("architecture"),
        "training_mode": checkpoint.get("training_mode"),
        "source_checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_score": checkpoint.get("best_score"),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "input": build_input_metadata(),
        "output": {"type": "logits", "postprocess": "argmax over class_names"},
    }


def build_pth_payload(model, checkpoint: dict, class_names: tuple[str, ...]) -> dict:
    return {
        "architecture": checkpoint.get("architecture"),
        "training_mode": checkpoint.get("training_mode"),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "image_size": IMAGE_SIZE,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "model_state_dict": model.state_dict(),
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_labels(path: Path, class_names: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for class_name in class_names:
            handle.write(f"{class_name}\n")


if __name__ == "__main__":
    raise SystemExit(main())
