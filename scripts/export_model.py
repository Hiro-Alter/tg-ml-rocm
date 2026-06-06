#!/usr/bin/env python3
"""Export a trained checkpoint to .pth, TorchScript or ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.checkpoints import checkpoint_class_names, load_model_from_checkpoint
from hand_gesture_rocm.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--format", default="all", choices=["pth", "torchscript", "onnx", "all"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--architecture", default=None, help="Required only for raw state_dict checkpoints.")
    parser.add_argument("--training-mode", default=None, help="Required only for raw state_dict checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"Missing export dependency: {exc}") from exc

    device = torch.device("cpu")
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
        architecture=args.architecture,
        training_mode=args.training_mode,
    )
    class_names = checkpoint_class_names(checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = ["pth", "torchscript", "onnx"] if args.format == "all" else [args.format]
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    written = []
    for export_format in formats:
        if export_format == "pth":
            path = output_dir / "model.pth"
            torch.save(build_export_payload(model, checkpoint, class_names), path)
        elif export_format == "torchscript":
            path = output_dir / "model_torchscript.pt"
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(path))
        elif export_format == "onnx":
            path = output_dir / "model.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        else:
            raise ValueError(f"Unsupported format: {export_format}")
        written.append(path)

    for path in written:
        print(f"Saved: {path}")
    return 0


def build_export_payload(model, checkpoint: dict, class_names: tuple[str, ...]) -> dict:
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


if __name__ == "__main__":
    raise SystemExit(main())
