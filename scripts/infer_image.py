#!/usr/bin/env python3
"""Run inference for a single image using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.checkpoints import checkpoint_class_names, load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer one image with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--architecture", default=None, help="Required only for raw state_dict checkpoints.")
    parser.add_argument("--training-mode", default=None, help="Required only for raw state_dict checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(f"Missing inference dependency: {exc}") from exc

    from hand_gesture_rocm.data import build_transforms

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = resolve_device(args.device, torch)
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
        architecture=args.architecture,
        training_mode=args.training_mode,
    )
    class_names = checkpoint_class_names(checkpoint)
    grayscale_to_rgb = bool(checkpoint.get("config", {}).get("data", {}).get("grayscale_to_rgb", True))
    transform = build_transforms(train=False, grayscale_to_rgb=grayscale_to_rgb)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(args.top_k, len(class_names))
    scores, indices = torch.topk(probabilities, k=top_k)
    predictions = [
        {
            "class_id": int(index),
            "class_name": class_names[int(index)],
            "probability": float(score),
        }
        for score, index in zip(scores.detach().cpu(), indices.detach().cpu())
    ]
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "image": str(image_path),
        "prediction": predictions[0],
        "top_k": predictions,
        "inference_ms": elapsed_ms,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0


def resolve_device(requested: str, torch_module):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but not visible to PyTorch.")
    return torch_module.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
