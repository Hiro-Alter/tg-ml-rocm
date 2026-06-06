#!/usr/bin/env python3
"""Check PyTorch, torchvision and ROCm/CUDA device visibility."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"HIP/ROCm: {getattr(torch.version, 'hip', None) or 'not reported'}")
    print(f"CUDA API available: {torch.cuda.is_available()}")

    try:
        import torchvision
    except ImportError:
        print("torchvision: not installed")
        return 1

    print(f"torchvision: {torchvision.__version__}")

    if not torch.cuda.is_available():
        print("No ROCm/CUDA device is visible to PyTorch.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"Visible devices: {device_count}")
    for index in range(device_count):
        print(f"Device {index}: {torch.cuda.get_device_name(index)}")

    device = torch.device("cuda:0")
    sample = torch.ones((1, 3, 224, 224), device=device)
    print(f"Tensor allocation test: ok on {sample.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
