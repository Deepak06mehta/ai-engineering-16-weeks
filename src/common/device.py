"""Device selection. Import this everywhere instead of hard-coding "cuda" or "cpu"."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict[str, object]:
    """Everything you must record in a benchmark. See the Experiment Log template."""
    dev = get_device()
    return {
        "torch_version": torch.__version__,
        "device": str(dev),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        # MPS has no float64. Stay in float32 / bfloat16.
        "default_dtype": str(torch.get_default_dtype()),
    }


if __name__ == "__main__":
    for k, v in device_report().items():
        print(f"{k:18s} {v}")
