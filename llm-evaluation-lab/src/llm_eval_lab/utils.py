"""Shared helpers: seeding, device selection, environment metadata."""
from __future__ import annotations

import platform
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch missing or broken: seeding the other RNGs is still useful
        pass


def select_device(preferred: str | None = None) -> str:
    """Best available torch device: cuda, then mps (Apple Silicon), then cpu."""
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def environment_info() -> dict:
    """Software/hardware context stored next to each run so results stay interpretable."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # metadata collection must never break a run
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    return info
