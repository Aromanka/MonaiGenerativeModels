from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def resolve_precision(device: torch.device, requested: str = "auto") -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda" or requested == "fp32":
        return None, False
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return torch.bfloat16, True
    if requested == "fp16":
        return torch.float16, True
    raise ValueError(f"Unknown precision: {requested}")


def autocast_context(device: torch.device, dtype: torch.dtype | None, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(dtype: torch.dtype | None) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Keep checkpoints compatible with torch.load(weights_only=True).
        "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4])
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.05) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


class ModuleEMA:
    """Device-resident EMA for diffusion and condition-projector parameters."""

    def __init__(self, modules: dict[str, nn.Module], decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            f"{prefix}.{name}": parameter.detach().clone()
            for prefix, module in modules.items()
            for name, parameter in module.named_parameters()
        }

    @torch.no_grad()
    def update(self, modules: dict[str, nn.Module]) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for prefix, module in modules.items():
            for name, parameter in module.named_parameters():
                self.shadow[f"{prefix}.{name}"].lerp_(parameter.detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        for name, value in state["shadow"].items():
            if name not in self.shadow:
                raise KeyError(f"Unexpected EMA parameter: {name}")
            self.shadow[name].copy_(value.to(self.shadow[name].device))

    @torch.no_grad()
    def copy_to(self, modules: dict[str, nn.Module]) -> None:
        for prefix, module in modules.items():
            for name, parameter in module.named_parameters():
                parameter.copy_(self.shadow[f"{prefix}.{name}"])
