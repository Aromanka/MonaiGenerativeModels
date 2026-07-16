from __future__ import annotations

import json
import math
import os
import random
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Process metadata for one single-process or torchrun-launched worker."""

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def is_main_process() -> bool:
    """Return whether the current process should own user-visible output."""

    return int(os.environ.get("RANK", "0")) == 0


@contextmanager
def distributed_session(requested_device: str = "auto") -> Iterator[DistributedContext]:
    """Initialize DDP from torchrun environment variables, or yield a standalone context."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        yield DistributedContext(device=select_device(requested_device))
        return

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "auto":
        requested = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested CUDA training, but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds the {torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_available():
        raise RuntimeError("This PyTorch build does not provide torch.distributed")
    dist.init_process_group(backend=backend, init_method="env://")
    context = DistributedContext(device=device, rank=rank, local_rank=local_rank, world_size=world_size)
    try:
        yield context
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def wrap_ddp(module: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap a trainable module in DDP while preserving standalone behavior."""

    if not context.distributed:
        return module
    kwargs: dict[str, Any] = {"broadcast_buffers": True, "gradient_as_bucket_view": True}
    if context.device.type == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return DistributedDataParallel(module, **kwargs)


@contextmanager
def ddp_sync_context(modules: tuple[nn.Module, ...], should_sync: bool) -> Iterator[None]:
    """Skip redundant DDP gradient reductions during accumulation microsteps."""

    with ExitStack() as stack:
        if not should_sync:
            for module in modules:
                if isinstance(module, DistributedDataParallel):
                    stack.enter_context(module.no_sync())
        yield


def all_reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def local_batch_limit(max_batches: int | None, context: DistributedContext) -> int | None:
    """Split a global batch cap across ranks without multiplying its meaning by world size."""

    if max_batches is None or not context.distributed:
        return max_batches
    quotient, remainder = divmod(max(0, max_batches), context.world_size)
    return quotient + int(context.rank < remainder)


def all_ranks_finite(value: torch.Tensor, context: DistributedContext) -> bool:
    finite = torch.tensor(int(torch.isfinite(value).all()), device=context.device, dtype=torch.int32)
    if context.distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


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


def capture_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Keep checkpoints compatible with torch.load(weights_only=True).
        "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        state["cuda"] = torch.cuda.get_rng_state(device) if device is not None else torch.cuda.get_rng_state_all()
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
        cuda_state = state["cuda"]
        if torch.is_tensor(cuda_state):
            torch.cuda.set_rng_state(cuda_state)
        else:  # Backward compatibility with checkpoints that stored all visible CUDA devices.
            torch.cuda.set_rng_state_all(cuda_state)


def gather_rng_states(context: DistributedContext) -> list[dict[str, Any]]:
    local_state = capture_rng_state(context.device)
    if not context.distributed:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def restore_distributed_rng_state(checkpoint: dict[str, Any], context: DistributedContext) -> None:
    states = checkpoint.get("rng_states")
    if isinstance(states, list) and context.rank < len(states):
        restore_rng_state(states[context.rank])
    else:
        restore_rng_state(checkpoint.get("rng_state"))


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
