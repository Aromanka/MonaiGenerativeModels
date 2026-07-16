from __future__ import annotations

import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .config import ProjectConfig
from .data import create_datasets, create_loader, load_manifest, safe_torch_load
from .models import build_cxr_autoencoder, load_module_checkpoint
from .runtime import (
    DistributedContext,
    all_ranks_finite,
    all_reduce_sum,
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    cosine_warmup_lambda,
    ddp_sync_context,
    gather_rng_states,
    local_batch_limit,
    make_grad_scaler,
    resolve_precision,
    restore_distributed_rng_state,
    seed_everything,
    select_device,
    wrap_ddp,
)


def _kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    variance = sigma.float().pow(2).clamp_min(1e-12)
    per_item = 0.5 * torch.sum(mu.float().pow(2) + variance - torch.log(variance) - 1.0, dim=(1, 2, 3))
    return per_item.mean()


def _load_autoencoder(path: Path, use_checkpointing: bool, device: torch.device) -> torch.nn.Module:
    model = build_cxr_autoencoder(use_checkpointing=use_checkpointing)
    load_module_checkpoint(model, path, preferred_keys=("model", "autoencoder"), strict=True)
    return model.to(device)


@torch.no_grad()
def validate_autoencoder(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    amp_enabled: bool,
    max_batches: int | None = None,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    context = distributed or DistributedContext(device=device)
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    element_count = 0
    local_max_batches = local_batch_limit(max_batches, context)
    total = max(0, min(len(loader), local_max_batches)) if local_max_batches is not None else len(loader)
    progress = tqdm(
        islice(loader, total),
        total=total,
        desc="Validating autoencoder",
        unit="batch",
        leave=False,
        disable=not context.is_main_process,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, dtype, amp_enabled):
            reconstruction = model.reconstruct(images)
        delta = reconstruction.float() - images.float()
        absolute_error += delta.abs().sum().item()
        squared_error += delta.pow(2).sum().item()
        element_count += delta.numel()
    statistics = torch.tensor(
        [absolute_error, squared_error, element_count], device=device, dtype=torch.float64
    )
    all_reduce_sum(statistics, context)
    absolute_error, squared_error, element_count = statistics.tolist()
    if element_count == 0:
        raise ValueError("Validation loader produced no images")
    mae = absolute_error / element_count
    mse = squared_error / element_count
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def _autoencoder_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val: float,
    config: ProjectConfig,
    manifest: dict[str, Any],
    rng_states: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    return {
        "kind": "oct_autoencoder",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_mae": best_val,
        "rng_state": rng_states[0],
        "rng_states": rng_states,
        "world_size": world_size,
        "manifest_fingerprint": manifest["fingerprint"],
        "config": config.data,
    }


def train_autoencoder(
    config: ProjectConfig,
    resume: str | Path | None = None,
    distributed: DistributedContext | None = None,
) -> Path:
    context = distributed or DistributedContext(
        device=select_device(str(config.get("training.device", "auto")))
    )
    seed = int(config.get("training.seed", 2026))
    seed_everything(seed + context.rank)
    device = context.device
    dtype, amp_enabled = resolve_precision(device, str(config.get("training.precision", "auto")))
    if context.is_main_process:
        manifest = load_manifest(config)
    context.barrier()
    if not context.is_main_process:
        manifest = load_manifest(config, create_if_missing=False)
    train_dataset, val_dataset = create_datasets(config, manifest)
    val_loader = create_loader(val_dataset, config, training=False, distributed=context)

    use_checkpointing = bool(config.get("autoencoder.activation_checkpointing", False))
    source_checkpoint = config.path("paths.cxr_autoencoder_checkpoint")
    model = _load_autoencoder(source_checkpoint, use_checkpointing, device)
    learning_rate = float(config.get("autoencoder.learning_rate", 1e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(config.get("autoencoder.weight_decay", 0.01)),
    )
    epochs = int(config.get("autoencoder.max_epochs", config.get("autoencoder.epochs", 20)))
    if epochs <= 0:
        raise ValueError("autoencoder.max_epochs must be positive")
    stage_batch_size = int(config.get("autoencoder.batch_size", config.get("data.batch_size", 4)))
    if stage_batch_size <= 0:
        raise ValueError("autoencoder.batch_size must be positive")
    accumulation = max(
        1,
        int(
            config.get(
                "autoencoder.gradient_accumulation_steps",
                config.get("training.gradient_accumulation_steps", 1),
            )
        ),
    )
    schedule_loader = create_loader(
        train_dataset,
        config,
        training=True,
        epoch=0,
        distributed=context,
        batch_size=stage_batch_size,
    )
    steps_per_epoch = max(1, math.ceil(len(schedule_loader) / accumulation))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * float(config.get("training.warmup_fraction", 0.03)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_lambda(step, total_steps, warmup_steps)
    )
    scaler = make_grad_scaler(dtype)

    perceptual_weight = float(config.get("autoencoder.perceptual_weight", 0.001))
    perceptual_loss = None
    if perceptual_weight > 0:
        try:
            from generative.losses import PerceptualLoss

            perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(device).eval()
        except ImportError as error:
            raise ImportError(
                "autoencoder.perceptual_weight > 0 requires the optional 'lpips' package; install requirements-oct.txt "
                "or set the weight to 0"
            ) from error

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if resume is not None:
        checkpoint = safe_torch_load(resume)
        if checkpoint.get("manifest_fingerprint") != manifest["fingerprint"]:
            raise ValueError("Resume checkpoint was created from a different data manifest")
        checkpoint_world_size = int(checkpoint.get("world_size", 1))
        if checkpoint_world_size != context.world_size:
            raise ValueError(
                f"Cannot exactly resume a world_size={checkpoint_world_size} checkpoint with "
                f"world_size={context.world_size}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint.get("grad_scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint.get("best_val_mae", best_val))

    train_model = wrap_ddp(model, context)
    if resume is not None:
        restore_distributed_rng_state(checkpoint, context)

    output_dir = config.path("paths.autoencoder_output_dir")
    if context.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()
    metrics_path = output_dir / "metrics.jsonl"
    kl_weight = float(config.get("autoencoder.kl_weight", 1e-6))
    clip_norm = float(config.get("training.gradient_clip_norm", 1.0))
    log_every = max(
        1,
        int(config.get("autoencoder.log_every_steps", config.get("training.log_every_steps", 20))),
    )
    max_val_batches = config.get("autoencoder.max_val_batches")
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    started = time.time()

    if context.is_main_process:
        print(
            json.dumps(
                {
                    "distributed": context.distributed,
                    "world_size": context.world_size,
                    "per_device_batch": stage_batch_size,
                    "gradient_accumulation": accumulation,
                    "effective_batch": stage_batch_size * accumulation * context.world_size,
                    "max_epochs": epochs,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "planned_optimizer_steps": total_steps,
                    "log_every_steps": log_every,
                }
            )
        )
    training_progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training autoencoder",
        unit="step",
        disable=not context.is_main_process,
    )
    running = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "perceptual": 0.0}
    running_batches = 0
    for epoch in range(start_epoch, epochs):
        train_model.train()
        loader = create_loader(
            train_dataset,
            config,
            training=True,
            epoch=epoch,
            distributed=context,
            batch_size=stage_batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "perceptual": 0.0}
        batch_count = 0
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            with ddp_sync_context((train_model,), should_step):
                with autocast_context(device, dtype, amp_enabled):
                    reconstruction, mu, sigma = train_model(images)
                    l1 = F.l1_loss(reconstruction.float(), images.float())
                    kl = _kl_loss(mu, sigma)
                    p_loss = (
                        perceptual_loss(reconstruction.float(), images.float())
                        if perceptual_loss is not None
                        else torch.zeros((), device=device)
                    )
                    loss = l1 + kl_weight * kl + perceptual_weight * p_loss
                if not all_ranks_finite(loss, context):
                    raise FloatingPointError(
                        f"Non-finite Autoencoder loss at epoch {epoch}, batch {batch_index}: {loss.item()}"
                    )
                scaler.scale(loss / accumulation).backward()
            batch_count += 1
            totals["loss"] += loss.item()
            totals["l1"] += l1.item()
            totals["kl"] += kl.item()
            totals["perceptual"] += p_loss.item()
            running["loss"] += loss.item()
            running["l1"] += l1.item()
            running["kl"] += kl.item()
            running["perceptual"] += p_loss.item()
            running_batches += 1
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                if not all_ranks_finite(gradient_norm, context):
                    raise FloatingPointError(f"Non-finite Autoencoder gradient norm at epoch {epoch}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1
                training_progress.update(1)
                training_progress.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch + 1)
                final_update = epoch + 1 >= epochs and batch_index + 1 >= len(loader)
                if global_step % log_every == 0 or final_update:
                    log_statistics = torch.tensor(
                        [
                            running["loss"],
                            running["l1"],
                            running["kl"],
                            running["perceptual"],
                            running_batches,
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    all_reduce_sum(log_statistics, context)
                    log_loss, log_l1, log_kl, log_perceptual, log_batches = log_statistics.tolist()
                    log_row = {
                        "stage": "autoencoder",
                        "event": "train",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": log_loss / max(1, log_batches),
                        "train_l1": log_l1 / max(1, log_batches),
                        "train_kl": log_kl / max(1, log_batches),
                        "train_perceptual": log_perceptual / max(1, log_batches),
                        "gradient_norm": float(gradient_norm),
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": time.time() - started,
                    }
                    if context.is_main_process:
                        append_jsonl(metrics_path, log_row)
                        print(json.dumps(log_row, ensure_ascii=False))
                    running = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "perceptual": 0.0}
                    running_batches = 0

        validation = validate_autoencoder(
            model,
            val_loader,
            device,
            dtype,
            amp_enabled,
            max_batches=max_val_batches,
            distributed=context,
        )
        train_statistics = torch.tensor(
            [totals["loss"], totals["l1"], totals["kl"], totals["perceptual"], batch_count],
            device=device,
            dtype=torch.float64,
        )
        all_reduce_sum(train_statistics, context)
        train_loss, train_l1, train_kl, train_perceptual, global_batch_count = train_statistics.tolist()
        row = {
            "stage": "autoencoder",
            "event": "validation",
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(1, global_batch_count),
            "train_l1": train_l1 / max(1, global_batch_count),
            "train_kl": train_kl / max(1, global_batch_count),
            "train_perceptual": train_perceptual / max(1, global_batch_count),
            "val_mae": validation["mae"],
            "val_mse": validation["mse"],
            "val_psnr": validation["psnr"],
            "elapsed_seconds": time.time() - started,
        }
        if context.is_main_process:
            append_jsonl(metrics_path, row)
            print(json.dumps(row, ensure_ascii=False))
        training_progress.set_postfix(
            train=f"{row['train_loss']:.4f}", val=f"{validation['mae']:.4f}", epoch=epoch + 1
        )
        is_best = validation["mae"] < best_val
        best_val = min(best_val, validation["mae"])
        rng_states = gather_rng_states(context)
        if context.is_main_process:
            payload = _autoencoder_payload(
                model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch + 1,
                global_step,
                best_val,
                config,
                manifest,
                rng_states,
                context.world_size,
            )
            atomic_torch_save(payload, output_dir / "last.pt")
            if is_best:
                atomic_torch_save(payload, output_dir / "best.pt")
        context.barrier()
    training_progress.close()
    return output_dir / "best.pt"


def _tensor_to_gray_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


@torch.no_grad()
def evaluate_autoencoder(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    max_batches: int | None = None,
    max_panels: int = 16,
) -> dict[str, float]:
    seed_everything(int(config.get("training.seed", 2026)))
    device = select_device(str(config.get("training.device", "auto")))
    dtype, amp_enabled = resolve_precision(device, str(config.get("training.precision", "auto")))
    manifest = load_manifest(config)
    _, val_dataset = create_datasets(config, manifest)
    val_loader = create_loader(val_dataset, config, training=False)
    model = _load_autoencoder(Path(checkpoint_path), False, device).eval()
    metrics = validate_autoencoder(model, val_loader, device, dtype, amp_enabled, max_batches=max_batches)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    panel_count = 0
    panel_batches = math.ceil(max_panels / max(1, val_loader.batch_size or 1))
    panel_total = min(len(val_loader), panel_batches)
    panel_progress = tqdm(
        islice(val_loader, panel_total),
        total=panel_total,
        desc="Saving reconstruction panels",
        unit="batch",
        leave=False,
    )
    for batch in panel_progress:
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, dtype, amp_enabled):
            reconstructions = model.reconstruct(images)
        for index in range(images.shape[0]):
            original = images[index, 0]
            reconstruction = reconstructions[index, 0]
            error = (original - reconstruction).abs()
            tiles = [_tensor_to_gray_image(item) for item in (original, reconstruction, error)]
            panel = Image.new("L", (tiles[0].width * 3, tiles[0].height))
            for tile_index, tile in enumerate(tiles):
                panel.paste(tile, (tile_index * tile.width, 0))
            patient_id = int(batch["patient_id"][index])
            view_code = str(batch["view_code"][index])
            panel.save(destination / f"{panel_count:04d}_eid-{patient_id}_view-{view_code}.png")
            panel_count += 1
            if panel_count >= max_panels:
                break
        if panel_count >= max_panels:
            break
    report = {**metrics, "checkpoint": str(Path(checkpoint_path).resolve()), "panels": panel_count}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return metrics
