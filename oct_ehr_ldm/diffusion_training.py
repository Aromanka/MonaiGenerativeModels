from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from .config import ProjectConfig
from .data import (
    UNKNOWN_VIEW,
    compute_train_ehr_stats,
    create_datasets,
    create_loader,
    load_ehr_dictionary,
    load_manifest,
    safe_torch_load,
)
from .models import (
    EHRConditionProjector,
    build_cxr_autoencoder,
    build_cxr_diffusion,
    build_cxr_scheduler,
    configure_diffusion_phase,
    count_parameters,
    load_module_checkpoint,
)
from .runtime import (
    ModuleEMA,
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    cosine_warmup_lambda,
    make_grad_scaler,
    resolve_precision,
    restore_rng_state,
    seed_everything,
    select_device,
)


def _projector_architecture(config: ProjectConfig, manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "ehr_dim": int(manifest["ehr_dim"]),
        "cross_attention_dim": 1024,
        "num_tokens": int(config.get("condition.num_tokens", 8)),
        "hidden_dim": int(config.get("condition.hidden_dim", 2048)),
        "num_views": max(int(value) for value in manifest["view_to_id"].values()) + 1,
        "view_embedding_dim": int(config.get("condition.view_embedding_dim", 32)),
    }


def _load_autoencoder(config: ProjectConfig, device: torch.device) -> torch.nn.Module:
    autoencoder = build_cxr_autoencoder(use_checkpointing=False)
    load_module_checkpoint(
        autoencoder,
        config.path("paths.oct_autoencoder_checkpoint"),
        preferred_keys=("model", "autoencoder"),
        strict=True,
    )
    autoencoder.to(device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    return autoencoder


@torch.no_grad()
def compute_latent_scale_factor(
    autoencoder: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    amp_enabled: bool,
    max_batches: int = 50,
) -> float:
    value_sum = 0.0
    square_sum = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, dtype, amp_enabled):
            latent = autoencoder.encode_stage_2_inputs(images)
        latent = latent.float()
        value_sum += latent.sum().item()
        square_sum += latent.pow(2).sum().item()
        count += latent.numel()
    if count < 2:
        raise ValueError("Not enough latent values to estimate a scale factor")
    variance = max(square_sum / count - (value_sum / count) ** 2, 1e-12)
    return 1.0 / math.sqrt(variance)


def _assert_manifest(checkpoint: dict[str, Any], manifest: dict[str, Any]) -> None:
    checkpoint_fingerprint = checkpoint.get("manifest_fingerprint")
    if checkpoint_fingerprint is not None and checkpoint_fingerprint != manifest["fingerprint"]:
        raise ValueError("Checkpoint was created from a different patient split/data manifest")


def _load_initial_diffusion(
    config: ProjectConfig,
    manifest: dict[str, Any],
    device: torch.device,
    init_checkpoint: str | Path | dict[str, Any] | None,
    apply_ema: bool = True,
) -> tuple[torch.nn.Module, EHRConditionProjector, dict[str, Any] | None]:
    diffusion = build_cxr_diffusion().to(device)
    projector = EHRConditionProjector(**_projector_architecture(config, manifest)).to(device)
    payload = None
    if init_checkpoint is None:
        load_module_checkpoint(
            diffusion,
            config.path("paths.cxr_diffusion_checkpoint"),
            preferred_keys=("diffusion", "model"),
            strict=True,
        )
    else:
        payload = init_checkpoint if isinstance(init_checkpoint, dict) else safe_torch_load(init_checkpoint)
        _assert_manifest(payload, manifest)
        diffusion.load_state_dict(payload["diffusion"], strict=True)
        projector.load_state_dict(payload["projector"], strict=True)
        if apply_ema and bool(config.get("diffusion.use_ema_for_phase_init", True)) and payload.get("ema"):
            ema = ModuleEMA({"diffusion": diffusion, "projector": projector})
            ema.load_state_dict(payload["ema"])
            ema.copy_to({"diffusion": diffusion, "projector": projector})
    return diffusion, projector, payload


def _diffusion_loss(
    batch: dict[str, Any],
    autoencoder: torch.nn.Module,
    diffusion: torch.nn.Module,
    projector: EHRConditionProjector,
    noise_scheduler: Any,
    scale_factor: float,
    device: torch.device,
    condition_dropout: float,
) -> torch.Tensor:
    images = batch["image"].to(device, non_blocking=True)
    ehr = batch["ehr"].to(device, non_blocking=True)
    view_ids = batch["view_id"].to(device, non_blocking=True)
    with torch.no_grad():
        latent = autoencoder.encode_stage_2_inputs(images) * scale_factor
    noise = torch.randn_like(latent)
    timesteps = torch.randint(
        0, noise_scheduler.num_train_timesteps, (latent.shape[0],), device=device, dtype=torch.long
    )
    noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
    target = noise_scheduler.get_velocity(latent, noise, timesteps)
    context = projector(ehr, view_ids, condition_dropout=condition_dropout)
    prediction = diffusion(noisy_latent, timesteps=timesteps, context=context)
    return F.mse_loss(prediction.float(), target.float())


@torch.no_grad()
def validate_diffusion(
    autoencoder: torch.nn.Module,
    diffusion: torch.nn.Module,
    projector: EHRConditionProjector,
    noise_scheduler: Any,
    loader: torch.utils.data.DataLoader,
    scale_factor: float,
    device: torch.device,
    dtype: torch.dtype | None,
    amp_enabled: bool,
    max_batches: int = 20,
    compare_conditions: bool = False,
) -> dict[str, float]:
    autoencoder.eval()
    diffusion.eval()
    projector.eval()
    totals = {"correct": 0.0, "shuffled": 0.0, "null": 0.0}
    counts = {key: 0 for key in totals}
    validation_patient_ids = sorted(
        {int(record["patient_id"]) for record in getattr(loader.dataset, "records", [])}
    )
    replacement_patient = {
        patient_id: validation_patient_ids[(index + 1) % len(validation_patient_ids)]
        for index, patient_id in enumerate(validation_patient_ids)
    }
    # Fixed RNG makes validation comparable and does not consume the training stream.
    rng_state = capture_rng_state()
    torch.manual_seed(314159)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(314159)
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            ehr = batch["ehr"].to(device, non_blocking=True)
            view_ids = batch["view_id"].to(device, non_blocking=True)
            with autocast_context(device, dtype, amp_enabled):
                latent = autoencoder.encode_stage_2_inputs(images) * scale_factor
                noise = torch.randn_like(latent)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (latent.shape[0],), device=device, dtype=torch.long
                )
                noisy = noise_scheduler.add_noise(latent, noise, timesteps)
                target = noise_scheduler.get_velocity(latent, noise, timesteps)
                correct_context = projector.conditional_context(ehr, view_ids)
                correct_prediction = diffusion(noisy, timesteps=timesteps, context=correct_context)
                totals["correct"] += F.mse_loss(correct_prediction.float(), target.float()).item()
                counts["correct"] += 1
                if compare_conditions:
                    if len(validation_patient_ids) > 1:
                        shuffled_ehr = torch.stack(
                            [
                                loader.dataset.ehr_by_patient[replacement_patient[int(patient_id)]]
                                for patient_id in batch["patient_id"]
                            ],
                            dim=0,
                        ).to(device, non_blocking=True)
                        shuffled_context = projector.conditional_context(shuffled_ehr, view_ids)
                        shuffled_prediction = diffusion(noisy, timesteps=timesteps, context=shuffled_context)
                        totals["shuffled"] += F.mse_loss(shuffled_prediction.float(), target.float()).item()
                        counts["shuffled"] += 1
                    null_prediction = diffusion(
                        noisy, timesteps=timesteps, context=projector.unconditional_context(ehr.shape[0])
                    )
                    totals["null"] += F.mse_loss(null_prediction.float(), target.float()).item()
                    counts["null"] += 1
    finally:
        restore_rng_state(rng_state)
    if counts["correct"] == 0:
        raise ValueError("Validation loader produced no batches")
    results = {"correct_loss": totals["correct"] / counts["correct"]}
    if counts["shuffled"]:
        results["shuffled_loss"] = totals["shuffled"] / counts["shuffled"]
        results["condition_gap"] = results["shuffled_loss"] - results["correct_loss"]
    if counts["null"]:
        results["null_loss"] = totals["null"] / counts["null"]
    return results


def _diffusion_payload(
    diffusion: torch.nn.Module,
    projector: EHRConditionProjector,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: ModuleEMA | None,
    phase: str,
    epoch: int,
    next_batch: int,
    global_step: int,
    best_val: float,
    scale_factor: float,
    config: ProjectConfig,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "ehr_conditioned_oct_ldm",
        "phase": phase,
        "diffusion": diffusion.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_val_loss": best_val,
        "scale_factor": scale_factor,
        "architecture": _projector_architecture(config, manifest),
        "image_size": int(config.get("data.image_size", 512)),
        "view_to_id": manifest["view_to_id"],
        "manifest_fingerprint": manifest["fingerprint"],
        "rng_state": capture_rng_state(),
        "config": config.data,
    }


def train_diffusion(
    config: ProjectConfig,
    phase: str,
    init_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    if init_checkpoint is not None and resume is not None:
        raise ValueError("Use either init_checkpoint (new phase) or resume (same phase), not both")
    seed_everything(int(config.get("training.seed", 2026)))
    device = select_device(str(config.get("training.device", "auto")))
    dtype, amp_enabled = resolve_precision(device, str(config.get("training.precision", "auto")))
    manifest = load_manifest(config)
    train_dataset, val_dataset = create_datasets(config, manifest)
    val_loader = create_loader(val_dataset, config, training=False)
    autoencoder = _load_autoencoder(config, device)
    noise_scheduler = build_cxr_scheduler()

    resume_payload = safe_torch_load(resume) if resume is not None else None
    effective_init = resume_payload if resume_payload is not None else init_checkpoint
    diffusion, projector, init_payload = _load_initial_diffusion(
        config, manifest, device, effective_init, apply_ema=resume_payload is None
    )
    if resume_payload is not None and resume_payload.get("phase") != phase:
        raise ValueError(f"Cannot resume phase {phase!r} from {resume_payload.get('phase')!r} checkpoint")

    if init_payload is None:
        mean, std = compute_train_ehr_stats(train_dataset, manifest["train_patient_ids"])
        projector.set_normalization(mean.to(device), std.to(device))

    trainable_names = configure_diffusion_phase(diffusion, phase)
    for parameter in projector.parameters():
        parameter.requires_grad = True
    print(
        json.dumps(
            {
                "phase": phase,
                "diffusion_trainable_parameters": count_parameters(diffusion, trainable_only=True),
                "diffusion_total_parameters": count_parameters(diffusion),
                "projector_parameters": count_parameters(projector),
                "cross_attention_tensors": len(trainable_names) if phase == "alignment" else None,
            }
        )
    )

    phase_config = f"diffusion.{phase}"
    unet_lr = float(config.get(f"{phase_config}.unet_learning_rate", 1e-5))
    projector_lr = float(config.get(f"{phase_config}.projector_learning_rate", 1e-4))
    parameter_groups = [
        {"params": [p for p in diffusion.parameters() if p.requires_grad], "lr": unet_lr, "name": "diffusion"},
        {"params": list(projector.parameters()), "lr": projector_lr, "name": "projector"},
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(config.get(f"{phase_config}.weight_decay", 0.01)),
    )
    max_steps = int(config.get(f"{phase_config}.max_steps", 10000 if phase == "alignment" else 100000))
    warmup_steps = int(max_steps * float(config.get("training.warmup_fraction", 0.03)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_lambda(step, max_steps, warmup_steps)
    )
    scaler = make_grad_scaler(dtype)
    ema = (
        ModuleEMA(
            {"diffusion": diffusion, "projector": projector},
            decay=float(config.get("diffusion.ema_decay", 0.9999)),
        )
        if bool(config.get("diffusion.use_ema", True))
        else None
    )

    configured_scale = config.get("diffusion.scale_factor")
    if resume_payload is not None:
        scale_factor = float(resume_payload["scale_factor"])
    elif init_payload is not None and "scale_factor" in init_payload:
        scale_factor = float(init_payload["scale_factor"])
    elif configured_scale is not None:
        scale_factor = float(configured_scale)
    else:
        saved_augmentation = train_dataset.intensity_augmentation
        train_dataset.intensity_augmentation = 0.0
        try:
            train_loader_for_scale = create_loader(train_dataset, config, training=False)
            scale_factor = compute_latent_scale_factor(
                autoencoder,
                train_loader_for_scale,
                device,
                dtype,
                amp_enabled,
                max_batches=int(config.get("diffusion.scale_estimation_batches", 50)),
            )
        finally:
            train_dataset.intensity_augmentation = saved_augmentation
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(f"Invalid latent scale factor: {scale_factor}")
    print(f"OCT latent scale factor: {scale_factor:.8f}")

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_val = float("inf")
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        lr_scheduler.load_state_dict(resume_payload["lr_scheduler"])
        scaler.load_state_dict(resume_payload.get("grad_scaler", {}))
        if ema is not None and resume_payload.get("ema") is not None:
            ema.load_state_dict(resume_payload["ema"])
        start_epoch = int(resume_payload["epoch"])
        start_batch = int(resume_payload.get("next_batch", 0))
        global_step = int(resume_payload["global_step"])
        best_val = float(resume_payload.get("best_val_loss", best_val))
        restore_rng_state(resume_payload.get("rng_state"))

    output_dir = config.path(f"paths.{phase}_output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    accumulation = max(1, int(config.get("training.gradient_accumulation_steps", 1)))
    clip_norm = float(config.get("training.gradient_clip_norm", 1.0))
    condition_dropout = float(config.get("condition.dropout_probability", 0.1))
    if not 0 <= condition_dropout < 1:
        raise ValueError("condition.dropout_probability must be in [0, 1)")
    validate_every = max(1, int(config.get(f"{phase_config}.validate_every_steps", 1000)))
    save_every = max(1, int(config.get(f"{phase_config}.save_every_steps", 1000)))
    log_every = max(1, int(config.get("training.log_every_steps", 20)))
    max_val_batches = int(config.get("diffusion.max_val_batches", 20))
    started = time.time()
    epoch = start_epoch
    running_loss = 0.0
    running_batches = 0

    while global_step < max_steps:
        loader = create_loader(train_dataset, config, training=True, epoch=epoch)
        diffusion.train()
        projector.train()
        autoencoder.eval()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            with autocast_context(device, dtype, amp_enabled):
                loss = _diffusion_loss(
                    batch,
                    autoencoder,
                    diffusion,
                    projector,
                    noise_scheduler,
                    scale_factor,
                    device,
                    condition_dropout,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite diffusion loss at phase {phase}, epoch {epoch}, batch {batch_index}: {loss.item()}"
                )
            scaler.scale(loss / accumulation).backward()
            running_loss += loss.item()
            running_batches += 1
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            trainable_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at phase {phase}, epoch {epoch}, batch {batch_index}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update({"diffusion": diffusion, "projector": projector})

            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= len(loader):
                next_epoch, next_batch = epoch + 1, 0
            if global_step % log_every == 0:
                row = {
                    "stage": "diffusion",
                    "phase": phase,
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": running_loss / max(1, running_batches),
                    "gradient_norm": float(gradient_norm),
                    "diffusion_lr": optimizer.param_groups[0]["lr"],
                    "projector_lr": optimizer.param_groups[1]["lr"],
                    "elapsed_seconds": time.time() - started,
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row))
                running_loss = 0.0
                running_batches = 0

            validation: dict[str, float] | None = None
            if global_step % validate_every == 0 or global_step == max_steps:
                validation = validate_diffusion(
                    autoencoder,
                    diffusion,
                    projector,
                    noise_scheduler,
                    val_loader,
                    scale_factor,
                    device,
                    dtype,
                    amp_enabled,
                    max_batches=max_val_batches,
                    compare_conditions=False,
                )
                validation_row = {"stage": "validation", "phase": phase, "global_step": global_step, **validation}
                append_jsonl(metrics_path, validation_row)
                print(json.dumps(validation_row))
                is_best = validation["correct_loss"] < best_val
                best_val = min(best_val, validation["correct_loss"])
            else:
                is_best = False

            if global_step % save_every == 0 or validation is not None or global_step == max_steps:
                payload = _diffusion_payload(
                    diffusion,
                    projector,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    ema,
                    phase,
                    next_epoch,
                    next_batch,
                    global_step,
                    best_val,
                    scale_factor,
                    config,
                    manifest,
                )
                atomic_torch_save(payload, output_dir / "last.pt")
                if is_best:
                    atomic_torch_save(payload, output_dir / "best.pt")
            if global_step >= max_steps:
                break
        epoch += 1
        start_batch = 0
    return output_dir / "best.pt"


def evaluate_diffusion(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    max_batches: int = 20,
    use_ema: bool = True,
) -> dict[str, float]:
    seed_everything(int(config.get("training.seed", 2026)))
    device = select_device(str(config.get("training.device", "auto")))
    dtype, amp_enabled = resolve_precision(device, str(config.get("training.precision", "auto")))
    manifest = load_manifest(config)
    _, val_dataset = create_datasets(config, manifest)
    val_loader = create_loader(val_dataset, config, training=False)
    autoencoder = _load_autoencoder(config, device)
    checkpoint = safe_torch_load(checkpoint_path)
    _assert_manifest(checkpoint, manifest)
    diffusion = build_cxr_diffusion().to(device)
    projector = EHRConditionProjector(**checkpoint["architecture"]).to(device)
    diffusion.load_state_dict(checkpoint["diffusion"], strict=True)
    projector.load_state_dict(checkpoint["projector"], strict=True)
    if use_ema and checkpoint.get("ema"):
        ema = ModuleEMA({"diffusion": diffusion, "projector": projector})
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to({"diffusion": diffusion, "projector": projector})
    return validate_diffusion(
        autoencoder,
        diffusion,
        projector,
        build_cxr_scheduler(),
        val_loader,
        float(checkpoint["scale_factor"]),
        device,
        dtype,
        amp_enabled,
        max_batches=max_batches,
        compare_conditions=True,
    )


def _save_grayscale(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.detach().float().clamp(0, 1).cpu() * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


@torch.no_grad()
def sample_conditioned_oct(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    ehr_path: str | Path,
    patient_ids: Iterable[int],
    output_dir: str | Path,
    view_codes: list[str] | None = None,
    samples_per_view: int = 1,
    guidance_scale: float = 4.0,
    inference_steps: int = 50,
    seed: int = 2026,
    use_ema: bool = True,
) -> list[Path]:
    seed_everything(seed)
    device = select_device(str(config.get("training.device", "auto")))
    dtype, amp_enabled = resolve_precision(device, str(config.get("training.precision", "auto")))
    checkpoint = safe_torch_load(checkpoint_path)
    diffusion = build_cxr_diffusion().to(device).eval()
    projector = EHRConditionProjector(**checkpoint["architecture"]).to(device).eval()
    diffusion.load_state_dict(checkpoint["diffusion"], strict=True)
    projector.load_state_dict(checkpoint["projector"], strict=True)
    if use_ema and checkpoint.get("ema"):
        ema = ModuleEMA({"diffusion": diffusion, "projector": projector})
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to({"diffusion": diffusion, "projector": projector})
    autoencoder = _load_autoencoder(config, device)
    ehr_by_patient = load_ehr_dictionary(ehr_path)
    selected_ids = [int(patient_id) for patient_id in patient_ids]
    absent = [patient_id for patient_id in selected_ids if patient_id not in ehr_by_patient]
    if absent:
        raise KeyError(f"Requested patient IDs absent from {ehr_path}: {absent[:20]}")

    view_to_id = {str(key): int(value) for key, value in checkpoint["view_to_id"].items()}
    if view_codes is None:
        view_codes = [code for code in view_to_id if code != UNKNOWN_VIEW] or [UNKNOWN_VIEW]
    invalid_views = [code for code in view_codes if code not in view_to_id]
    if invalid_views:
        raise KeyError(f"Unknown view codes {invalid_views}; trained mapping is {view_to_id}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scheduler = build_cxr_scheduler()
    scheduler.set_timesteps(num_inference_steps=inference_steps, device=device)
    image_size = int(checkpoint.get("image_size", 512))
    latent_size = image_size // 8
    scale_factor = float(checkpoint["scale_factor"])
    generator = torch.Generator(device=device).manual_seed(seed)
    saved: list[Path] = []
    metadata: list[dict[str, Any]] = []

    for patient_id in selected_ids:
        ehr = ehr_by_patient[patient_id].unsqueeze(0).to(device)
        for view_code in view_codes:
            view_ids = torch.tensor([view_to_id[view_code]], device=device, dtype=torch.long)
            with autocast_context(device, dtype, amp_enabled):
                conditional = projector.conditional_context(ehr, view_ids)
                unconditional = projector.unconditional_context(1)
            context = torch.cat([unconditional, conditional], dim=0)
            for sample_index in range(samples_per_view):
                noise = torch.randn((1, 3, latent_size, latent_size), generator=generator, device=device)
                for timestep in scheduler.timesteps:
                    timestep_value = int(timestep.item())
                    model_input = torch.cat([noise, noise], dim=0)
                    timesteps = torch.full((2,), timestep_value, device=device, dtype=torch.long)
                    with autocast_context(device, dtype, amp_enabled):
                        prediction = diffusion(model_input, timesteps=timesteps, context=context)
                    unconditional_prediction, conditional_prediction = prediction.chunk(2)
                    guided = unconditional_prediction + guidance_scale * (
                        conditional_prediction - unconditional_prediction
                    )
                    noise, _ = scheduler.step(guided, timestep_value, noise)
                with autocast_context(device, dtype, amp_enabled):
                    image = autoencoder.decode_stage_2_outputs(noise / scale_factor)
                output_path = destination / f"eid-{patient_id}_view-{view_code}_sample-{sample_index:03d}.png"
                _save_grayscale(image[0, 0], output_path)
                saved.append(output_path)
                metadata.append(
                    {
                        "patient_id": patient_id,
                        "view_code": view_code,
                        "sample_index": sample_index,
                        "path": str(output_path),
                        "guidance_scale": guidance_scale,
                        "inference_steps": inference_steps,
                        "seed": seed,
                    }
                )
    (destination / "samples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return saved
