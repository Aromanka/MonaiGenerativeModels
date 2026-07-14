from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oct_ehr_ldm",
        description="Fine-tune the MONAI CXR LDM into an EHR-conditioned 2D OCT generator.",
    )
    parser.add_argument("--config", default="configs/oct_ehr_ldm.json", help="Project JSON configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data", help="Validate paired data and write a patient-level split manifest.")
    subparsers.add_parser("inspect-data", help="Print the current split/data summary without loading images.")

    autoencoder = subparsers.add_parser("train-autoencoder", help="Adapt the CXR AutoencoderKL to OCT.")
    autoencoder.add_argument("--resume", help="Resume from an oct_autoencoder checkpoint.")

    evaluate_autoencoder = subparsers.add_parser(
        "evaluate-autoencoder", help="Measure reconstruction and save original/reconstruction/error panels."
    )
    evaluate_autoencoder.add_argument("--checkpoint", required=True)
    evaluate_autoencoder.add_argument("--output-dir", required=True)
    evaluate_autoencoder.add_argument("--max-batches", type=int)
    evaluate_autoencoder.add_argument("--max-panels", type=int, default=16)

    diffusion = subparsers.add_parser("train-diffusion", help="Run condition alignment or full U-Net fine-tuning.")
    diffusion.add_argument("--phase", choices=("alignment", "full"), required=True)
    diffusion.add_argument("--init-checkpoint", help="Initialize a new phase from an earlier diffusion checkpoint.")
    diffusion.add_argument("--resume", help="Resume the same phase, including optimizer/RNG state.")

    evaluate_diffusion = subparsers.add_parser(
        "evaluate-diffusion", help="Compare correct, shuffled, and null EHR conditioning losses."
    )
    evaluate_diffusion.add_argument("--checkpoint", required=True)
    evaluate_diffusion.add_argument("--max-batches", type=int, default=20)
    evaluate_diffusion.add_argument("--no-ema", action="store_true")

    sample = subparsers.add_parser("sample", help="Generate OCT images from EHR-only or paired patient latents.")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--ehr-file", help="Defaults to paths.ehr_only in the config.")
    sample.add_argument("--patient-id", action="append", type=int, default=[])
    sample.add_argument("--all", action="store_true", help="Generate for every patient in --ehr-file.")
    sample.add_argument("--view-code", action="append", help="Repeat for selected trained view codes.")
    sample.add_argument("--samples-per-view", type=int, default=1)
    sample.add_argument("--guidance-scale", type=float, default=4.0)
    sample.add_argument("--inference-steps", type=int, default=50)
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--no-ema", action="store_true")

    smoke = subparsers.add_parser("smoke-test", help="Strict-load all checkpoints and check one real data batch.")
    smoke.add_argument("--diffusion-checkpoint", help="Optional trained diffusion checkpoint to validate.")
    return parser


def _smoke_test(config, diffusion_checkpoint: str | None) -> dict[str, object]:
    import torch

    from .data import create_datasets, create_loader, load_manifest
    from .models import EHRConditionProjector, build_cxr_autoencoder, build_cxr_diffusion, load_module_checkpoint

    manifest = load_manifest(config)
    train, _ = create_datasets(config, manifest)
    batch = next(iter(create_loader(train, config, training=False)))
    autoencoder = build_cxr_autoencoder()
    load_module_checkpoint(
        autoencoder,
        config.path("paths.cxr_autoencoder_checkpoint"),
        preferred_keys=("model", "autoencoder"),
        strict=True,
    )
    diffusion = build_cxr_diffusion()
    load_module_checkpoint(
        diffusion,
        config.path("paths.cxr_diffusion_checkpoint"),
        preferred_keys=("diffusion", "model"),
        strict=True,
    )
    result: dict[str, object] = {
        "image_shape": list(batch["image"].shape),
        "ehr_shape": list(batch["ehr"].shape),
        "cxr_autoencoder_strict_load": True,
        "cxr_diffusion_strict_load": True,
    }
    if diffusion_checkpoint:
        from .data import safe_torch_load

        checkpoint = safe_torch_load(diffusion_checkpoint)
        diffusion.load_state_dict(checkpoint["diffusion"], strict=True)
        projector = EHRConditionProjector(**checkpoint["architecture"])
        projector.load_state_dict(checkpoint["projector"], strict=True)
        context = projector.conditional_context(batch["ehr"], batch["view_id"])
        result["trained_diffusion_strict_load"] = True
        result["context_shape"] = list(context.shape)
        result["context_finite"] = bool(torch.isfinite(context).all())
    return result


def main(argv: list[str] | None = None) -> None:
    parser = _base_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command in {"prepare-data", "inspect-data"}:
        from .data import load_manifest, prepare_manifest, summarize_manifest

        manifest = prepare_manifest(config) if args.command == "prepare-data" else load_manifest(config, False)
        print(json.dumps(summarize_manifest(manifest), indent=2, ensure_ascii=False))
        return

    if args.command == "train-autoencoder":
        from .autoencoder_training import train_autoencoder

        print(f"Best checkpoint: {train_autoencoder(config, resume=args.resume)}")
        return

    if args.command == "evaluate-autoencoder":
        from .autoencoder_training import evaluate_autoencoder

        metrics = evaluate_autoencoder(
            config,
            args.checkpoint,
            args.output_dir,
            max_batches=args.max_batches,
            max_panels=args.max_panels,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-diffusion":
        from .diffusion_training import train_diffusion

        print(
            f"Best checkpoint: {train_diffusion(config, args.phase, args.init_checkpoint, resume=args.resume)}"
        )
        return

    if args.command == "evaluate-diffusion":
        from .diffusion_training import evaluate_diffusion

        metrics = evaluate_diffusion(
            config, args.checkpoint, max_batches=args.max_batches, use_ema=not args.no_ema
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "sample":
        from .data import load_ehr_dictionary
        from .diffusion_training import sample_conditioned_oct

        ehr_path = Path(args.ehr_file).expanduser().resolve() if args.ehr_file else config.path("paths.ehr_only")
        patient_ids = args.patient_id
        if args.all:
            patient_ids = sorted(load_ehr_dictionary(ehr_path))
        if not patient_ids:
            parser.error("sample requires at least one --patient-id, or the explicit --all flag")
        saved = sample_conditioned_oct(
            config,
            args.checkpoint,
            ehr_path,
            patient_ids,
            args.output_dir,
            view_codes=args.view_code,
            samples_per_view=args.samples_per_view,
            guidance_scale=args.guidance_scale,
            inference_steps=args.inference_steps,
            seed=args.seed,
            use_ema=not args.no_ema,
        )
        print(json.dumps({"saved_images": len(saved), "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
        return

    if args.command == "smoke-test":
        print(json.dumps(_smoke_test(config, args.diffusion_checkpoint), indent=2))
        return
    raise AssertionError(f"Unhandled command: {args.command}")
