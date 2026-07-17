from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from .common import DEFAULT_SCHEMA_PROJECT_ROOT, REPO_ROOT, resolve_path


def _run(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run(args: argparse.Namespace) -> None:
    output_root = resolve_path(args.output_root)
    common_generate = [
        "--ehr-pickle",
        str(resolve_path(args.ehr_pickle)),
        "--generator-checkpoint",
        str(resolve_path(args.generator_checkpoint)),
        "--autoencoder-checkpoint",
        str(resolve_path(args.autoencoder_checkpoint)),
        "--output-root",
        str(output_root),
        "--config",
        str(resolve_path(args.config, base=REPO_ROOT)),
        "--patient-id-mode",
        args.patient_id_mode,
        "--sequential-start",
        str(args.sequential_start),
        "--offset",
        str(args.offset),
        "--samples-per-view",
        str(args.samples_per_view),
        "--guidance-scale",
        str(args.guidance_scale),
        "--inference-steps",
        str(args.inference_steps),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        common_generate.extend(["--limit", str(args.limit)])
    for patient_id in args.patient_id:
        common_generate.extend(["--patient-id", str(patient_id)])
    for view_code in args.view_code:
        common_generate.extend(["--view-code", view_code])
    for mapping in args.view_laterality:
        common_generate.extend(["--view-laterality", mapping])
    if args.no_ema:
        common_generate.append("--no-ema")
    if args.force:
        common_generate.append("--force")

    encode_arguments = [
        "--output-root",
        str(output_root),
        "--retfound-checkpoint",
        str(resolve_path(args.retfound_checkpoint)),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.encoder_device,
        "--image-size",
        str(args.image_size),
    ]
    if args.skip_bad_images:
        encode_arguments.append("--skip-bad-images")
    if args.force:
        encode_arguments.append("--force")

    package_arguments = [
        "--ehr-pickle",
        str(resolve_path(args.ehr_pickle)),
        "--output-root",
        str(output_root),
        "--schema-project-root",
        str(resolve_path(args.schema_project_root)),
        "--dataset-name",
        args.dataset_name,
    ]
    if args.force:
        package_arguments.append("--force")

    stages = ("generate", "encode", "package")
    start = stages.index(args.start_stage)
    if start <= 0:
        _run("factory.generate_oct", common_generate)
    if start <= 1:
        _run("factory.encode_oct", encode_arguments)
    if start <= 2:
        _run("factory.build_schema_v2", package_arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EHR-conditioned OCT generation, RETFound encoding, and Schema V2 packaging."
    )
    parser.add_argument("--ehr-pickle", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument("--retfound-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/oct_ehr_ldm.json")
    parser.add_argument("--schema-project-root", default=str(DEFAULT_SCHEMA_PROJECT_ROOT))
    parser.add_argument("--dataset-name", default="ukbehr_ehr_oct_synthetic_train")
    parser.add_argument("--start-stage", choices=("generate", "encode", "package"), default="generate")
    parser.add_argument("--patient-id", action="append", type=int, default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patient-id-mode", choices=("source", "sequential"), default="source")
    parser.add_argument("--sequential-start", type=int, default=1)
    parser.add_argument("--view-code", action="append", default=[])
    parser.add_argument("--view-laterality", action="append", default=[], metavar="VIEW=LATERALITY")
    parser.add_argument("--samples-per-view", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--skip-bad-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
