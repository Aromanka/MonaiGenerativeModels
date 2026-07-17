from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .common import (
    REPO_ROOT,
    atomic_json_dump,
    atomic_torch_save,
    load_trajectories,
    parse_key_value,
    resolve_path,
    trajectory_index_time,
    trajectory_latent,
    trajectory_length,
)


DEFAULT_LATERALITY = {"21017": "left", "21018": "right"}


def _select_source_ids(
    trajectories: Mapping[int, Any], requested: list[int], offset: int, limit: int | None
) -> list[int]:
    if requested:
        missing = sorted(set(requested) - trajectories.keys())
        if missing:
            raise KeyError(f"Requested source patient IDs are absent: {missing[:20]}")
        selected = list(dict.fromkeys(requested))
    else:
        selected = sorted(trajectories)
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    return selected[offset : offset + limit if limit is not None else None]


def _build_patient_map(
    source_ids: list[int], mode: str, sequential_start: int
) -> list[dict[str, int]]:
    if mode == "source":
        output_ids = source_ids
    else:
        if sequential_start < 0:
            raise ValueError("--sequential-start must be non-negative")
        output_ids = list(range(sequential_start, sequential_start + len(source_ids)))
    return [
        {"source_patient_id": source_id, "patient_id": output_id}
        for source_id, output_id in zip(source_ids, output_ids)
    ]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "patient_id",
        "source_patient_id",
        "visit_id",
        "index_time_days",
        "image_name",
        "image_path",
        "laterality",
        "view_code",
        "sample_index",
        "generator_checkpoint",
        "seed",
        "qc_pass",
        "width",
        "height",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-") or "unknown"


def generate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_path(args.output_root)
    manifest_path = output_root / "oct_manifest.csv"
    image_index_path = output_root / "oct_image_index.json"
    patient_map_path = output_root / "patient_map.json"
    if manifest_path.exists() and image_index_path.exists() and patient_map_path.exists() and not args.force:
        print(f"Generation artifacts already exist under {output_root}; use --force to regenerate.")
        return json.loads(patient_map_path.read_text(encoding="utf-8"))

    trajectory_path = resolve_path(args.ehr_pickle)
    config_path = resolve_path(args.config, base=REPO_ROOT)
    checkpoint_path = resolve_path(args.generator_checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    trajectories = load_trajectories(trajectory_path)
    source_ids = _select_source_ids(trajectories, args.patient_id, args.offset, args.limit)
    if not source_ids:
        raise ValueError("Patient selection is empty")
    patient_entries = _build_patient_map(source_ids, args.patient_id_mode, args.sequential_start)

    work_dir = output_root / ".work"
    raw_dir = work_dir / "generated_raw"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ehr_latents: dict[int, Any] = {}
    index_times: dict[int, tuple[float, str]] = {}
    source_by_output: dict[int, int] = {}
    for entry in patient_entries:
        source_id = entry["source_patient_id"]
        output_id = entry["patient_id"]
        trajectory = trajectories[source_id]
        ehr_latents[output_id] = trajectory_latent(trajectory, source_id)
        index_times[output_id] = trajectory_index_time(trajectory, source_id)
        trajectory_length(trajectory, source_id)
        source_by_output[output_id] = source_id
    latent_path = work_dir / "ehr_latents.pt"
    atomic_torch_save(ehr_latents, latent_path)

    view_codes = args.view_code or ["21017", "21018"]
    command = [
        sys.executable,
        "-m",
        "oct_ehr_ldm",
        "--config",
        str(config_path),
        "sample",
        "--checkpoint",
        str(checkpoint_path),
        "--ehr-file",
        str(latent_path),
        "--all",
        "--samples-per-view",
        str(args.samples_per_view),
        "--guidance-scale",
        str(args.guidance_scale),
        "--inference-steps",
        str(args.inference_steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(raw_dir),
    ]
    for view_code in view_codes:
        command.extend(["--view-code", view_code])
    if args.no_ema:
        command.append("--no-ema")
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    samples_path = raw_dir / "samples.json"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Generator did not create {samples_path}")
    raw_records = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise TypeError("Generator samples.json must contain a list")

    laterality_by_view = {**DEFAULT_LATERALITY, **parse_key_value(args.view_laterality, label="--view-laterality")}
    rows: list[dict[str, Any]] = []
    image_index: dict[str, list[str]] = {str(entry["patient_id"]): [] for entry in patient_entries}
    seen_visit_laterality: set[tuple[str, str]] = set()
    for record in raw_records:
        output_id = int(record["patient_id"])
        if output_id not in source_by_output:
            raise ValueError(f"Generator returned unexpected patient ID {output_id}")
        source_id = source_by_output[output_id]
        sample_index = int(record["sample_index"])
        view_code = str(record["view_code"])
        laterality = laterality_by_view.get(view_code, f"view-{_safe_component(view_code)}")
        visit_id = f"{output_id}@synthetic_{sample_index:04d}"
        collision_key = (visit_id, laterality)
        if collision_key in seen_visit_laterality:
            raise ValueError(f"Duplicate laterality {laterality!r} for visit {visit_id}")
        seen_visit_laterality.add(collision_key)
        image_name = f"{output_id}_synthetic_{sample_index:04d}_{_safe_component(laterality)}.png"
        source_path = Path(record["path"])
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = output_root / "images" / str(output_id) / image_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
            qc_pass = True
        except Exception:
            width, height, qc_pass = 0, 0, False
        relative_path = destination.relative_to(output_root).as_posix()
        image_index[str(output_id)].append(str(destination.resolve()))
        index_time, _ = index_times[output_id]
        rows.append(
            {
                "patient_id": str(output_id),
                "source_patient_id": str(source_id),
                "visit_id": visit_id,
                "index_time_days": index_time,
                "image_name": image_name,
                "image_path": relative_path,
                "laterality": laterality,
                "view_code": view_code,
                "sample_index": sample_index,
                "generator_checkpoint": str(checkpoint_path),
                "seed": int(record.get("seed", args.seed)),
                "qc_pass": str(qc_pass).lower(),
                "width": width,
                "height": height,
            }
        )

    expected_images = len(patient_entries) * len(view_codes) * args.samples_per_view
    if len(rows) != expected_images:
        raise ValueError(f"Generator produced {len(rows)} images; expected {expected_images}")
    expected_per_patient = len(view_codes) * args.samples_per_view
    incomplete = {
        patient_id: len(paths)
        for patient_id, paths in image_index.items()
        if len(paths) != expected_per_patient
    }
    if incomplete:
        raise ValueError(
            f"Per-patient generated image count mismatch (expected {expected_per_patient}): {incomplete}"
        )
    if not all(row["qc_pass"] == "true" for row in rows):
        raise ValueError("One or more generated images failed PIL verification")
    rows.sort(key=lambda row: (int(row["patient_id"]), int(row["sample_index"]), row["laterality"]))

    patient_map = {
        "version": 1,
        "patient_id_mode": args.patient_id_mode,
        "sequential_start": args.sequential_start if args.patient_id_mode == "sequential" else None,
        "ehr_pickle": str(trajectory_path),
        "entries": patient_entries,
        "generation": {
            "config": str(config_path),
            "generator_checkpoint": str(checkpoint_path),
            "view_codes": view_codes,
            "view_laterality": laterality_by_view,
            "samples_per_view": args.samples_per_view,
            "guidance_scale": args.guidance_scale,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
            "use_ema": not args.no_ema,
        },
        "index_time_source_by_patient": {
            str(patient_id): source for patient_id, (_, source) in index_times.items()
        },
    }
    _write_csv(rows, manifest_path)
    atomic_json_dump(image_index, image_index_path)
    atomic_json_dump(patient_map, patient_map_path)
    print(f"Generated and organized {len(rows)} images for {len(patient_entries)} patients in {output_root}")
    return patient_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate auditable synthetic OCT images from EHR trajectories.")
    parser.add_argument("--ehr-pickle", required=True, help="Pickle containing payload['trajectories'].")
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/oct_ehr_ldm.json")
    parser.add_argument("--patient-id", action="append", type=int, default=[], help="Repeat to select source EIDs.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patient-id-mode", choices=("source", "sequential"), default="source")
    parser.add_argument("--sequential-start", type=int, default=1)
    parser.add_argument("--view-code", action="append", default=[])
    parser.add_argument(
        "--view-laterality",
        action="append",
        default=[],
        metavar="VIEW=LATERALITY",
        help="Override the default 21017=left, 21018=right mapping.",
    )
    parser.add_argument("--samples-per-view", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_per_view <= 0:
        raise ValueError("--samples-per-view must be positive")
    generate(args)


if __name__ == "__main__":
    main()
