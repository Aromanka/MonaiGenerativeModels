from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import torch

from .common import EXPECTED_OCT_DIM, REPO_ROOT, atomic_torch_save, resolve_path, safe_torch_load


def _restore_image_extensions(
    features_by_eid: dict[str, dict[str, Any]], image_index: dict[str, list[str]]
) -> None:
    for eid, record in features_by_eid.items():
        names_by_stem: dict[str, deque[str]] = defaultdict(deque)
        for path in image_index.get(str(eid), []):
            image_path = Path(path)
            names_by_stem[image_path.stem].append(image_path.name)
        restored: list[str] = []
        for encoded_name in record["image_names"]:
            choices = names_by_stem.get(str(encoded_name))
            if not choices:
                raise ValueError(f"Cannot map encoded image name {encoded_name!r} back to patient {eid} index")
            restored.append(choices.popleft())
        record["image_names"] = restored


def encode(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_path(args.output_root)
    input_json = resolve_path(args.input_json or output_root / "oct_image_index.json")
    output_pt = resolve_path(args.output_pt or output_root / "OCT_features_synthetic.pt")
    checkpoint_path = resolve_path(args.retfound_checkpoint)
    patient_map_path = output_root / "patient_map.json"
    if output_pt.exists() and not args.force:
        print(f"Encoded feature file already exists: {output_pt}; use --force to recreate it.")
        return safe_torch_load(output_pt)
    if not patient_map_path.is_file():
        raise FileNotFoundError(f"Missing generation metadata: {patient_map_path}")
    if not input_json.is_file():
        raise FileNotFoundError(input_json)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    work_output = output_root / ".work" / "OCT_features_raw.pt"
    work_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "encoder" / "encode.py"),
        "--input_json",
        str(input_json),
        "--checkpoint_path",
        str(checkpoint_path),
        "--output_pt",
        str(work_output),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--image_size",
        str(args.image_size),
    ]
    if not args.skip_bad_images:
        command.append("--no_skip_bad_images")
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    payload = safe_torch_load(work_output)
    if not isinstance(payload, dict) or not isinstance(payload.get("features_by_eid"), dict):
        raise TypeError("RETFound encoder output has no features_by_eid mapping")
    image_index = json.loads(input_json.read_text(encoding="utf-8"))
    features_by_eid = payload["features_by_eid"]
    _restore_image_extensions(features_by_eid, image_index)

    encoded_images = 0
    empty_patients: list[str] = []
    for eid, record in features_by_eid.items():
        features = torch.as_tensor(record["features"], dtype=torch.float32).cpu().contiguous()
        if features.ndim != 2 or features.shape[1] != EXPECTED_OCT_DIM:
            raise ValueError(
                f"RETFound features for patient {eid} must have shape [N,{EXPECTED_OCT_DIM}], "
                f"received {tuple(features.shape)}"
            )
        if len(record["image_names"]) != features.shape[0]:
            raise ValueError(f"Image/feature count mismatch for patient {eid}")
        if features.shape[0] == 0:
            empty_patients.append(str(eid))
        if not torch.isfinite(features).all():
            raise ValueError(f"RETFound features for patient {eid} contain NaN/Inf")
        record["features"] = features
        encoded_images += features.shape[0]
    if empty_patients:
        raise ValueError(f"Patients without encoded OCT images: {empty_patients[:20]}")

    patient_map = json.loads(patient_map_path.read_text(encoding="utf-8"))
    generation = patient_map["generation"]
    payload["metadata"] = {
        **dict(payload.get("metadata", {})),
        "synthetic": True,
        "feature_dim": EXPECTED_OCT_DIM,
        "encoder_name": "RETFound",
        "encoder_checkpoint": str(checkpoint_path),
        "generator_name": "oct_ehr_ldm",
        "generator_checkpoint": generation["generator_checkpoint"],
        "preprocessing": {
            "rgb": True,
            "resize": [args.image_size, args.image_size],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "image_names_include_extension": True,
        "num_patients": len(features_by_eid),
        "num_images": encoded_images,
    }
    atomic_torch_save(payload, output_pt)
    print(f"Saved {encoded_images} RETFound features for {len(features_by_eid)} patients to {output_pt}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode synthetic OCT images with the project's RETFound implementation."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--retfound-checkpoint", required=True)
    parser.add_argument("--input-json", help="Defaults to OUTPUT_ROOT/oct_image_index.json.")
    parser.add_argument("--output-pt", help="Defaults to OUTPUT_ROOT/OCT_features_synthetic.pt.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--skip-bad-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.image_size <= 0:
        raise ValueError("Invalid encoder batch/worker/image-size argument")
    encode(args)


if __name__ == "__main__":
    main()
