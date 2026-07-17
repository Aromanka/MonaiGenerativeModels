from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .common import (
    DEFAULT_SCHEMA_PROJECT_ROOT,
    EXPECTED_EHR_DIM,
    EXPECTED_OCT_DIM,
    LABEL_NAMES,
    atomic_json_dump,
    atomic_torch_save,
    load_trajectories,
    resolve_path,
    safe_torch_load,
    trajectory_index_time,
    trajectory_latent,
    trajectory_length,
)


def _load_schema_api(project_root: Path) -> tuple[Any, Any, Any]:
    schema_path = project_root / "src" / "data" / "schema.py"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema V2 API not found under {project_root}")
    sys.path.insert(0, str(project_root))
    try:
        encoder_base = importlib.import_module("src.models.encoders.base")
        # Loading ``src.data.schema`` normally executes ``src.data.__init__`` first.
        # That initializer imports the entire training data stack and can fail in a
        # lightweight/older Python factory environment for reasons unrelated to the
        # schema. Registering only the parent package lets us execute the requested
        # schema module itself while preserving its relative EncoderOutput import.
        data_package = types.ModuleType("src.data")
        data_package.__path__ = [str(schema_path.parent)]
        data_package.__package__ = "src.data"
        sys.modules["src.data"] = data_package
        spec = importlib.util.spec_from_file_location("src.data.schema", schema_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create an import specification for {schema_path}")
        schema = importlib.util.module_from_spec(spec)
        sys.modules["src.data.schema"] = schema
        spec.loader.exec_module(schema)
    finally:
        sys.path.pop(0)
    return schema.EncodedSample, schema.build_v2_payload, encoder_base.EncoderOutput


def _read_oct_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"OCT manifest is empty: {path}")
    required = {"patient_id", "source_patient_id", "visit_id", "index_time_days", "image_name", "qc_pass"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"OCT manifest is missing columns: {sorted(missing)}")
    failed = [row["image_name"] for row in rows if row["qc_pass"].strip().lower() != "true"]
    if failed:
        raise ValueError(f"Manifest contains images that failed QC: {failed[:20]}")
    return rows


def _feature_lookup(record: dict[str, Any], patient_id: str) -> dict[str, torch.Tensor]:
    names = [str(value) for value in record["image_names"]]
    features = torch.as_tensor(record["features"], dtype=torch.float32).cpu()
    if features.ndim != 2 or features.shape[1] != EXPECTED_OCT_DIM:
        raise ValueError(
            f"OCT features for patient {patient_id} must have shape [N,{EXPECTED_OCT_DIM}], "
            f"received {tuple(features.shape)}"
        )
    if len(names) != features.shape[0] or len(set(names)) != len(names):
        raise ValueError(f"OCT image names are missing, duplicated, or misaligned for patient {patient_id}")
    return {name: features[index] for index, name in enumerate(names)}


def package(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_path(args.output_root)
    output_pt = resolve_path(args.output_pt or output_root / "ukb_synthetic_train.pt")
    if output_pt.exists() and not args.force:
        print(f"Schema V2 file already exists: {output_pt}; use --force to recreate it.")
        return safe_torch_load(output_pt)

    trajectory_path = resolve_path(args.ehr_pickle)
    features_path = resolve_path(args.oct_features or output_root / "OCT_features_synthetic.pt")
    oct_manifest_path = resolve_path(args.oct_manifest or output_root / "oct_manifest.csv")
    patient_map_path = output_root / "patient_map.json"
    schema_root = resolve_path(args.schema_project_root)
    EncodedSample, build_v2_payload, EncoderOutput = _load_schema_api(schema_root)

    trajectories = load_trajectories(trajectory_path)
    patient_map = json.loads(patient_map_path.read_text(encoding="utf-8"))
    oct_payload = safe_torch_load(features_path)
    features_by_eid = oct_payload["features_by_eid"]
    oct_metadata = dict(oct_payload.get("metadata", {}))
    rows = _read_oct_manifest(oct_manifest_path)
    rows_by_visit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_visit[row["visit_id"]].append(row)

    entries_by_patient = {str(entry["patient_id"]): entry for entry in patient_map["entries"]}
    feature_lookup_by_patient = {
        str(patient_id): _feature_lookup(record, str(patient_id))
        for patient_id, record in features_by_eid.items()
    }
    samples: list[Any] = []
    used_images: set[tuple[str, str]] = set()
    for visit_id in sorted(rows_by_visit, key=lambda value: (int(value.split("@", 1)[0]), value)):
        visit_rows = sorted(rows_by_visit[visit_id], key=lambda row: (row["laterality"], row["image_name"]))
        patient_id = visit_rows[0]["patient_id"]
        if any(row["patient_id"] != patient_id for row in visit_rows):
            raise ValueError(f"Visit {visit_id} contains multiple patient IDs")
        if patient_id not in entries_by_patient or patient_id not in feature_lookup_by_patient:
            raise KeyError(f"Missing patient mapping or OCT feature record for patient {patient_id}")
        source_id = int(entries_by_patient[patient_id]["source_patient_id"])
        if any(int(row["source_patient_id"]) != source_id for row in visit_rows):
            raise ValueError(f"source_patient_id mismatch for visit {visit_id}")
        trajectory = trajectories[source_id]
        index_time, index_time_source = trajectory_index_time(trajectory, source_id)
        manifest_index_time = float(visit_rows[0]["index_time_days"])
        if abs(index_time - manifest_index_time) > 1e-4:
            raise ValueError(f"index_time mismatch for visit {visit_id}")
        ehr_latent = trajectory_latent(trajectory, source_id).reshape(1, 1, EXPECTED_EHR_DIM)
        image_names = [row["image_name"] for row in visit_rows]
        try:
            oct_features = torch.stack(
                [feature_lookup_by_patient[patient_id][name] for name in image_names], dim=0
            ).reshape(1, -1, EXPECTED_OCT_DIM)
        except KeyError as error:
            raise KeyError(f"Feature missing for manifest image {error.args[0]!r}, patient {patient_id}") from error
        for image_name in image_names:
            key = (patient_id, image_name)
            if key in used_images:
                raise ValueError(f"Image is assigned to more than one Schema sample: {key}")
            used_images.add(key)
        num_oct_tokens = oct_features.shape[1]

        samples.append(
            EncodedSample(
                patient_id=str(patient_id),
                visit_id=visit_id,
                index_time=float(index_time),
                modalities={
                    "ehr": EncoderOutput(
                        features=ehr_latent,
                        token_mask=torch.ones(1, 1, dtype=torch.bool),
                        timestamps=torch.tensor([[index_time]], dtype=torch.float32),
                        observed_mask=torch.tensor([True]),
                        quality=torch.ones(1, 1),
                        fidelity=torch.ones(1, 1),
                        generated=torch.ones(1, 1, dtype=torch.bool),
                        metadata={
                            "trajectory_length": trajectory_length(trajectory, source_id),
                            "cutoff_age_days": index_time,
                            "cutoff_age_days_source": index_time_source,
                        },
                        provenance={
                            "source": "synthetic_ehr_trajectory",
                            "encoder": "DelphiSMURF",
                        },
                    ),
                    "oct": EncoderOutput(
                        features=oct_features,
                        token_mask=torch.ones(1, num_oct_tokens, dtype=torch.bool),
                        timestamps=torch.full((1, num_oct_tokens), index_time, dtype=torch.float32),
                        observed_mask=torch.tensor([True]),
                        quality=torch.ones(1, num_oct_tokens),
                        fidelity=torch.ones(1, num_oct_tokens),
                        generated=torch.ones(1, num_oct_tokens, dtype=torch.bool),
                        metadata={
                            "image_names": image_names,
                            "lateralities": [row["laterality"] for row in visit_rows],
                            "synthetic": True,
                        },
                        provenance={
                            "source": "synthetic_oct",
                            "encoder": "RETFound",
                            "generator_checkpoint": oct_metadata["generator_checkpoint"],
                            "encoder_checkpoint": oct_metadata["encoder_checkpoint"],
                        },
                    ),
                },
                target={
                    "case_weight": torch.tensor(1.0, dtype=torch.float32),
                    "tasks": {
                        "diabetes": {
                            "value": torch.zeros(len(LABEL_NAMES), dtype=torch.float32),
                            "mask": torch.zeros(len(LABEL_NAMES), dtype=torch.bool),
                        }
                    },
                },
                metadata={
                    "synthetic": True,
                    "unlabeled": True,
                    "source_patient_id": str(source_id),
                },
            )
        )

    expected_images = sum(len(record["image_names"]) for record in features_by_eid.values())
    if len(used_images) != expected_images:
        unused = expected_images - len(used_images)
        raise ValueError(f"{unused} encoded OCT images are not referenced by oct_manifest.csv")
    payload = build_v2_payload(
        samples,
        meta={
            "synthetic": True,
            "unlabeled": True,
            "dataset_name": args.dataset_name,
            "label_names": LABEL_NAMES,
            "ehr_feature_dim": EXPECTED_EHR_DIM,
            "oct_feature_dim": EXPECTED_OCT_DIM,
            "ehr_trajectory_file": str(trajectory_path),
            "oct_feature_file": str(features_path),
            "oct_manifest": str(oct_manifest_path),
            "patient_id_mode": patient_map["patient_id_mode"],
            "generator_checkpoint": oct_metadata["generator_checkpoint"],
            "encoder_name": oct_metadata["encoder_name"],
            "encoder_checkpoint": oct_metadata["encoder_checkpoint"],
        },
    )
    atomic_torch_save(payload, output_pt)

    dataset_manifest = {
        "schema_version": 2,
        "dataset_name": args.dataset_name,
        "synthetic": True,
        "unlabeled": True,
        "patient_id_mode": patient_map["patient_id_mode"],
        "num_source_patients": len(patient_map["entries"]),
        "num_samples": len(samples),
        "num_images": len(rows),
        "ehr_feature_dim": EXPECTED_EHR_DIM,
        "oct_feature_dim": EXPECTED_OCT_DIM,
        "label_names": LABEL_NAMES,
        "files": {
            "training_input": output_pt.name,
            "oct_features": features_path.name,
            "oct_manifest": oct_manifest_path.name,
            "image_directory": "images",
            "patient_map": patient_map_path.name,
        },
        "sources": {
            "ehr_trajectories": str(trajectory_path),
            "generator_name": "oct_ehr_ldm",
            "generator_checkpoint": oct_metadata["generator_checkpoint"],
            "oct_encoder": oct_metadata["encoder_name"],
            "oct_encoder_checkpoint": oct_metadata["encoder_checkpoint"],
            "schema_api_root": str(schema_root),
        },
        "generation": patient_map["generation"],
    }
    atomic_json_dump(dataset_manifest, output_root / "dataset_manifest.json")
    print(f"Saved Schema V2 payload with {len(samples)} samples to {output_pt}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package synthetic EHR and RETFound OCT features as Schema V2.")
    parser.add_argument("--ehr-pickle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--schema-project-root", default=str(DEFAULT_SCHEMA_PROJECT_ROOT))
    parser.add_argument("--oct-features", help="Defaults to OUTPUT_ROOT/OCT_features_synthetic.pt.")
    parser.add_argument("--oct-manifest", help="Defaults to OUTPUT_ROOT/oct_manifest.csv.")
    parser.add_argument("--output-pt", help="Defaults to OUTPUT_ROOT/ukb_synthetic_train.pt.")
    parser.add_argument("--dataset-name", default="ukbehr_ehr_oct_synthetic_train")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    package(build_parser().parse_args())


if __name__ == "__main__":
    main()
