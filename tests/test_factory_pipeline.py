from __future__ import annotations

import argparse
import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from factory.build_schema_v2 import package
from factory.common import DEFAULT_SCHEMA_PROJECT_ROOT, load_trajectories, trajectory_index_time, trajectory_latent
from factory.generate_oct import _build_patient_map, generate


class FactoryCommonTests(unittest.TestCase):
    def test_year_cutoff_and_singleton_latent_are_normalized(self) -> None:
        trajectory = {
            "tokens": [1, 2],
            "ages_days": [100.0, 200.0],
            "cutoff_age_years": 50,
            "ehr_latent": torch.ones(1, 120),
        }
        index_time, source = trajectory_index_time(trajectory, 123)
        self.assertEqual(index_time, 50 * 365.25)
        self.assertEqual(source, "cutoff_age_years_x_365.25")
        self.assertEqual(tuple(trajectory_latent(trajectory, 123).shape), (120,))

    def test_sequential_ids_are_disjoint_when_starts_are_disjoint(self) -> None:
        first = _build_patient_map([100, 200], "sequential", 1)
        second = _build_patient_map([300, 400], "sequential", 1001)
        self.assertFalse(
            {entry["patient_id"] for entry in first} & {entry["patient_id"] for entry in second}
        )

    def test_generation_organizes_views_into_synthetic_visits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trajectory_path = root / "trajectories.pkl"
            with trajectory_path.open("wb") as stream:
                pickle.dump(
                    {
                        "trajectories": {
                            1000010: {
                                "tokens": [1],
                                "ages_days": [100.0],
                                "cutoff_age_days": 26750.0,
                                "ehr_latent": torch.ones(120),
                            }
                        }
                    },
                    stream,
                )

            def fake_sampler(command, **_):
                raw_dir = Path(command[command.index("--output-dir") + 1])
                raw_dir.mkdir(parents=True, exist_ok=True)
                records = []
                for sample_index in range(2):
                    for view_code in ("21017", "21018"):
                        image_path = raw_dir / f"raw-{view_code}-{sample_index}.png"
                        Image.new("L", (16, 8), color=sample_index + 1).save(image_path)
                        records.append(
                            {
                                "patient_id": 50,
                                "view_code": view_code,
                                "sample_index": sample_index,
                                "path": str(image_path),
                                "seed": 42,
                            }
                        )
                (raw_dir / "samples.json").write_text(json.dumps(records), encoding="utf-8")

            args = argparse.Namespace(
                output_root=str(root / "dataset"),
                force=False,
                ehr_pickle=str(trajectory_path),
                config="configs/oct_ehr_ldm.json",
                generator_checkpoint=str(root / "generator.pt"),
                patient_id=[],
                offset=0,
                limit=None,
                patient_id_mode="sequential",
                sequential_start=50,
                view_code=[],
                view_laterality=[],
                samples_per_view=2,
                guidance_scale=4.0,
                inference_steps=10,
                seed=42,
                no_ema=False,
            )
            (root / "generator.pt").touch()
            with patch("factory.generate_oct.subprocess.run", side_effect=fake_sampler):
                patient_map = generate(args)
            self.assertEqual(patient_map["entries"][0]["patient_id"], 50)
            images = sorted((root / "dataset" / "images" / "50").glob("*.png"))
            self.assertEqual(len(images), 4)
            self.assertEqual(images[0].name, "50_synthetic_0000_left.png")
            with (root / "dataset" / "oct_manifest.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["visit_id"] for row in rows}, {"50@synthetic_0000", "50@synthetic_0001"})
            self.assertTrue(all(row["qc_pass"] == "true" for row in rows))


@unittest.skipUnless(DEFAULT_SCHEMA_PROJECT_ROOT.is_dir(), "xdiabetes2 sibling project is unavailable")
class SchemaPackagingTests(unittest.TestCase):
    def test_packages_two_visits_with_two_oct_tokens_each(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trajectory_path = root / "trajectories.pkl"
            trajectory = {
                "tokens": [11, 12, 13],
                "ages_days": [100.0, 200.0, 300.0],
                "cutoff_age_days": 26750.0,
                "ehr_latent": torch.arange(120, dtype=torch.float32),
            }
            with trajectory_path.open("wb") as stream:
                pickle.dump({"trajectories": {1000010: trajectory}}, stream)
            self.assertEqual(set(load_trajectories(trajectory_path)), {1000010})

            patient_map = {
                "version": 1,
                "patient_id_mode": "source",
                "entries": [{"source_patient_id": 1000010, "patient_id": 1000010}],
                "generation": {
                    "generator_checkpoint": "/checkpoints/generator.pt",
                    "samples_per_view": 2,
                    "view_codes": ["21017", "21018"],
                    "seed": 42,
                },
            }
            (root / "patient_map.json").write_text(json.dumps(patient_map), encoding="utf-8")
            image_names = [
                "1000010_synthetic_0000_left.png",
                "1000010_synthetic_0000_right.png",
                "1000010_synthetic_0001_left.png",
                "1000010_synthetic_0001_right.png",
            ]
            torch.save(
                {
                    "features_by_eid": {
                        "1000010": {
                            "image_names": image_names,
                            "features": torch.arange(4 * 1024, dtype=torch.float32).reshape(4, 1024),
                        }
                    },
                    "metadata": {
                        "generator_checkpoint": "/checkpoints/generator.pt",
                        "encoder_name": "RETFound",
                        "encoder_checkpoint": "/checkpoints/retfound.pth",
                    },
                },
                root / "OCT_features_synthetic.pt",
            )
            with (root / "oct_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                fieldnames = [
                    "patient_id",
                    "source_patient_id",
                    "visit_id",
                    "index_time_days",
                    "image_name",
                    "qc_pass",
                    "laterality",
                ]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for sample_index in range(2):
                    for laterality in ("left", "right"):
                        writer.writerow(
                            {
                                "patient_id": "1000010",
                                "source_patient_id": "1000010",
                                "visit_id": f"1000010@synthetic_{sample_index:04d}",
                                "index_time_days": "26750.0",
                                "image_name": f"1000010_synthetic_{sample_index:04d}_{laterality}.png",
                                "qc_pass": "true",
                                "laterality": laterality,
                            }
                        )

            payload = package(
                argparse.Namespace(
                    output_root=str(root),
                    output_pt=None,
                    force=False,
                    ehr_pickle=str(trajectory_path),
                    oct_features=None,
                    oct_manifest=None,
                    schema_project_root=str(DEFAULT_SCHEMA_PROJECT_ROOT),
                    dataset_name="test_synthetic",
                )
            )
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(len(payload["samples"]), 2)
            sample = payload["samples"][0]
            self.assertEqual(tuple(sample["modalities"]["ehr"]["features"].shape), (1, 120))
            self.assertEqual(tuple(sample["modalities"]["oct"]["features"].shape), (2, 1024))
            self.assertFalse(sample["target"]["tasks"]["diabetes"]["mask"].any())
            self.assertTrue((root / "dataset_manifest.json").is_file())
            self.assertTrue((root / "ukb_synthetic_train.pt").is_file())


if __name__ == "__main__":
    unittest.main()
