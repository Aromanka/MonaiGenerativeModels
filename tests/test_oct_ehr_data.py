from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from oct_ehr_ldm.config import load_config
from oct_ehr_ldm.data import create_datasets, load_oct_image, prepare_manifest


class TestOCTEHRData(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        data_dir = root / "data"
        data_dir.mkdir()
        images = data_dir / "images"
        images.mkdir()
        ehr_with_oct = {patient_id: torch.full((6,), float(patient_id)) for patient_id in range(100, 106)}
        ehr_only = {999: torch.ones(6)}
        torch.save(ehr_with_oct, data_dir / "with.pt")
        torch.save(ehr_only, data_dir / "only.pt")
        oct_index = {}
        for patient_id in ehr_with_oct:
            image_path = images / f"{patient_id}_21017_0_0_image{patient_id}_64.png"
            Image.new("L", (12, 8), color=patient_id % 255).save(image_path)
            oct_index[str(patient_id)] = [str(image_path)]
        (data_dir / "oct.json").write_text(json.dumps(oct_index), encoding="utf-8")
        config = {
            "project_root": "..",
            "paths": {
                "ehr_with_oct": "data/with.pt",
                "ehr_only": "data/only.pt",
                "oct_index": "data/oct.json",
                "split_manifest": "outputs/split.json"
            },
            "data": {"image_size": 16, "val_fraction": 0.34, "split_seed": 7, "resize_mode": "pad"}
        }
        config_dir = root / "configs"
        config_dir.mkdir()
        config_path = config_dir / "test.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_patient_split_has_no_leakage_and_excludes_ehr_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(self._fixture(Path(temp)))
            manifest = prepare_manifest(config)
            train_ids = set(manifest["train_patient_ids"])
            val_ids = set(manifest["val_patient_ids"])
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(train_ids | val_ids, set(range(100, 106)))
            self.assertNotIn(999, train_ids | val_ids)
            self.assertEqual(manifest["ehr_dim"], 6)
            self.assertEqual(manifest["view_to_id"], {"<unknown>": 0, "21017": 1})

            train, val = create_datasets(config, manifest)
            self.assertEqual({item["patient_id"] for item in train.records}, train_ids)
            self.assertEqual({item["patient_id"] for item in val.records}, val_ids)

    def test_pad_resize_preserves_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "oct.png"
            Image.new("L", (12, 6), color=128).save(image_path)
            tensor = load_oct_image(image_path, 16, "pad")
            self.assertEqual(tuple(tensor.shape), (1, 16, 16))
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
