from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ProjectConfig


VIEW_PATTERN = re.compile(r"^(?P<eid>\d+)_(?P<field_id>\d+)_(?P<instance>\d+)_(?P<array>\d+)_")
UNKNOWN_VIEW = "<unknown>"


@dataclass(frozen=True)
class OCTRecord:
    patient_id: int
    image_path: str
    view_code: str


def safe_torch_load(path: str | Path) -> Any:
    """Load tensor-only project files without silently enabling arbitrary pickle globals."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 does not expose weights_only
        return torch.load(path, map_location="cpu")


def normalize_patient_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid patient IDs")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid patient ID: {value!r}") from error


def normalize_ehr_latent(value: Any, patient_id: int) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"EHR latent for patient {patient_id} is not a torch.Tensor: {type(value)!r}")
    latent = value.detach().cpu().float()
    if latent.ndim == 2 and 1 in latent.shape:
        latent = latent.reshape(-1)
    if latent.ndim != 1:
        raise ValueError(
            f"EHR latent for patient {patient_id} must have shape [D] (or singleton [1,D]); got {tuple(latent.shape)}"
        )
    if latent.numel() == 0 or not torch.isfinite(latent).all():
        raise ValueError(f"EHR latent for patient {patient_id} is empty or contains NaN/Inf")
    return latent.contiguous()


def load_ehr_dictionary(path: str | Path) -> dict[int, torch.Tensor]:
    raw = safe_torch_load(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a pure patient->latent dictionary in {path}; got {type(raw)!r}")
    normalized: dict[int, torch.Tensor] = {}
    expected_dim: int | None = None
    for raw_id, raw_latent in tqdm(raw.items(), desc="Loading EHR latents", unit="patient"):
        patient_id = normalize_patient_id(raw_id)
        if patient_id in normalized:
            raise ValueError(f"Duplicate normalized patient ID {patient_id} in {path}")
        latent = normalize_ehr_latent(raw_latent, patient_id)
        if expected_dim is None:
            expected_dim = latent.numel()
        elif latent.numel() != expected_dim:
            raise ValueError(
                f"Inconsistent EHR latent size in {path}: patient {patient_id} has {latent.numel()}, expected {expected_dim}"
            )
        normalized[patient_id] = latent
    if not normalized:
        raise ValueError(f"EHR dictionary is empty: {path}")
    return normalized


def load_oct_index(path: str | Path) -> dict[int, list[str]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise TypeError(f"OCT index must be a JSON object: {path}")
    normalized: dict[int, list[str]] = {}
    for raw_id, raw_paths in tqdm(raw.items(), desc="Loading OCT index", unit="patient"):
        patient_id = normalize_patient_id(raw_id)
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, list) and all(isinstance(item, str) for item in raw_paths):
            paths = raw_paths
        else:
            raise TypeError(f"OCT paths for patient {patient_id} must be a string list")
        normalized[patient_id] = paths
    return normalized


def infer_view_code(image_path: str) -> str:
    match = VIEW_PATTERN.match(Path(image_path).name)
    return match.group("field_id") if match else UNKNOWN_VIEW


def build_records(
    ehr_by_patient: dict[int, torch.Tensor],
    oct_by_patient: dict[int, list[str]],
    fail_missing_images: bool = True,
) -> tuple[list[OCTRecord], list[str]]:
    records: list[OCTRecord] = []
    missing: list[str] = []
    paired_patient_ids = sorted(ehr_by_patient.keys() & oct_by_patient.keys())
    for patient_id in tqdm(paired_patient_ids, desc="Validating paired OCT", unit="patient"):
        for raw_path in oct_by_patient[patient_id]:
            image_path = str(Path(raw_path).expanduser())
            if not Path(image_path).is_file():
                missing.append(image_path)
                continue
            records.append(OCTRecord(patient_id, image_path, infer_view_code(image_path)))
    if missing and fail_missing_images:
        examples = "\n".join(f"  - {item}" for item in missing[:10])
        raise FileNotFoundError(f"{len(missing)} OCT images from the JSON index do not exist. Examples:\n{examples}")
    if not records:
        raise ValueError("No usable paired EHR/OCT records remain after joining the two data sources")
    return records, missing


def _fingerprint(records: Iterable[OCTRecord], ehr_by_patient: dict[int, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    patient_ids = sorted({record.patient_id for record in records})
    for patient_id in patient_ids:
        latent = ehr_by_patient[patient_id].contiguous()
        digest.update(f"ehr\0{patient_id}\0{tuple(latent.shape)}\0".encode("ascii"))
        digest.update(latent.numpy().tobytes())
    for record in sorted(records, key=lambda item: (item.patient_id, item.image_path)):
        digest.update(f"{record.patient_id}\0{record.image_path}\0{record.view_code}\n".encode("utf-8"))
        stat = Path(record.image_path).stat()
        digest.update(f"{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def prepare_manifest(config: ProjectConfig, output_path: str | Path | None = None) -> dict[str, Any]:
    ehr_path = config.path("paths.ehr_with_oct")
    oct_path = config.path("paths.oct_index")
    ehr = load_ehr_dictionary(ehr_path)
    oct_index = load_oct_index(oct_path)
    fail_missing = bool(config.get("data.fail_missing_images", True))
    records, missing = build_records(ehr, oct_index, fail_missing_images=fail_missing)

    paired_ids = sorted({record.patient_id for record in records})
    if len(paired_ids) < 2:
        raise ValueError("At least two paired patients are required for separate train and validation sets")
    seed = int(config.get("data.split_seed", 2026))
    val_fraction = float(config.get("data.val_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("data.val_fraction must be strictly between 0 and 1")
    shuffled = paired_ids.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    train_set, val_set = set(train_ids), set(val_ids)

    view_codes = sorted({record.view_code for record in records if record.view_code != UNKNOWN_VIEW})
    view_to_id = {UNKNOWN_VIEW: 0, **{code: index + 1 for index, code in enumerate(view_codes)}}
    manifest = {
        "version": 1,
        "ehr_with_oct": str(ehr_path),
        "oct_index": str(oct_path),
        "ehr_dim": next(iter(ehr.values())).numel(),
        "split_seed": seed,
        "val_fraction": val_fraction,
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "view_to_id": view_to_id,
        "records": [
            {**asdict(record), "split": "train" if record.patient_id in train_set else "val"}
            for record in records
            if record.patient_id in train_set or record.patient_id in val_set
        ],
        "missing_image_count": len(missing),
        "fingerprint": _fingerprint(records, ehr),
    }
    destination = Path(output_path) if output_path is not None else config.path("paths.split_manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    return manifest


def load_manifest(config: ProjectConfig, create_if_missing: bool = True) -> dict[str, Any]:
    path = config.path("paths.split_manifest")
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(path)
        return prepare_manifest(config, path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    train_ids = set(manifest["train_patient_ids"])
    val_ids = set(manifest["val_patient_ids"])
    if train_ids & val_ids:
        raise ValueError("Patient leakage detected: train and validation IDs overlap in split manifest")
    return manifest


def _resampling_bicubic() -> int:
    return Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def load_oct_image(path: str | Path, image_size: int, resize_mode: str = "pad") -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        if resize_mode == "pad":
            image = ImageOps.pad(
                image,
                (image_size, image_size),
                method=_resampling_bicubic(),
                color=0,
                centering=(0.5, 0.5),
            )
        elif resize_mode == "stretch":
            image = image.resize((image_size, image_size), resample=_resampling_bicubic())
        else:
            raise ValueError(f"Unsupported resize mode: {resize_mode!r}; expected 'pad' or 'stretch'")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


class PairedOCTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        ehr_by_patient: dict[int, torch.Tensor],
        view_to_id: dict[str, int],
        image_size: int,
        resize_mode: str,
        intensity_augmentation: float = 0.0,
    ) -> None:
        self.records = records
        self.ehr_by_patient = ehr_by_patient
        self.view_to_id = view_to_id
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.intensity_augmentation = intensity_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        patient_id = normalize_patient_id(record["patient_id"])
        image = load_oct_image(record["image_path"], self.image_size, self.resize_mode)
        if self.intensity_augmentation > 0:
            magnitude = self.intensity_augmentation
            gain = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * magnitude
            offset = (torch.rand(1).item() * 2.0 - 1.0) * magnitude * 0.25
            image = (image * gain + offset).clamp_(0.0, 1.0)
        view_code = str(record.get("view_code", UNKNOWN_VIEW))
        return {
            "image": image,
            "ehr": self.ehr_by_patient[patient_id],
            "patient_id": patient_id,
            "view_id": self.view_to_id.get(view_code, 0),
            "view_code": view_code,
            "image_path": record["image_path"],
        }


def create_datasets(config: ProjectConfig, manifest: dict[str, Any]) -> tuple[PairedOCTDataset, PairedOCTDataset]:
    ehr = load_ehr_dictionary(config.path("paths.ehr_with_oct"))
    manifest_ids = set(manifest["train_patient_ids"]) | set(manifest["val_patient_ids"])
    missing_ehr = manifest_ids - set(ehr)
    if missing_ehr:
        raise KeyError(f"{len(missing_ehr)} manifest patients are absent from the current EHR file")
    image_size = int(config.get("data.image_size", 512))
    if image_size <= 0 or image_size % 8:
        raise ValueError("data.image_size must be a positive multiple of 8 for the CXR AutoencoderKL")
    resize_mode = str(config.get("data.resize_mode", "pad"))
    view_to_id = {str(key): int(value) for key, value in manifest["view_to_id"].items()}
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    val_records = [record for record in manifest["records"] if record["split"] == "val"]
    train = PairedOCTDataset(
        train_records,
        ehr,
        view_to_id,
        image_size,
        resize_mode,
        intensity_augmentation=float(config.get("data.intensity_augmentation", 0.0)),
    )
    val = PairedOCTDataset(val_records, ehr, view_to_id, image_size, resize_mode, intensity_augmentation=0.0)
    return train, val


def create_loader(
    dataset: Dataset,
    config: ProjectConfig,
    training: bool,
    epoch: int = 0,
) -> DataLoader:
    workers = int(config.get("data.num_workers", 4))
    generator = torch.Generator().manual_seed(int(config.get("training.seed", 2026)) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(config.get("data.batch_size", 4)),
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=int(config.get("data.prefetch_factor", 2)) if workers > 0 else None,
        drop_last=training and len(dataset) >= int(config.get("data.batch_size", 4)),
        generator=generator,
    )


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    train_records = sum(record["split"] == "train" for record in manifest["records"])
    val_records = sum(record["split"] == "val" for record in manifest["records"])
    return {
        "ehr_dim": manifest["ehr_dim"],
        "train_patients": len(manifest["train_patient_ids"]),
        "val_patients": len(manifest["val_patient_ids"]),
        "train_images": train_records,
        "val_images": val_records,
        "views": manifest["view_to_id"],
        "missing_images_skipped": manifest.get("missing_image_count", 0),
        "fingerprint": manifest["fingerprint"],
    }


def compute_train_ehr_stats(dataset: PairedOCTDataset, patient_ids: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    unique_ids = sorted(set(int(item) for item in patient_ids))
    matrix = torch.stack([dataset.ehr_by_patient[patient_id] for patient_id in unique_ids], dim=0).float()
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Non-finite EHR normalization statistics")
    return mean, std
