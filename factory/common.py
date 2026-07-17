from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PROJECT_ROOT = REPO_ROOT.parent.parent / "xdiabetes2"
EXPECTED_EHR_DIM = 120
EXPECTED_OCT_DIM = 1024
LABEL_NAMES = [
    "token_220",
    "token_221",
    "token_222",
    "token_223",
    "token_224",
    "token_961",
]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


def atomic_json_dump(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(Path(path), map_location="cpu")


def normalize_patient_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid patient IDs")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Patient ID must be integer-like, received {value!r}") from error
    if str(value).strip() not in {str(normalized), f"+{normalized}"}:
        try:
            if float(value) != normalized:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"Patient ID must be integer-like, received {value!r}") from None
    return normalized


def load_trajectories(path: str | Path) -> dict[int, Mapping[str, Any]]:
    # The trajectory file is a trusted project artifact and intentionally uses pickle.
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("trajectories"), Mapping):
        raise TypeError("Expected a pickle payload containing a mapping at payload['trajectories']")
    trajectories: dict[int, Mapping[str, Any]] = {}
    for raw_id, trajectory in payload["trajectories"].items():
        patient_id = normalize_patient_id(raw_id)
        if patient_id in trajectories:
            raise ValueError(f"Duplicate normalized source patient ID: {patient_id}")
        if not isinstance(trajectory, Mapping):
            raise TypeError(f"Trajectory for patient {patient_id} must be a mapping")
        trajectories[patient_id] = trajectory
    if not trajectories:
        raise ValueError(f"No trajectories found in {path}")
    return trajectories


def trajectory_index_time(trajectory: Mapping[str, Any], patient_id: int) -> tuple[float, str]:
    if trajectory.get("cutoff_age_days") is not None:
        value = float(trajectory["cutoff_age_days"])
        source = "cutoff_age_days"
    elif trajectory.get("cutoff_age_years") is not None:
        value = float(trajectory["cutoff_age_years"]) * 365.25
        source = "cutoff_age_years_x_365.25"
    else:
        raise KeyError(
            f"Trajectory for patient {patient_id} has neither cutoff_age_days nor cutoff_age_years"
        )
    if not math.isfinite(value):
        raise ValueError(f"Non-finite cutoff age for patient {patient_id}: {value}")
    return value, source


def trajectory_latent(
    trajectory: Mapping[str, Any], patient_id: int, expected_dim: int = EXPECTED_EHR_DIM
) -> torch.Tensor:
    if "ehr_latent" not in trajectory:
        raise KeyError(f"Trajectory for patient {patient_id} has no ehr_latent")
    latent = torch.as_tensor(trajectory["ehr_latent"], dtype=torch.float32).detach().cpu()
    if latent.ndim == 2 and 1 in latent.shape:
        latent = latent.reshape(-1)
    if latent.ndim != 1 or latent.numel() != expected_dim:
        raise ValueError(
            f"ehr_latent for patient {patient_id} must contain {expected_dim} values; "
            f"received shape {tuple(latent.shape)}"
        )
    if not torch.isfinite(latent).all():
        raise ValueError(f"ehr_latent for patient {patient_id} contains NaN/Inf")
    return latent.contiguous()


def trajectory_length(trajectory: Mapping[str, Any], patient_id: int) -> int:
    tokens = trajectory.get("tokens")
    if tokens is None:
        raise KeyError(f"Trajectory for patient {patient_id} has no tokens")
    length = len(tokens)
    ages = trajectory.get("ages_days")
    if ages is not None and len(ages) != length:
        raise ValueError(
            f"tokens/ages_days length mismatch for patient {patient_id}: {length} != {len(ages)}"
        )
    return length


def parse_key_value(values: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip() or not item.strip():
            raise ValueError(f"{label} must use KEY=VALUE syntax, received {value!r}")
        parsed[key.strip()] = item.strip()
    return parsed
