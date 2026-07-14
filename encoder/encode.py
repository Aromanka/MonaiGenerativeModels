#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import models_vit


class OCTImageDataset(Dataset):
    def __init__(
        self,
        records: List[Tuple[str, str, Path]],
        transform: transforms.Compose,
        skip_bad_images: bool,
    ) -> None:
        self.records = records
        self.transform = transform
        self.skip_bad_images = skip_bad_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        eid, image_name, image_path = self.records[index]
        try:
            with Image.open(image_path) as img:
                image = self.transform(img.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            if self.skip_bad_images:
                return None
            raise

        return {
            "eid": eid,
            "image_name": image_name,
            "image_path": str(image_path),
            "image": image,
        }


def collate_oct_batch(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None

    return {
        "eids": [item["eid"] for item in valid],
        "image_names": [item["image_name"] for item in valid],
        "image_paths": [item["image_path"] for item in valid],
        "images": torch.stack([item["image"] for item in valid], dim=0),
    }


def load_eid_image_index(path: Union[str, Path]) -> Dict[str, List[str]]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object mapping eid -> image paths, got {type(data)}")

    index: Dict[str, List[str]] = {}
    for eid, image_paths in data.items():
        if not isinstance(image_paths, list):
            raise ValueError(f"Expected a list of image paths for eid {eid}, got {type(image_paths)}")
        index[str(eid)] = [str(path) for path in image_paths]

    return index


def make_records(eid_image_index: Dict[str, List[str]]) -> List[Tuple[str, str, Path]]:
    records: List[Tuple[str, str, Path]] = []
    for eid, image_paths in eid_image_index.items():
        for image_path in image_paths:
            path = Path(image_path).expanduser()
            records.append((eid, path.stem, path))
    return records


def get_checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or contain a 'model'/'state_dict' entry.")

    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def build_model(checkpoint_path: Union[str, Path], device: torch.device) -> torch.nn.Module:
    model = models_vit.vit_large_patch16(num_classes=0, global_pool=True)
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    state_dict = get_checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"[WARN] Missing model keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")

    model.to(device)
    model.eval()
    return model


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def encode_oct_images(
    input_json: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_pt: Union[str, Path],
    batch_size: int,
    num_workers: int,
    device_name: str,
    image_size: int,
    skip_bad_images: bool,
) -> None:
    device = torch.device(device_name)
    eid_image_index = load_eid_image_index(input_json)
    records = make_records(eid_image_index)

    if not records:
        raise ValueError(f"No image paths found in {input_json}")

    dataset = OCTImageDataset(
        records=records,
        transform=build_transform(image_size),
        skip_bad_images=skip_bad_images,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_oct_batch,
    )

    model = build_model(checkpoint_path, device)
    encoded: Dict[str, Dict[str, Any]] = {
        eid: {"image_names": [], "features": []} for eid in eid_image_index
    }
    skipped: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding OCT images", total=len(dataloader)):
            if batch is None:
                continue

            images = batch["images"].to(device, non_blocking=True)
            features = model(images).detach().cpu()

            for eid, image_name, feature in zip(
                batch["eids"],
                batch["image_names"],
                features,
            ):
                encoded[eid]["image_names"].append(image_name)
                encoded[eid]["features"].append(feature)

    for eid in list(encoded):
        feature_list = encoded[eid]["features"]
        if feature_list:
            encoded[eid]["features"] = torch.stack(feature_list, dim=0)
        else:
            encoded[eid]["features"] = torch.empty((0, 0), dtype=torch.float32)
            skipped.append(eid)

    output_pt = Path(output_pt).expanduser()
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features_by_eid": encoded,
            "metadata": {
                "input_json": str(Path(input_json).expanduser()),
                "checkpoint_path": str(Path(checkpoint_path).expanduser()),
                "image_size": image_size,
                "feature_layout": "features_by_eid[eid]['features'][i] matches features_by_eid[eid]['image_names'][i]",
            },
        },
        output_pt,
    )

    image_count = sum(len(value["image_names"]) for value in encoded.values())
    print(f"Encoded patients: {sum(bool(value['image_names']) for value in encoded.values())}")
    print(f"Encoded images: {image_count}")
    if skipped:
        print(f"[WARN] Patients without encoded images: {len(skipped)}")
    print(f"Saved feature file: {output_pt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch encode selected OCT images with RETFound and save an eid -> "
            "image-name-aligned feature tensor mapping as a .pt file."
        )
    )
    parser.add_argument(
        "--input_json",
        default="data/OCT_eid.json",
        help="JSON mapping eid -> list of OCT image paths.",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="RETFound OCT checkpoint path, e.g. /data/home/wanglidi/model/RETFound_oct_weights.pth.",
    )
    parser.add_argument(
        "--output_pt",
        default="data/OCT_features.pt",
        help="Output .pt file path.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Images per inference batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Model input image size.")
    parser.add_argument(
        "--no_skip_bad_images",
        action="store_true",
        help="Raise an error instead of skipping missing or unreadable images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encode_oct_images(
        input_json=args.input_json,
        checkpoint_path=args.checkpoint_path,
        output_pt=args.output_pt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        image_size=args.image_size,
        skip_bad_images=not args.no_skip_bad_images,
    )


if __name__ == "__main__":
    main()

"""
Example:
python encode.py \
  --input_json data/OCT_eid.json \
  --checkpoint_path /data/home/wanglidi/model/RETFound_oct_weights.pth \
  --output_pt data/OCT_features.pt \
  --batch_size 64 \
  --num_workers 4
"""
