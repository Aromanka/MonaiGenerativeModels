#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from tqdm import tqdm


OCT_ZIP_PATTERN = re.compile(
    r"^(?P<eid>\d+)_(?P<field_id>\d+)_(?P<instance>\d+)_(?P<array>\d+)\.zip$",
    re.IGNORECASE,
)


def load_lines(path: str | Path) -> list[str]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_eids(path: str | Path) -> set[str]:
    eids: set[str] = set()
    with open(Path(path).expanduser(), "r", encoding="utf-8", newline="") as f:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(f)
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        return eids

    header = [cell.lower() for cell in rows[0]]
    eid_col = None
    for column_name in ("eid", "csvpatient_id", "patient_id"):
        if column_name in header:
            eid_col = header.index(column_name)
            rows = rows[1:]
            break

    # Accept plain txt, one-column csv, a csv with eid/csvpatient_id/patient_id,
    # or a csv with eid as the first value.
    if eid_col is None:
        eid_col = 0
        if header and header[0] in {"eid", "csvpatient_id", "patient_id"}:
            rows = rows[1:]

    for row in rows:
        if eid_col < len(row):
            value = row[eid_col].strip()
            if value:
                eids.add(value)
    return eids


def eid_from_zip_name(zip_name: str) -> str | None:
    match = OCT_ZIP_PATTERN.match(Path(zip_name).name)
    if match:
        return match.group("eid")

    # Fallback for non-standard names that still begin with the eid.
    prefix = Path(zip_name).name.split("_", maxsplit=1)[0]
    return prefix if prefix.isdigit() else None


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (extract_root / info.filename).resolve()
            try:
                target.relative_to(extract_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path in {zip_path}: {info.filename}")
            zf.extract(info, extract_root)


def find_middle_image(extract_dir: Path, eid=None) -> Path | None:
    # rglob("image*_64.*") 会匹配所有以 image 开头、_64 结尾，且带有任意后缀的文件
    for path in extract_dir.rglob("image*_64.*"):
        if path.is_file():
            return path

    return None


def copy_middle_oct_images(
    zip_names: list[str],
    oct_basepath: str | Path,
    target_eids: set[str],
    final_data_folder: str | Path,
) -> dict[str, list[str]]:
    oct_basepath = Path(oct_basepath).expanduser().resolve()
    final_data_folder = Path(final_data_folder).expanduser().resolve()
    final_data_folder.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}

    for raw_zip_name in tqdm(zip_names, desc="Processing ZIPs"):
        zip_name = Path(raw_zip_name).name
        eid = eid_from_zip_name(zip_name)

        if eid is None or eid not in target_eids:
            continue

        zip_path = (oct_basepath / zip_name).resolve()
        if not zip_path.exists():
            print(f"[WARN] Missing zip file for eid {eid}: {zip_path}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"oct_{eid}_") as temp_dir:
            temp_path = Path(temp_dir)
            try:
                safe_extract_zip(zip_path, temp_path)
            except zipfile.BadZipFile:
                print(f"[WARN] Bad zip file for eid {eid}: {zip_path}")
                continue

            middle_image = find_middle_image(temp_path, eid)
            if middle_image is None:
                print(f"[WARN] Missing image{eid}_64 in: {zip_path}")
                continue

            dest_name = f"{Path(zip_name).stem}_{middle_image.name}"
            dest_path = (final_data_folder / dest_name).resolve()
            shutil.copy2(middle_image, dest_path)
            results.setdefault(eid, [])
            results[eid].append(str(dest_path))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract image{eid}_64 from target patients' OCT zip files, copy the "
            "images to a final folder, and write an eid-to-image-paths JSON index."
        )
    )
    parser.add_argument(
        "--zip_names_path",
        required=True,
        help="Text file containing OCT zip file names, one per line.",
    )
    parser.add_argument(
        "--target_eids_path",
        required=True,
        help="Text or CSV file containing target patient eids.",
    )
    parser.add_argument(
        "--OCT_basepath",
        required=True,
        help="Base folder containing OCT zip files. OCT_basepath / filename is used.",
    )
    parser.add_argument(
        "--final_data_folder",
        required=True,
        help="Folder where selected OCT images will be copied.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to write the JSON mapping eid -> absolute selected image paths.",
    )

    args = parser.parse_args()

    zip_names = load_lines(args.zip_names_path)
    target_eids = load_eids(args.target_eids_path)

    results = copy_middle_oct_images(
        zip_names=zip_names,
        oct_basepath=args.OCT_basepath,
        target_eids=target_eids,
        final_data_folder=args.final_data_folder,
    )

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    copied_count = sum(len(paths) for paths in results.values())
    print(f"Target eids: {len(target_eids)}")
    print(f"Copied OCT images: {copied_count}")
    print(f"Saved JSON index: {output_json}")


if __name__ == "__main__":
    main()

"""
python OCT_extract_middle_image.py --zip_names_path data/oct_zip_filenames.txt --target_eids_path data/train_diabetes_labels.csv --OCT_basepath /data/home/UKB/data/OCT/ --final_data_folder data/OCT_Images --output_json data/OCT_eid.json
"""