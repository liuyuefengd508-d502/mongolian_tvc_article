#!/usr/bin/env python3
"""Create deterministic page-level splits for full archival scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def page_id_from_path(path: Path) -> str:
    return path.stem


def choose_split(page_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    value = stable_unit_interval(page_id, seed)
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train_unlabeled"


def collect_images(data_root: Path) -> list[Path]:
    return sorted(
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def write_manifest(images: list[Path], output: Path, data_root: Path, seed: int, val_ratio: float, test_ratio: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page_id",
                "split",
                "image_path",
                "relative_path",
                "source_folder",
            ],
        )
        writer.writeheader()
        for image in images:
            page_id = page_id_from_path(image)
            writer.writerow(
                {
                    "page_id": page_id,
                    "split": choose_split(page_id, seed, val_ratio, test_ratio),
                    "image_path": str(image),
                    "relative_path": str(image.relative_to(data_root)),
                    "source_folder": image.parent.name,
                }
            )


def summarize(output: Path) -> dict[str, int]:
    counts = {"train_unlabeled": 0, "val": 0, "test": 0}
    with output.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["split"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Root containing full-page archival images.")
    parser.add_argument("--output", required=True, type=Path, help="CSV manifest path to create.")
    parser.add_argument("--seed", default=20260506, type=int, help="Deterministic split seed.")
    parser.add_argument("--val-ratio", default=0.15, type=float, help="Validation page ratio.")
    parser.add_argument("--test-ratio", default=0.15, type=float, help="Test page ratio.")
    args = parser.parse_args()

    images = collect_images(args.data_root)
    if not images:
        raise SystemExit(f"No images found under {args.data_root}")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("Require non-negative ratios with val_ratio + test_ratio < 1.")

    write_manifest(images, args.output, args.data_root, args.seed, args.val_ratio, args.test_ratio)
    counts = summarize(args.output)
    print(f"Wrote {len(images)} pages to {args.output}")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
