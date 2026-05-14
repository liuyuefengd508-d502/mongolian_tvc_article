#!/usr/bin/env python3
"""Create annotation skeletons from automatic layout detections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from layout_columns import detect_columns, load_gray


def read_manifest(path: Path, split: str | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows


def build_annotation_pages(rows: list[dict], method: str) -> list[dict]:
    pages: list[dict] = []
    for row in rows:
        image_path = Path(row["image_path"])
        detections = detect_columns(image_path, method)
        gray = load_gray(image_path)
        columns = []
        for idx, det in enumerate(detections):
            columns.append(
                {
                    "column_id": f"{row['page_id']}_col_{idx:03d}",
                    "bbox": det["bbox"],
                    "reading_order": int(det.get("reading_order", idx)),
                    "orientation": det.get("orientation", "correct"),
                    "transcript": "",
                    "degradation_tags": [],
                    "ignore": False,
                    "notes": f"bootstrapped from {method}",
                }
            )
        pages.append(
            {
                "page_id": row["page_id"],
                "image_path": str(image_path),
                "split": row["split"],
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "columns": columns,
            }
        )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--method",
        default="proposed",
        choices=["sauvola_projection", "connected_components", "proposed"],
    )
    parser.add_argument("--split", choices=["train_unlabeled", "val", "test"])
    parser.add_argument("--dataset-name", default="Traditional Mongolian Page-Level OCR")
    parser.add_argument("--version", default="bootstrap-v1")
    args = parser.parse_args()

    rows = read_manifest(args.manifest, args.split)
    payload = {
        "dataset": {
            "name": args.dataset_name,
            "version": args.version,
            "split_manifest": str(args.manifest),
            "notes": f"Bootstrapped from automatic detections using {args.method}. Review manually before evaluation."
        },
        "pages": build_annotation_pages(rows, args.method),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote bootstrap annotations for {len(payload['pages'])} pages to {args.output}")


if __name__ == "__main__":
    main()
