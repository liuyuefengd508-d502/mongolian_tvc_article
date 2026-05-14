#!/usr/bin/env python3
"""Prepare a balanced train-labeling batch for Label Studio."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
from pathlib import Path

from PIL import Image

from layout_columns import detect_columns


ROOT = Path(__file__).resolve().parent
MEDIA_DIR = ROOT / "label_studio_media"
IMPORT_DIR = ROOT / "label_studio_import"
MANIFEST = ROOT / "page_split_manifest.csv"


DEFAULT_BATCH = [
    "80-48-61-2",
    "80-48-62-1(2)",
    "80-48-63-1",
    "80-48-64-1(1)",
    "80-48-65-1(3)",
    "80-48-66-1(2)",
    "80-48-67-1",
    "80-48-68-1",
    "80-48-69-1",
    "80-48-70-3",
    "80-48-71-1",
    "80-48-72-1",
    "80-48-72-3",
    "80-48-73-2",
    "80-48-74-1",
    "80-48-75-1",
    "80-48-76-1",
    "80-48-78-1",
    "80-48-79-1",
    "80-48-80-1",
]


def read_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open(encoding="utf-8") as f:
        return {row["page_id"]: row for row in csv.DictReader(f)}


def convert_to_jpg(source: Path, dest: Path) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        rgb = image.convert("RGB")
        rgb.save(dest, quality=95)
        return rgb.size


def bbox_to_percent(bbox: list[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    return {
        "x": x1 / width * 100.0,
        "y": y1 / height * 100.0,
        "width": max(0.0, x2 - x1) / width * 100.0,
        "height": max(0.0, y2 - y1) / height * 100.0,
        "rotation": 0,
        "rectanglelabels": ["TextColumn"],
    }


def prediction_result(detection: dict, width: int, height: int) -> list[dict]:
    region_id = secrets.token_hex(8)
    return [
        {
            "id": region_id,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": bbox_to_percent(detection["bbox"], width, height),
            "origin": "prediction",
        },
        {
            "id": region_id,
            "from_name": "reading_order",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [str(detection.get("reading_order", ""))]},
            "origin": "prediction",
        },
        {
            "id": region_id,
            "from_name": "orientation",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [detection.get("orientation", "correct")]},
            "origin": "prediction",
        },
    ]


def prepare_batch(page_ids: list[str], batch_name: str, method: str) -> dict:
    manifest = read_manifest()
    media_subdir = f"{batch_name}_jpg"
    output_media_dir = MEDIA_DIR / media_subdir
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    missing: list[str] = []
    for page_id in page_ids:
        row = manifest.get(page_id)
        if row is None:
            missing.append(page_id)
            continue
        source = Path(row["image_path"])
        target = output_media_dir / f"{page_id}.jpg"
        width, height = convert_to_jpg(source, target)
        detections = detect_columns(target, method)
        pred_result: list[dict] = []
        for detection in detections:
            pred_result.extend(prediction_result(detection, width, height))
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d={media_subdir}/{target.name}",
                    "page_id": page_id,
                    "split": "train",
                    "source_split": row["split"],
                    "source_folder": row["source_folder"],
                    "original_image_path": row["image_path"],
                    "jpg_path": str(target),
                    "width": width,
                    "height": height,
                },
                "predictions": (
                    [{"model_version": f"bootstrap-{method}", "score": 0.5, "result": pred_result}]
                    if pred_result
                    else []
                ),
            }
        )

    task_file = IMPORT_DIR / f"{batch_name}_tasks.json"
    task_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_csv = IMPORT_DIR / f"{batch_name}_selected_pages.csv"
    with selected_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page_id", "source_folder", "image_path"])
        writer.writeheader()
        for task in tasks:
            data = task["data"]
            writer.writerow(
                {
                    "page_id": data["page_id"],
                    "source_folder": data["source_folder"],
                    "image_path": data["original_image_path"],
                }
            )

    return {
        "batch_name": batch_name,
        "tasks": len(tasks),
        "missing": missing,
        "media_dir": str(output_media_dir),
        "task_file": str(task_file),
        "selected_csv": str(selected_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-name", default="train_batch01")
    parser.add_argument("--method", default="proposed", choices=["sauvola_projection", "connected_components", "proposed"])
    parser.add_argument("--page-ids", help="Comma-separated page ids. Defaults to a balanced 20-page batch.")
    args = parser.parse_args()

    page_ids = [page.strip() for page in args.page_ids.split(",")] if args.page_ids else DEFAULT_BATCH
    summary = prepare_batch(page_ids=page_ids, batch_name=args.batch_name, method=args.method)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
