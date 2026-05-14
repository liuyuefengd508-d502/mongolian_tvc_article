#!/usr/bin/env python3
"""Prepare rotated Label Studio tasks for selected pages."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from PIL import Image

from layout_columns import detect_columns


ROOT = Path(__file__).resolve().parent
MEDIA_DIR = ROOT / "label_studio_media"
IMPORT_DIR = ROOT / "label_studio_import"


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


def load_source_tasks(task_file: Path) -> dict[str, dict]:
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    return {task["data"]["page_id"]: task for task in tasks}


def rotate_image(source: Path, target: Path, direction: str) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        if direction == "ccw":
            rotated = image.convert("RGB").rotate(90, expand=True)
        elif direction == "cw":
            rotated = image.convert("RGB").rotate(-90, expand=True)
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        rotated.save(target, quality=95)
        return rotated.size


def prepare_rotated_tasks(
    source_task_file: Path,
    page_ids: list[str],
    output_name: str,
    direction: str,
    method: str,
) -> dict:
    source_tasks = load_source_tasks(source_task_file)
    media_subdir = f"{output_name}_jpg"
    output_media_dir = MEDIA_DIR / media_subdir
    output_tasks: list[dict] = []
    missing: list[str] = []

    for page_id in page_ids:
        source_task = source_tasks.get(page_id)
        if source_task is None:
            missing.append(page_id)
            continue

        data = dict(source_task["data"])
        source_image = Path(data["jpg_path"])
        target_image = output_media_dir / f"{page_id}_rot90{direction}.jpg"
        width, height = rotate_image(source_image, target_image, direction)
        detections = detect_columns(target_image, method)

        pred_result: list[dict] = []
        for detection in detections:
            pred_result.extend(prediction_result(detection, width, height))

        data.update(
            {
                "image": f"http://localhost:8090/{media_subdir}/{target_image.name}",
                "image_access_mode": "http_static_8090",
                "jpg_path": str(target_image),
                "width": width,
                "height": height,
                "rotation_note": f"rotated_90_{direction}_for_labeling",
            }
        )
        output_tasks.append(
            {
                "data": data,
                "predictions": (
                    [{"model_version": f"bootstrap-{method}-rot90{direction}", "score": 0.5, "result": pred_result}]
                    if pred_result
                    else []
                ),
            }
        )

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_task_file = IMPORT_DIR / f"{output_name}_tasks.json"
    output_task_file.write_text(json.dumps(output_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_name": output_name,
        "direction": direction,
        "tasks": len(output_tasks),
        "missing": missing,
        "media_dir": str(output_media_dir),
        "task_file": str(output_task_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task-file", required=True, type=Path)
    parser.add_argument("--page-ids", required=True, help="Comma-separated page ids to rotate.")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--direction", default="ccw", choices=["ccw", "cw"])
    parser.add_argument("--method", default="proposed", choices=["sauvola_projection", "connected_components", "proposed"])
    args = parser.parse_args()

    page_ids = [page.strip() for page in args.page_ids.split(",") if page.strip()]
    summary = prepare_rotated_tasks(
        source_task_file=args.source_task_file,
        page_ids=page_ids,
        output_name=args.output_name,
        direction=args.direction,
        method=args.method,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
