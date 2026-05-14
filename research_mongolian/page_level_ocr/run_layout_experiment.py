#!/usr/bin/env python3
"""Run page-level layout extraction and optional evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_layout import aggregate_page_metrics, evaluate_page
from layout_columns import detect_columns


def read_manifest(path: Path, split: str | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows


def read_annotations(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {page["page_id"]: page for page in data.get("pages", [])}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--method",
        default="proposed",
        choices=["sauvola_projection", "connected_components", "proposed"],
    )
    parser.add_argument("--split", choices=["train_unlabeled", "val", "test"])
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    args = parser.parse_args()

    rows = read_manifest(args.manifest, args.split)
    annotations = read_annotations(args.annotations)

    detections_by_page: dict[str, dict] = {}
    page_metrics: list[dict] = []

    for row in rows:
        page_id = row["page_id"]
        annotation_page = annotations.get(page_id)
        image_path = Path(
            annotation_page.get("image_path", row["image_path"]) if annotation_page else row["image_path"]
        )
        detections = detect_columns(image_path, args.method)
        detections_by_page[page_id] = {
            "page_id": page_id,
            "split": row["split"],
            "image_path": str(image_path),
            "method": args.method,
            "detections": detections,
        }
        if annotation_page:
            metric = evaluate_page(detections, annotation_page.get("columns", []), args.iou_threshold)
            metric.update({"page_id": page_id, "split": row["split"], "method": args.method})
            page_metrics.append(metric)

    write_json(args.output_dir / f"detections_{args.method}.json", {"pages": list(detections_by_page.values())})

    if page_metrics:
        summary = aggregate_page_metrics(page_metrics)
        write_json(
            args.output_dir / f"metrics_{args.method}.json",
            {"summary": summary, "pages": page_metrics},
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote detections for {len(detections_by_page)} pages. No annotations supplied or matched.")


if __name__ == "__main__":
    main()
