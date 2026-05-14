#!/usr/bin/env python3
"""Evaluate a fine-tuned DocLayout-YOLO baseline on project annotations.

This script deliberately uses the same project-level evaluator as the
YOLOv8n/Faster R-CNN comparisons: predictions are filtered by validation-
selected confidence threshold, then matched to non-ignored TextColumn boxes
with greedy IoU@0.5 matching. The output is intended as a supplementary
baseline because DocLayout-YOLO is pre-trained for modern document-layout
categories and may not transfer well to vertical handwritten archive columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_yolo_rule_gt import assign_reading_order
from evaluate_layout import aggregate_page_metrics, evaluate_page


def load_pages(annotation_path: Path) -> list[dict]:
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("pages", []))


def evaluate_split(pages: list[dict], predictions: dict[str, list[dict]], split: str, threshold: float) -> dict:
    page_metrics: list[dict] = []
    for page in pages:
        if page.get("split") != split:
            continue
        detections = [det for det in predictions.get(page["page_id"], []) if float(det.get("score", 0.0)) >= threshold]
        detections = assign_reading_order(detections, page)
        metric = evaluate_page(detections, page.get("columns", []), iou_threshold=0.5)
        metric.update({"page_id": page["page_id"], "split": split, "method": "doclayout_yolo", "threshold": threshold})
        page_metrics.append(metric)
    return {"summary": aggregate_page_metrics(page_metrics), "pages": page_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    args = parser.parse_args()

    from doclayout_yolo import YOLOv10

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = load_pages(args.annotations)
    model = YOLOv10(str(args.weights))

    predictions: dict[str, list[dict]] = {}
    runtimes: list[float] = []
    for index, page in enumerate(pages, start=1):
        start = time.time()
        result = model.predict(
            source=page["image_path"],
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.7,
            verbose=False,
            device=args.device,
        )[0]
        runtimes.append(time.time() - start)
        detections: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            classes = result.boxes.cls.cpu().numpy().tolist()
            for box, score, cls in zip(boxes, scores, classes):
                detections.append(
                    {
                        "bbox": [float(v) for v in box],
                        "score": float(score),
                        "class": int(cls),
                        "method": "doclayout_yolo",
                    }
                )
        predictions[page["page_id"]] = detections
        print(f"{index}/{len(pages)} {page['page_id']}: {len(detections)} detections")

    prediction_pages = [
        {
            "page_id": page["page_id"],
            "split": page.get("split"),
            "image_path": page.get("image_path"),
            "detections": predictions.get(page["page_id"], []),
        }
        for page in pages
    ]
    (args.output_dir / "doclayout_project_predictions.json").write_text(
        json.dumps({"pages": prediction_pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    for threshold in args.thresholds:
        val_summary = evaluate_split(pages, predictions, "val", threshold)["summary"]
        test_summary = evaluate_split(pages, predictions, "test", threshold)["summary"]
        row = {"threshold": threshold}
        for prefix, summary in (("val", val_summary), ("test", test_summary)):
            for key in (
                "num_predictions",
                "true_positive",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1",
                "mean_iou",
                "reading_order_accuracy",
            ):
                row[f"{prefix}_{key}"] = summary.get(key)
        rows.append(row)

    best = max(rows, key=lambda row: (float(row["val_f1"]), float(row["val_precision"])))
    selected = float(best["threshold"])
    summary = {
        "model": "DocLayout-YOLO DocStructBench-pretrained + project fine-tuning",
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "device": args.device,
        "val_selected_threshold": selected,
        "mean_seconds_per_page": sum(runtimes) / len(runtimes) if runtimes else None,
        "val": evaluate_split(pages, predictions, "val", selected)["summary"],
        "test": evaluate_split(pages, predictions, "test", selected)["summary"],
    }

    with (args.output_dir / "doclayout_threshold_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "doclayout_threshold_scan.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "doclayout_val_selected_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
