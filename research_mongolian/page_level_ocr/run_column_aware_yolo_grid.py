#!/usr/bin/env python3
"""Grid-search a simple column-aware YOLO post-processing diagnostic.

This experiment tests whether validation-selected geometric priors for
vertical text columns (aspect ratio, relative width, relative height) improve
generalization over plain confidence-threshold filtering. It is intentionally
reported as a diagnostic experiment rather than the main method because the
small validation split can overfit these hand-tuned priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_yolo_rule_gt import assign_reading_order, yolo_xywh_to_xyxy
from evaluate_layout import aggregate_page_metrics, evaluate_page
from export_yolo_dataset import safe_stem


def load_annotations(path: Path, split: str) -> dict[str, dict]:
    pages = json.loads(path.read_text(encoding="utf-8")).get("pages", [])
    return {page["page_id"]: page for page in pages if page.get("split") == split}


def load_raw_yolo(path: Path, gt_by_page: dict[str, dict]) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stem_map = {safe_stem(page_id): page_id for page_id in gt_by_page}
    by_page = {page_id: [] for page_id in gt_by_page}
    for item in raw:
        raw_id = str(item.get("image_id"))
        stem = Path(str(item.get("file_name", ""))).stem
        page_id = stem_map.get(raw_id, stem_map.get(stem, raw_id))
        if page_id not in by_page:
            continue
        by_page[page_id].append(
            {
                "bbox": yolo_xywh_to_xyxy(item["bbox"]),
                "score": float(item.get("score", 0.0)),
                "method": "yolo_column_aware",
            }
        )
    return by_page


def filter_detections(
    detections: list[dict],
    page: dict,
    threshold: float,
    min_aspect: float,
    min_height_ratio: float,
    max_width_ratio: float,
    min_width_ratio: float,
) -> list[dict]:
    page_width = float(page["width"])
    page_height = float(page["height"])
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        width = x2 - x1
        height = y2 - y1
        if det["score"] < threshold or width <= 0 or height <= 0:
            continue
        if height / width < min_aspect:
            continue
        if height / page_height < min_height_ratio:
            continue
        if width / page_width > max_width_ratio:
            continue
        if width / page_width < min_width_ratio:
            continue
        filtered.append(dict(det))
    return assign_reading_order(filtered, page)


def evaluate_split(
    raw_by_page: dict[str, list[dict]],
    gt_by_page: dict[str, dict],
    params: tuple[float, float, float, float, float],
) -> dict:
    metrics = []
    for page_id, page in gt_by_page.items():
        detections = filter_detections(raw_by_page.get(page_id, []), page, *params)
        metric = evaluate_page(detections, page.get("columns", []), iou_threshold=0.5)
        metric.update({"page_id": page_id})
        metrics.append(metric)
    return aggregate_page_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    val_gt = load_annotations(args.annotations, "val")
    test_gt = load_annotations(args.annotations, "test")
    val_raw = load_raw_yolo(args.val_predictions, val_gt)
    test_raw = load_raw_yolo(args.test_predictions, test_gt)

    candidates: list[tuple[float, float, float, float, tuple[float, float, float, float, float], dict]] = []
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for min_aspect in [2, 3, 4, 5, 6, 8]:
            for min_height_ratio in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
                for max_width_ratio in [0.06, 0.08, 0.10, 0.12, 0.16]:
                    for min_width_ratio in [0.005, 0.01, 0.015, 0.02]:
                        params = (threshold, min_aspect, min_height_ratio, max_width_ratio, min_width_ratio)
                        val_summary = evaluate_split(val_raw, val_gt, params)
                        candidates.append(
                            (
                                float(val_summary["f1"]),
                                float(val_summary["precision"]),
                                float(val_summary.get("reading_order_accuracy") or 0.0),
                                float(val_summary["recall"]),
                                params,
                                val_summary,
                            )
                        )
    candidates.sort(reverse=True, key=lambda row: (row[0], row[1], row[2], row[3]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for val_f1, val_precision, val_ro, val_recall, params, _ in candidates[:30]:
        test_summary = evaluate_split(test_raw, test_gt, params)
        threshold, min_aspect, min_height_ratio, max_width_ratio, min_width_ratio = params
        rows.append(
            {
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_ro": val_ro,
                "threshold": threshold,
                "min_aspect": min_aspect,
                "min_height_ratio": min_height_ratio,
                "max_width_ratio": max_width_ratio,
                "min_width_ratio": min_width_ratio,
                **{f"test_{key}": value for key, value in test_summary.items() if key in {"num_predictions", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1", "mean_iou", "reading_order_accuracy"}},
            }
        )

    with (args.output_dir / "column_aware_grid_top.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_params = candidates[0][4]
    summary = {
        "selected_by": "validation F1, then precision, then reading-order accuracy",
        "params": {
            "threshold": best_params[0],
            "min_aspect": best_params[1],
            "min_height_ratio": best_params[2],
            "max_width_ratio": best_params[3],
            "min_width_ratio": best_params[4],
        },
        "val": evaluate_split(val_raw, val_gt, best_params),
        "test": evaluate_split(test_raw, test_gt, best_params),
    }
    (args.output_dir / "column_aware_val_selected_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
