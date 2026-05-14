#!/usr/bin/env python3
"""Compute supplementary page-level and reading-order metrics.

The main paper reports detection P/R/F1 and pairwise reading-order accuracy.
This script adds stricter page-level metrics requested by reviewers:

* page-level complete detection rate;
* page-level full success rate (complete detection + perfect order);
* Kendall's tau and Spearman's rho over matched column orders;
* grouped performance by page complexity/orientation/noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_yolo_rule_gt import assign_reading_order, load_rule_detections, load_yolo_predictions
from evaluate_layout import aggregate_page_metrics, evaluate_page
from export_yolo_dataset import safe_stem


def corr_spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def corr_kendall(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            sy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            prod = sx * sy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def read_annotations(path: Path, split: str) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {page["page_id"]: page for page in data.get("pages", []) if page.get("split") == split}


def page_group(page: dict) -> list[str]:
    valid = [c for c in page.get("columns", []) if not c.get("ignore", False)]
    ignore = [c for c in page.get("columns", []) if c.get("ignore", False)]
    groups = []
    n = len(valid)
    if n <= 5:
        groups.append("few_columns_<=5")
    elif n <= 15:
        groups.append("medium_columns_6_15")
    else:
        groups.append("many_columns_>15")
    groups.append("rotated_pages" if "rot" in str(page.get("image_path", "")).lower() else "ordinary_pages")
    groups.append("has_ignore" if ignore else "no_ignore")
    return groups


def enrich_page_metric(metric: dict, detections: list[dict], page: dict) -> dict:
    valid_gt = [gt for gt in page.get("columns", []) if not gt.get("ignore", False)]
    matched = metric.get("matches", [])
    pred_orders = [int(det.get("reading_order", idx)) for idx, det in enumerate(detections)]
    gt_orders = [int(gt.get("reading_order", idx)) for idx, gt in enumerate(valid_gt)]
    matched_pred_orders = [pred_orders[m["pred_idx"]] for m in matched]
    matched_gt_orders = [gt_orders[m["gt_idx"]] for m in matched]
    exact_detection = metric["true_positive"] == len(valid_gt) and metric["false_positive"] == 0 and metric["false_negative"] == 0
    exact_order = metric.get("reading_order_accuracy") == 1.0 if len(matched) >= 2 else exact_detection
    metric.update(
        {
            "complete_detection": bool(exact_detection),
            "full_page_success": bool(exact_detection and exact_order),
            "kendall_tau": corr_kendall(matched_pred_orders, matched_gt_orders),
            "spearman_rho": corr_spearman(matched_pred_orders, matched_gt_orders),
            "num_ignore": sum(1 for c in page.get("columns", []) if c.get("ignore", False)),
        }
    )
    return metric


def aggregate_enhanced(metrics: list[dict]) -> dict:
    base = aggregate_page_metrics(metrics)
    vals_tau = [m["kendall_tau"] for m in metrics if m.get("kendall_tau") is not None]
    vals_rho = [m["spearman_rho"] for m in metrics if m.get("spearman_rho") is not None]
    base.update(
        {
            "complete_detection_rate": sum(1 for m in metrics if m.get("complete_detection")) / len(metrics) if metrics else 0,
            "full_page_success_rate": sum(1 for m in metrics if m.get("full_page_success")) / len(metrics) if metrics else 0,
            "mean_kendall_tau": sum(vals_tau) / len(vals_tau) if vals_tau else None,
            "mean_spearman_rho": sum(vals_rho) / len(vals_rho) if vals_rho else None,
        }
    )
    return base


def evaluate_method(method: str, detections_by_page: dict[str, list[dict]], gt_by_page: dict[str, dict]) -> dict:
    metrics = []
    for page_id, page in gt_by_page.items():
        dets = detections_by_page.get(page_id, [])
        dets = assign_reading_order(dets, page)
        metric = evaluate_page(dets, page.get("columns", []), iou_threshold=0.5)
        metric.update({"page_id": page_id, "method": method})
        metrics.append(enrich_page_metric(metric, dets, page))
    group_metrics: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for metric in metrics:
        for group in page_group(gt_by_page[metric["page_id"]]):
            grouped[group].append(metric)
    for group, items in grouped.items():
        group_metrics[group] = aggregate_enhanced(items)
    return {"summary": aggregate_enhanced(metrics), "pages": metrics, "groups": group_metrics}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rule-detections", type=Path, required=True)
    parser.add_argument("--yolo-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-threshold", type=float, default=0.35)
    args = parser.parse_args()

    gt_by_page = read_annotations(args.annotations, args.split)
    rule_by_page = load_rule_detections(args.rule_detections)
    page_id_by_stem = {safe_stem(page_id): page_id for page_id in gt_by_page}
    yolo_by_page = load_yolo_predictions(args.yolo_predictions, args.yolo_threshold, page_id_by_stem)

    results = {
        "rule_proposed": evaluate_method("rule_proposed", rule_by_page, gt_by_page),
        "yolov8n": evaluate_method("yolov8n", yolo_by_page, gt_by_page),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "enhanced_layout_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    for method, result in results.items():
        row = {"method": method}
        row.update(result["summary"])
        summary_rows.append(row)
    write_csv(args.output_dir / "enhanced_layout_summary.csv", summary_rows)

    group_rows = []
    for method, result in results.items():
        for group, summary in result["groups"].items():
            row = {"method": method, "group": group}
            row.update(summary)
            group_rows.append(row)
    write_csv(args.output_dir / "enhanced_layout_groups.csv", group_rows)

    print(json.dumps({"summary": summary_rows, "groups_csv": str(args.output_dir / "enhanced_layout_groups.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
