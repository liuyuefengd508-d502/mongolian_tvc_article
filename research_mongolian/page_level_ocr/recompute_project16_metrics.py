#!/usr/bin/env python3
"""Recompute layout and reading-order regression metrics after Project 16 GT merge.

Outputs are intentionally self-contained so the paper directory can archive the
post-reading-order-review numbers used in response documents and LaTeX tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path
from statistics import mean

from compare_yolo_rule_gt import (
    assign_reading_order,
    load_annotations,
    load_rule_detections,
    load_yolo_predictions,
)
from evaluate_layout import aggregate_page_metrics, bbox_iou, evaluate_page, greedy_match, pairwise_order_accuracy
from export_yolo_dataset import safe_stem

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "project16_recomputed_metrics_20260512"
DEFAULT_PAPER_DIR = Path("/Users/liuyu/Desktop/mydocuments/codes/claudcodeTest/page_layout_no_transcription_overleaf")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_faster_predictions(raw: dict[str, list[dict]], score_threshold: float) -> dict[str, list[dict]]:
    by_page: dict[str, list[dict]] = {}
    for page_id, dets in raw.items():
        kept = []
        for det in dets:
            if float(det.get("score", 0.0)) < score_threshold:
                continue
            kept.append({
                "bbox": [float(v) for v in det["bbox"]],
                "score": float(det.get("score", 0.0)),
                "method": det.get("method", "fasterrcnn_mobilenet"),
                "orientation": det.get("orientation", "correct"),
            })
        by_page[page_id] = kept
    return by_page


def with_order(detections: dict[str, list[dict]], gt_by_page: dict[str, dict], rotation_aware: bool = True) -> dict[str, list[dict]]:
    ordered: dict[str, list[dict]] = {}
    for page_id, page in gt_by_page.items():
        dets = [dict(d) for d in detections.get(page_id, [])]
        if rotation_aware:
            ordered[page_id] = assign_reading_order(dets, page)
        else:
            dets.sort(key=lambda det: ((det["bbox"][0] + det["bbox"][2]) / 2.0, det["bbox"][1]))
            for idx, det in enumerate(dets):
                det["reading_order"] = idx
            ordered[page_id] = dets
    return ordered


def evaluate_page_no_ignore_filter(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> dict:
    gt_valid = [gt for gt in ground_truth if not gt.get("ignore", False)]
    pred_boxes = [pred["bbox"] for pred in predictions]
    gt_boxes = [gt["bbox"] for gt in gt_valid]
    matches = greedy_match(pred_boxes, gt_boxes, iou_threshold)
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_orders = [int(pred.get("reading_order", idx)) for idx, pred in enumerate(predictions)]
    gt_orders = [int(gt.get("reading_order", idx)) for idx, gt in enumerate(gt_valid)]
    order_acc = pairwise_order_accuracy(matches, pred_orders, gt_orders)
    return {
        "num_predictions": len(pred_boxes),
        "num_ground_truth": len(gt_boxes),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": sum(m.iou for m in matches) / tp if tp else 0.0,
        "reading_order_accuracy": order_acc,
        "matches": [m.__dict__ for m in matches],
    }


def evaluate_detector(
    name: str,
    detections: dict[str, list[dict]],
    gt_by_page: dict[str, dict],
    iou_threshold: float = 0.5,
    ignore_filter: bool = True,
) -> dict:
    page_metrics = []
    for page_id, page in gt_by_page.items():
        dets = detections.get(page_id, [])
        if ignore_filter:
            metric = evaluate_page(dets, page.get("columns", []), iou_threshold=iou_threshold)
        else:
            metric = evaluate_page_no_ignore_filter(dets, page.get("columns", []), iou_threshold=iou_threshold)
        metric.update({"page_id": page_id, "method": name, "split": page.get("split"), "iou_threshold": iou_threshold})
        page_metrics.append(metric)
    return {"summary": aggregate_page_metrics(page_metrics), "pages": page_metrics}


def bootstrap_ci(page_metrics: list[dict], n: int = 10000, seed: int = 20260512) -> dict:
    rng = random.Random(seed)
    vals = {"precision": [], "recall": [], "f1": []}
    m = len(page_metrics)
    for _ in range(n):
        sample = [page_metrics[rng.randrange(m)] for _ in range(m)]
        agg = aggregate_page_metrics(sample)
        for k in vals:
            vals[k].append(agg[k])
    out = {}
    for k, arr in vals.items():
        arr.sort()
        lo = arr[int(0.025 * (n - 1))]
        hi = arr[int(0.975 * (n - 1))]
        out[f"{k}_ci95_low"] = lo
        out[f"{k}_ci95_high"] = hi
    return out


def row_from_summary(method: str, summary: dict, extra: dict | None = None) -> dict:
    keys = [
        "num_pages", "num_predictions", "num_ground_truth", "true_positive", "false_positive", "false_negative",
        "precision", "recall", "f1", "mean_iou", "reading_order_accuracy",
    ]
    row = {"method": method}
    if extra:
        row.update(extra)
    row.update({k: summary.get(k) for k in keys})
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pr_sweep(name: str, raw_loader, thresholds: list[float], gt_by_page: dict[str, dict], iou_threshold: float = 0.5) -> list[dict]:
    rows = []
    for thr in thresholds:
        dets = with_order(raw_loader(thr), gt_by_page, rotation_aware=True)
        ev = evaluate_detector(name, dets, gt_by_page, iou_threshold=iou_threshold, ignore_filter=True)
        rows.append(row_from_summary(name, ev["summary"], {"score_threshold": thr, "iou_threshold": iou_threshold}))
    return rows


def approximate_pr_auc(rows: list[dict]) -> float:
    """Trapezoidal area under project-level precision-recall points."""
    pts = sorted((float(r["recall"]), float(r["precision"])) for r in rows)
    if len(pts) < 2:
        return 0.0
    return sum((r2 - r1) * (p1 + p2) / 2.0 for (r1, p1), (r2, p2) in zip(pts, pts[1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, default=ROOT / "page_level_annotations.json")
    ap.add_argument("--rule-detections", type=Path, default=ROOT / "results/layout_test_final_after_test_merge/detections_proposed.json")
    ap.add_argument("--yolo-predictions", type=Path, default=ROOT / "results/yolo_column_detector/yolov8n_50ep_train43_test/predictions.json")
    ap.add_argument("--faster-predictions", type=Path, default=ROOT / "results/major_revision_support/fasterrcnn_ap_sweep/fasterrcnn_test_predictions.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    ap.add_argument("--yolo-threshold", type=float, default=0.35)
    ap.add_argument("--faster-threshold", type=float, default=0.80)
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_page = load_annotations(args.annotations, "test")
    page_id_by_stem = {safe_stem(page_id): page_id for page_id in gt_by_page}

    rule_by_page = load_rule_detections(args.rule_detections)
    yolo_loader = lambda thr: load_yolo_predictions(args.yolo_predictions, thr, page_id_by_stem)
    faster_raw = load_json(args.faster_predictions)
    faster_loader = lambda thr: normalize_faster_predictions(faster_raw, thr)

    yolo_by_page = with_order(yolo_loader(args.yolo_threshold), gt_by_page, rotation_aware=True)
    faster_by_page = with_order(faster_loader(args.faster_threshold), gt_by_page, rotation_aware=True)

    main_results = {
        "Rule proposed@0.5": evaluate_detector("Rule proposed", rule_by_page, gt_by_page, 0.5),
        "YOLOv8n + rotation-aware order@0.5": evaluate_detector("YOLOv8n + rotation-aware order", yolo_by_page, gt_by_page, 0.5),
        "Faster R-CNN@0.5": evaluate_detector("Faster R-CNN", faster_by_page, gt_by_page, 0.5),
        "Rule proposed@0.75": evaluate_detector("Rule proposed", rule_by_page, gt_by_page, 0.75),
        "YOLOv8n + rotation-aware order@0.75": evaluate_detector("YOLOv8n + rotation-aware order", yolo_by_page, gt_by_page, 0.75),
        "Faster R-CNN@0.75": evaluate_detector("Faster R-CNN", faster_by_page, gt_by_page, 0.75),
    }
    (args.out_dir / "strict_iou_metrics_project16.json").write_text(json.dumps(main_results, ensure_ascii=False, indent=2), encoding="utf-8")
    strict_rows = []
    for key, ev in main_results.items():
        method, thr_s = key.rsplit("@", 1)
        strict_rows.append(row_from_summary(method, ev["summary"], {"iou_threshold": float(thr_s)}))
    write_csv(args.out_dir / "strict_iou_metrics_project16.csv", strict_rows)

    boot_rows = []
    for key in ["Rule proposed@0.5", "YOLOv8n + rotation-aware order@0.5", "Faster R-CNN@0.5"]:
        ev = main_results[key]
        method = key.rsplit("@", 1)[0]
        summ = ev["summary"]
        row = {"method": method, "precision": summ["precision"], "recall": summ["recall"], "f1": summ["f1"]}
        row.update(bootstrap_ci(ev["pages"], n=args.bootstrap_n))
        boot_rows.append(row)
    (args.out_dir / "bootstrap_ci_summary_project16.json").write_text(json.dumps(boot_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "bootstrap_ci_summary_project16.csv", boot_rows)

    # Ablation: YOLO ignore filtering and rotation-aware order assignment.
    ablation_rows = []
    for ignore_filter in [True, False]:
        for rotation_aware in [True, False]:
            dets = with_order(yolo_loader(args.yolo_threshold), gt_by_page, rotation_aware=rotation_aware)
            ev = evaluate_detector("YOLOv8n", dets, gt_by_page, 0.5, ignore_filter=ignore_filter)
            ablation_rows.append(row_from_summary("YOLOv8n", ev["summary"], {
                "ignore_filter": ignore_filter,
                "rotation_aware_order": rotation_aware,
            }))
    (args.out_dir / "ablation_ignore_order_project16.json").write_text(json.dumps(ablation_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "ablation_ignore_order_project16.csv", ablation_rows)

    thresholds = [round(x / 100, 2) for x in range(0, 96, 5)]
    sweep_rows = []
    ap_summary = {}
    for detector_name, loader in [("YOLOv8n + rotation-aware order", yolo_loader), ("Faster R-CNN", faster_loader)]:
        for iou_thr in [0.5, 0.75]:
            rows = pr_sweep(detector_name, loader, thresholds, gt_by_page, iou_threshold=iou_thr)
            sweep_rows.extend(rows)
            ap_summary[f"{detector_name}@{iou_thr}"] = approximate_pr_auc(rows)
    (args.out_dir / "project_pr_ap_sweep_project16.json").write_text(json.dumps(sweep_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "project_ap_summary_project16.json").write_text(json.dumps(ap_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "project_pr_ap_sweep_project16.csv", sweep_rows)

    summary = {
        "source_annotations": str(args.annotations),
        "num_test_pages": len(gt_by_page),
        "yolo_score_threshold": args.yolo_threshold,
        "faster_score_threshold": args.faster_threshold,
        "main_iou05": {k: v["summary"] for k, v in main_results.items() if k.endswith("@0.5")},
        "strict_iou075": {k: v["summary"] for k, v in main_results.items() if k.endswith("@0.75")},
        "bootstrap_ci95": boot_rows,
        "yolo_ablation": ablation_rows,
        "project_pr_auc_trapezoidal": ap_summary,
    }
    (args.out_dir / "project16_recomputed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Project 16 merged GT regression metrics (2026-05-12)",
        "",
        f"Annotations: `{args.annotations}`",
        f"Test pages: {len(gt_by_page)}; YOLO score threshold: {args.yolo_threshold}; Faster R-CNN score threshold: {args.faster_threshold}.",
        "",
        "## Main metrics (IoU@0.5)",
        "",
        "| Method | Pred | GT | TP | FP | FN | P | R | F1 | mIoU | RO Acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["Rule proposed@0.5", "YOLOv8n + rotation-aware order@0.5", "Faster R-CNN@0.5"]:
        s = main_results[key]["summary"]
        method = key.rsplit("@", 1)[0]
        ro = s["reading_order_accuracy"]
        md_lines.append(f"| {method} | {s['num_predictions']} | {s['num_ground_truth']} | {s['true_positive']} | {s['false_positive']} | {s['false_negative']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['mean_iou']:.3f} | {ro:.3f} |")
    md_lines.extend(["", "## Strict IoU@0.75", "", "| Method | P | R | F1 | mIoU | RO Acc. |", "|---|---:|---:|---:|---:|---:|"])
    for key in ["Rule proposed@0.75", "YOLOv8n + rotation-aware order@0.75", "Faster R-CNN@0.75"]:
        s = main_results[key]["summary"]
        method = key.rsplit("@", 1)[0]
        ro = s["reading_order_accuracy"]
        md_lines.append(f"| {method} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['mean_iou']:.3f} | {ro:.3f} |")
    md_lines.extend(["", "## Bootstrap 95% CI (IoU@0.5)", "", "| Method | P 95% CI | R 95% CI | F1 95% CI |", "|---|---:|---:|---:|"])
    for r in boot_rows:
        md_lines.append(f"| {r['method']} | [{r['precision_ci95_low']:.3f}, {r['precision_ci95_high']:.3f}] | [{r['recall_ci95_low']:.3f}, {r['recall_ci95_high']:.3f}] | [{r['f1_ci95_low']:.3f}, {r['f1_ci95_high']:.3f}] |")
    (args.out_dir / "project16_recomputed_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if args.paper_dir.exists():
        for p in args.out_dir.glob("*project16*"):
            shutil.copy2(p, args.paper_dir / p.name)
        shutil.copy2(args.out_dir / "project16_recomputed_metrics.md", args.paper_dir / "Project16_Recomputed_Regression_Metrics_2026-05-12.md")

    print(json.dumps(summary["main_iou05"], ensure_ascii=False, indent=2))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
