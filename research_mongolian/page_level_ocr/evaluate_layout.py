#!/usr/bin/env python3
"""Evaluate page-level column detection and reading order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    pred_idx: int
    gt_idx: int
    iou: float


def greedy_match(pred_boxes: list[list[float]], gt_boxes: list[list[float]], iou_threshold: float) -> list[Match]:
    candidates: list[Match] = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = bbox_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append(Match(pi, gi, iou))
    candidates.sort(key=lambda m: m.iou, reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[Match] = []
    for match in candidates:
        if match.pred_idx in used_pred or match.gt_idx in used_gt:
            continue
        used_pred.add(match.pred_idx)
        used_gt.add(match.gt_idx)
        matches.append(match)
    return matches


def pairwise_order_accuracy(matches: Iterable[Match], pred_orders: list[int], gt_orders: list[int]) -> float | None:
    matched = list(matches)
    if len(matched) < 2:
        return None
    total = 0
    correct = 0
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            a, b = matched[i], matched[j]
            pred_relation = pred_orders[a.pred_idx] < pred_orders[b.pred_idx]
            gt_relation = gt_orders[a.gt_idx] < gt_orders[b.gt_idx]
            total += 1
            if pred_relation == gt_relation:
                correct += 1
    return correct / total if total else None


def evaluate_page(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> dict:
    gt_valid = [gt for gt in ground_truth if not gt.get("ignore", False)]
    gt_ignore = [gt for gt in ground_truth if gt.get("ignore", False)]
    filtered_predictions = []
    for pred in predictions:
        if any(bbox_iou(pred["bbox"], ignore["bbox"]) >= 0.3 for ignore in gt_ignore):
            continue
        filtered_predictions.append(pred)
    pred_boxes = [pred["bbox"] for pred in filtered_predictions]
    gt_boxes = [gt["bbox"] for gt in gt_valid]
    matches = greedy_match(pred_boxes, gt_boxes, iou_threshold)

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(m.iou for m in matches) / tp if tp else 0.0

    pred_orders = [int(pred.get("reading_order", idx)) for idx, pred in enumerate(filtered_predictions)]
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
        "mean_iou": mean_iou,
        "reading_order_accuracy": order_acc,
        "matches": [match.__dict__ for match in matches],
    }


def aggregate_page_metrics(page_metrics: list[dict]) -> dict:
    totals = {
        "num_pages": len(page_metrics),
        "num_predictions": 0,
        "num_ground_truth": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    ious: list[float] = []
    order_accs: list[float] = []

    for metric in page_metrics:
        for key in totals:
            if key == "num_pages":
                continue
            totals[key] += int(metric.get(key, 0))
        for match in metric.get("matches", []):
            ious.append(float(match["iou"]))
        if metric.get("reading_order_accuracy") is not None:
            order_accs.append(float(metric["reading_order_accuracy"]))

    tp, fp, fn = totals["true_positive"], totals["false_positive"], totals["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    totals.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": sum(ious) / len(ious) if ious else 0.0,
            "reading_order_accuracy": sum(order_accs) / len(order_accs) if order_accs else None,
        }
    )
    return totals
