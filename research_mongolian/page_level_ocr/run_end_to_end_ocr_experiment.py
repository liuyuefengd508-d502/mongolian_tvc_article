#!/usr/bin/env python3
"""Run page-level end-to-end OCR pipeline experiments.

The script builds oracle and detected text-column crops from page-level annotations
and YOLO predictions, optionally runs an existing CRNN/MSSE recognizer checkpoint,
and writes page/column-level JSON/CSV outputs plus detection/error-propagation
summaries. If column transcripts are unavailable, CER/WER are reported as not
computable while all pipeline outputs are still generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    F = None


DEFAULT_CHARSET_27 = "abcdefghijklmnopqrstuvwxyz "


def safe_stem(page_id: str) -> str:
    return page_id.replace("/", "_").replace("\\", "_").replace("(", "_").replace(")", "_").replace(" ", "_")


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float | None:
    return edit_distance(pred, gt) / len(gt) if gt else None


def wer(pred: str, gt: str) -> float | None:
    gt_words = gt.split()
    pred_words = pred.split()
    return edit_distance("\u0001".join(pred_words), "\u0001".join(gt_words)) / len(gt_words) if gt_words else None


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


def greedy_match(pred_boxes: list[list[float]], gt_boxes: list[list[float]], threshold: float = 0.5) -> list[dict[str, Any]]:
    candidates = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = bbox_iou(pred, gt)
            if iou >= threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"pred_idx": pi, "gt_idx": gi, "iou": iou})
    return matches


def pairwise_order_accuracy(matches: list[dict[str, Any]], pred_orders: list[int], gt_orders: list[int]) -> float | None:
    if len(matches) < 2:
        return None
    total = correct = 0
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            a, b = matches[i], matches[j]
            total += 1
            if (pred_orders[a["pred_idx"]] < pred_orders[b["pred_idx"]]) == (gt_orders[a["gt_idx"]] < gt_orders[b["gt_idx"]]):
                correct += 1
    return correct / total if total else None


def aggregate_detection(page_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"num_pages": len(page_metrics), "num_predictions": 0, "num_ground_truth": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
    ious, ro_accs = [], []
    for m in page_metrics:
        for k in ["num_predictions", "num_ground_truth", "true_positive", "false_positive", "false_negative"]:
            totals[k] += int(m[k])
        ious.extend(match["iou"] for match in m.get("matches", []))
        if m.get("reading_order_accuracy") is not None:
            ro_accs.append(float(m["reading_order_accuracy"]))
    tp, fp, fn = totals["true_positive"], totals["false_positive"], totals["false_negative"]
    totals.update({
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "reading_order_accuracy": sum(ro_accs) / len(ro_accs) if ro_accs else None,
    })
    return totals


def evaluate_detection(preds: list[dict[str, Any]], columns: list[dict[str, Any]]) -> dict[str, Any]:
    gt_valid = [c for c in columns if not c.get("ignore")]
    gt_ignore = [c for c in columns if c.get("ignore")]
    filtered_preds = []
    for pred in preds:
        if any(bbox_iou(pred["bbox"], ign["bbox"]) >= 0.3 for ign in gt_ignore):
            continue
        filtered_preds.append(pred)
    pred_boxes = [p["bbox"] for p in filtered_preds]
    gt_boxes = [g["bbox"] for g in gt_valid]
    matches = greedy_match(pred_boxes, gt_boxes, threshold=0.5)
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    pred_orders = [int(p.get("reading_order", i)) for i, p in enumerate(filtered_preds)]
    gt_orders = [int(g.get("reading_order", i)) for i, g in enumerate(gt_valid)]
    return {
        "num_predictions": len(pred_boxes),
        "num_ground_truth": len(gt_boxes),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_iou": sum(m["iou"] for m in matches) / tp if tp else 0.0,
        "reading_order_accuracy": pairwise_order_accuracy(matches, pred_orders, gt_orders),
        "matches": matches,
    }


if torch is not None:
    class MSSEModule(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv_r1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
            self.conv_r2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
            self.conv_r4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
            self.gate_conv = nn.Conv2d(channels * 3, channels, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            f1 = F.relu(self.conv_r1(x))
            f2 = F.relu(self.conv_r2(x))
            f4 = F.relu(self.conv_r4(x))
            return x * self.sigmoid(self.gate_conv(torch.cat([f1, f2, f4], dim=1)))

    class CRNN(nn.Module):
        def __init__(self, nclass: int, nh: int = 256):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
            self.enhancer = MSSEModule(64)
            self.cnn = nn.Sequential(
                nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            )
            self.rnn = nn.LSTM(512, nh, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(nh * 2, nclass)

        def forward(self, x):
            x = self.conv1(x)
            x = self.enhancer(x)
            conv = self.cnn(x)
            conv = conv.squeeze(2).permute(0, 2, 1)
            rnn_out, _ = self.rnn(conv)
            return self.fc(rnn_out)
else:
    CRNN = None


@dataclass
class RecognizerStatus:
    enabled: bool
    checkpoint: str | None
    status: str
    error: str | None = None


class Recognizer:
    def __init__(self, checkpoint_candidates: list[Path], charset: str, device: str = "auto", crop_height: int = 32, max_width: int = 512):
        self.charset = charset
        self.idx_to_char = {i + 1: c for i, c in enumerate(charset)}
        self.crop_height = crop_height
        self.max_width = max_width
        self.model = None
        self.device = "cpu"
        self.status = RecognizerStatus(False, None, "torch_unavailable" if torch is None else "not_loaded")
        if torch is None:
            return
        self.device = "mps" if device == "auto" and torch.backends.mps.is_available() else ("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        for ckpt in checkpoint_candidates:
            if not ckpt.exists():
                continue
            try:
                state = torch.load(str(ckpt), map_location="cpu")
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                nclass = int(state["fc.bias"].shape[0])
                if len(self.charset) != nclass - 1:
                    # keep deterministic fallback labels if exact charset is unknown
                    self.charset = "".join(chr(0x2460 + i) for i in range(nclass - 1))
                    self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}
                model = CRNN(nclass=nclass, nh=256)
                load_info = model.load_state_dict(state, strict=False)
                blocking_missing = [k for k in load_info.missing_keys if not k.startswith("domain_classifier")]
                if blocking_missing:
                    raise RuntimeError(f"Missing required recognizer keys: {blocking_missing}; unexpected={load_info.unexpected_keys}")
                model.eval().to(self.device)
                self.model = model
                status = "loaded" if not load_info.unexpected_keys else f"loaded_ignored_unexpected:{','.join(load_info.unexpected_keys[:6])}"
                self.status = RecognizerStatus(True, str(ckpt), status)
                return
            except Exception as exc:
                self.status = RecognizerStatus(False, str(ckpt), "load_failed", repr(exc))
        if self.status.status == "not_loaded":
            self.status = RecognizerStatus(False, None, "no_checkpoint_found")

    def preprocess(self, crop: Image.Image) -> Any:
        image = ImageOps.grayscale(crop)
        if image.height > image.width:
            image = image.rotate(90, expand=True)
        scale = self.crop_height / max(1, image.height)
        new_w = max(16, min(self.max_width, int(round(image.width * scale))))
        image = image.resize((new_w, self.crop_height), Image.BILINEAR)
        arr = np.asarray(image).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def decode(self, indices: Any) -> str:
        chars, prev = [], -1
        for item in indices:
            val = int(item)
            if val != prev and val != 0:
                chars.append(self.idx_to_char.get(val, ""))
            prev = val
        return "".join(chars)

    def predict(self, crop: Image.Image) -> str:
        if self.model is None:
            return ""
        with torch.no_grad():
            logits = self.model(self.preprocess(crop))
            pred = logits.argmax(2)[0].detach().cpu().tolist()
        return self.decode(pred)


def load_annotations(path: Path, split: str) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["page_id"]: p for p in data.get("pages", []) if p.get("split") == split}


def load_yolo_predictions(path: Path, page_ids: list[str], threshold: float) -> dict[str, list[dict[str, Any]]]:
    mapping = {safe_stem(pid): pid for pid in page_ids}
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_page: dict[str, list[dict[str, Any]]] = {pid: [] for pid in page_ids}
    for item in raw:
        score = float(item.get("score", 0.0))
        if score < threshold:
            continue
        raw_id = str(item.get("image_id"))
        file_stem = Path(str(item.get("file_name", ""))).stem
        page_id = mapping.get(raw_id, mapping.get(file_stem, raw_id))
        if page_id not in by_page:
            continue
        x, y, w, h = [float(v) for v in item["bbox"]]
        by_page[page_id].append({"bbox": [x, y, x + w, y + h], "score": score, "method": "yolov8n"})
    return by_page


def assign_reading_order(detections: list[dict[str, Any]], page: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = [dict(d) for d in detections]
    if "rot90" in str(page.get("image_path", "")).lower():
        ordered.sort(key=lambda d: (-(d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][1]))
    else:
        ordered.sort(key=lambda d: ((d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][1]))
    for i, det in enumerate(ordered):
        det["reading_order"] = i
    return ordered


def clamp_bbox(bbox: list[float], width: int, height: int, pad: int = 0) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(math.floor(x1 - pad))))
    y1 = max(0, min(height, int(math.floor(y1 - pad))))
    x2 = max(0, min(width, int(math.ceil(x2 + pad))))
    y2 = max(0, min(height, int(math.ceil(y2 + pad))))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def make_oracle_detections(page: dict[str, Any]) -> list[dict[str, Any]]:
    cols = [c for c in page.get("columns", []) if not c.get("ignore")]
    cols.sort(key=lambda c: int(c.get("reading_order", 0)))
    out = []
    for i, c in enumerate(cols):
        out.append({"bbox": [float(v) for v in c["bbox"]], "reading_order": i, "column_id": c.get("column_id"), "transcript": c.get("transcript", ""), "method": "oracle"})
    return out


def run_condition(name: str, pages: dict[str, dict[str, Any]], detections_by_page: dict[str, list[dict[str, Any]]], recognizer: Recognizer, out_dir: Path, pad: int, save_crops: bool) -> dict[str, Any]:
    rows, page_results, metrics_pages = [], [], []
    crop_dir = out_dir / "crops" / name
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    cer_vals, wer_vals = [], []
    transcript_count = 0
    for page_id, page in pages.items():
        image = Image.open(page["image_path"]).convert("RGB")
        width, height = image.size
        detections = detections_by_page.get(page_id, [])
        metric = evaluate_detection(detections, page.get("columns", []))
        metric.update({"page_id": page_id, "condition": name})
        metrics_pages.append(metric)
        col_results = []
        for det in detections:
            bbox = clamp_bbox(det["bbox"], width, height, pad=pad)
            if bbox is None:
                continue
            crop = image.crop(tuple(bbox))
            pred_text = recognizer.predict(crop) if recognizer.status.enabled else ""
            best_gt, best_iou, gt_text = None, 0.0, ""
            for col in page.get("columns", []):
                if col.get("ignore"):
                    continue
                iou = bbox_iou(det["bbox"], col["bbox"])
                if iou > best_iou:
                    best_iou, best_gt, gt_text = iou, col.get("column_id"), col.get("transcript", "")
            col_cer = cer(pred_text, gt_text) if gt_text else None
            col_wer = wer(pred_text, gt_text) if gt_text else None
            if col_cer is not None:
                cer_vals.append(col_cer)
                transcript_count += 1
            if col_wer is not None:
                wer_vals.append(col_wer)
            crop_path = ""
            if save_crops:
                crop_path = str(crop_dir / f"{safe_stem(page_id)}_{int(det.get('reading_order', 0)):03d}.jpg")
                crop.save(crop_path, quality=92)
            rec = {
                "page_id": page_id,
                "condition": name,
                "reading_order": int(det.get("reading_order", 0)),
                "bbox": bbox,
                "score": det.get("score"),
                "matched_gt_column_id": best_gt,
                "matched_iou": best_iou,
                "gt_transcript": gt_text,
                "pred_text": pred_text,
                "cer": col_cer,
                "wer": col_wer,
                "crop_path": crop_path,
            }
            rows.append(rec)
            col_results.append(rec)
        page_text_pred = "\n".join(r["pred_text"] for r in sorted(col_results, key=lambda r: r["reading_order"]))
        page_text_gt = "\n".join(c.get("transcript", "") for c in sorted([c for c in page.get("columns", []) if not c.get("ignore")], key=lambda c: int(c.get("reading_order", 0))) if c.get("transcript", ""))
        page_results.append({"page_id": page_id, "condition": name, "num_columns": len(col_results), "pred_text": page_text_pred, "gt_text": page_text_gt, "detection_metrics": metric})
    summary = {
        "condition": name,
        "recognition_status": "computed" if transcript_count else "not_computed_no_transcripts",
        "mean_cer": sum(cer_vals) / len(cer_vals) if cer_vals else None,
        "mean_wer": sum(wer_vals) / len(wer_vals) if wer_vals else None,
        "num_transcribed_columns": transcript_count,
        "num_predicted_columns": len(rows),
        "detection_summary": aggregate_detection(metrics_pages),
    }
    return {"summary": summary, "pages": page_results, "columns": rows, "detection_pages": metrics_pages}


def write_condition_outputs(result: dict[str, Any], out_dir: Path) -> None:
    condition = result["summary"]["condition"]
    (out_dir / f"{condition}_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / f"{condition}_columns.csv"
    fields = ["page_id", "condition", "reading_order", "bbox", "score", "matched_gt_column_id", "matched_iou", "gt_transcript", "pred_text", "cer", "wer", "crop_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["columns"]:
            writer.writerow({k: row.get(k) for k in fields})


def write_summary_md(results: list[dict[str, Any]], recognizer: Recognizer, out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# 页面级端到端 OCR 实验结果", "",
        "## 运行设置", "",
        f"- split: `{args.split}`",
        f"- YOLO threshold: `{args.score_threshold}`",
        f"- crop padding: `{args.crop_padding}`",
        f"- recognizer status: `{recognizer.status.status}`",
        f"- recognizer checkpoint: `{recognizer.status.checkpoint}`",
        f"- recognizer error: `{recognizer.status.error}`",
        "", "## 汇总结果", "",
        "| condition | det P | det R | det F1 | det Mean IoU | RO Acc | CER | WER | transcribed columns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        summary_item = r["summary"]
        det = summary_item["detection_summary"]
        ro_text = "" if det["reading_order_accuracy"] is None else f"{det['reading_order_accuracy']:.3f}"
        cer_text = "" if summary_item["mean_cer"] is None else f"{summary_item['mean_cer']:.3f}"
        wer_text = "" if summary_item["mean_wer"] is None else f"{summary_item['mean_wer']:.3f}"
        lines.append(
            f"| {summary_item['condition']} | {det['precision']:.3f} | {det['recall']:.3f} | "
            f"{det['f1']:.3f} | {det['mean_iou']:.3f} | {ro_text} | {cer_text} | {wer_text} | "
            f"{summary_item['num_transcribed_columns']} |"
        )
    lines.extend([
        "", "## 说明", "",
        "当前页面级标注中的列级 `transcript` 字段为空，因此 CER/WER 暂无法真实计算。脚本已经完成 oracle/detected 裁切、识别模型加载、预测文本导出和检测错误传播统计；待补充列级转写后，可直接复用同一脚本计算 CER/WER。",
    ])
    (out_dir / "end_to_end_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "axc") == 1
    assert edit_distance("", "abc") == 3
    assert abs(cer("abc", "axc") - 1 / 3) < 1e-9
    assert wer("a b", "a c") == 0.5
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("page_level_ocr/page_level_annotations.json"))
    parser.add_argument("--yolo-predictions", type=Path, default=Path("page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43_test_conf0001/predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("page_level_ocr/results/end_to_end_ocr_test"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--charset", default=DEFAULT_CHARSET_27)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    default_ckpts = [
        Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_ocr_final.pth"),
        Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_uda_epoch_4.pth"),
        Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_uda_epoch_3.pth"),
    ]
    ckpts = args.checkpoint or default_ckpts
    pages = load_annotations(args.annotations, args.split)
    if args.limit_pages:
        pages = dict(list(pages.items())[: args.limit_pages])
    yolo_raw = load_yolo_predictions(args.yolo_predictions, list(pages), args.score_threshold)
    yolo_ordered = {pid: assign_reading_order(dets, pages[pid]) for pid, dets in yolo_raw.items()}
    oracle = {pid: make_oracle_detections(page) for pid, page in pages.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recognizer = Recognizer(ckpts, args.charset, device=args.device)
    results = [
        run_condition("oracle", pages, oracle, recognizer, args.output_dir, args.crop_padding, args.save_crops),
        run_condition("detected_yolo_thr035_ordered", pages, yolo_ordered, recognizer, args.output_dir, args.crop_padding, args.save_crops),
    ]
    for result in results:
        write_condition_outputs(result, args.output_dir)
    summary = {"recognizer": recognizer.status.__dict__, "conditions": [r["summary"] for r in results]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(results, recognizer, args.output_dir, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
