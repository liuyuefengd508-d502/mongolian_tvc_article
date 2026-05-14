#!/usr/bin/env python3
"""Prepare a small OCR case-study package for expert transcription.

The package contains:
  * original page images for selected test pages;
  * oracle crops from ground-truth TextColumn boxes;
  * detected crops from YOLO predictions;
  * contact sheets for convenient review;
  * CSV templates to be filled by a Mongolian-script expert;
  * a summary of layout matching for the selected pages.

This script deliberately does not compute CER/WER unless real transcripts are
provided later. It prepares the minimum material needed for a 3--5 page OCR
case study without pretending that pseudo labels are ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_yolo_rule_gt import assign_reading_order, load_yolo_predictions
from evaluate_layout import aggregate_page_metrics, evaluate_page
from export_yolo_dataset import safe_stem
from run_end_to_end_ocr_experiment import Recognizer, bbox_iou, clamp_bbox, edit_distance


DEFAULT_PAGES = ["80-48-69-4", "80-48-70-4", "80-48-66-1(1)"]


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {page["page_id"]: page for page in data.get("pages", [])}


def load_font(size: int = 18) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_contact_sheet(crop_paths: list[Path], labels: list[str], out_path: Path, thumb_h: int = 260, cols: int = 6) -> None:
    if not crop_paths:
        return
    font = load_font(16)
    thumbs = []
    for path, label in zip(crop_paths, labels):
        img = Image.open(path).convert("RGB")
        scale = thumb_h / max(1, img.height)
        thumb = img.resize((max(1, int(img.width * scale)), thumb_h), Image.BILINEAR)
        canvas = Image.new("RGB", (max(thumb.width, 120), thumb_h + 28), "white")
        canvas.paste(thumb, ((canvas.width - thumb.width) // 2, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), label, fill=(0, 0, 0), font=font)
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / cols)
    col_w = max(t.width for t in thumbs)
    row_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * col_w, rows * row_h), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * col_w
        y = (i // cols) * row_h
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def crop_and_save(image: Image.Image, bbox: list[float], out_path: Path, pad: int = 4) -> list[int] | None:
    clamped = clamp_bbox(bbox, image.width, image.height, pad=pad)
    if clamped is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(clamped)).save(out_path, quality=92)
    return clamped


def best_match(det: dict[str, Any], columns: list[dict[str, Any]]) -> tuple[str, float]:
    best_id, best_iou = "", 0.0
    for col in columns:
        if col.get("ignore"):
            continue
        iou = bbox_iou(det["bbox"], col["bbox"])
        if iou > best_iou:
            best_id, best_iou = col.get("column_id", ""), iou
    return best_id, best_iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("page_level_ocr/page_level_annotations.json"))
    parser.add_argument("--yolo-predictions", type=Path, default=Path("page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43_test/predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("page_level_ocr/results/ocr_case_study_3pages"))
    parser.add_argument("--page-id", action="append", default=[])
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--run-recognizer", action="store_true")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    page_ids = args.page_id or DEFAULT_PAGES
    pages = {pid: annotations[pid] for pid in page_ids}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yolo_by_page = load_yolo_predictions(args.yolo_predictions, args.score_threshold, {safe_stem(pid): pid for pid in pages})
    yolo_by_page = {pid: assign_reading_order(yolo_by_page.get(pid, []), pages[pid]) for pid in pages}

    recognizer = None
    if args.run_recognizer:
        recognizer = Recognizer(
            [
                Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_ocr_final.pth"),
                Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_uda_epoch_4.pth"),
                Path("/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/mongolian_uda_epoch_3.pth"),
            ],
            charset="abcdefghijklmnopqrstuvwxyz ",
            device="auto",
        )

    oracle_rows: list[dict[str, Any]] = []
    detected_rows: list[dict[str, Any]] = []
    page_metrics = []

    for page_id, page in pages.items():
        image = Image.open(page["image_path"]).convert("RGB")
        page_copy = args.output_dir / "pages" / f"{safe_stem(page_id)}.jpg"
        page_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(page["image_path"], page_copy)

        valid_cols = sorted([c for c in page["columns"] if not c.get("ignore")], key=lambda c: int(c.get("reading_order", 0)))
        oracle_crop_paths, oracle_labels = [], []
        for col in valid_cols:
            order = int(col.get("reading_order", 0))
            crop_path = args.output_dir / "oracle_crops" / f"{safe_stem(page_id)}_{order:03d}_{col.get('column_id','')}.jpg"
            bbox = crop_and_save(image, col["bbox"], crop_path, pad=args.crop_padding)
            pred_text = recognizer.predict(Image.open(crop_path).convert("RGB")) if recognizer and bbox else ""
            oracle_rows.append(
                {
                    "page_id": page_id,
                    "column_id": col.get("column_id", ""),
                    "reading_order": order,
                    "bbox": bbox,
                    "oracle_crop_path": str(crop_path),
                    "expert_transcript": "",
                    "ocr_prediction_optional": pred_text,
                    "notes": "",
                }
            )
            oracle_crop_paths.append(crop_path)
            oracle_labels.append(f"{page_id} #{order}")
        make_contact_sheet(oracle_crop_paths, oracle_labels, args.output_dir / "contact_sheets" / f"{safe_stem(page_id)}_oracle_contact.jpg")

        detections = yolo_by_page.get(page_id, [])
        page_metric = evaluate_page(detections, page["columns"], iou_threshold=0.5)
        page_metric.update({"page_id": page_id})
        page_metrics.append(page_metric)
        det_crop_paths, det_labels = [], []
        for det in detections:
            order = int(det.get("reading_order", 0))
            crop_path = args.output_dir / "detected_crops" / f"{safe_stem(page_id)}_{order:03d}_score{float(det.get('score',0)):.3f}.jpg"
            bbox = crop_and_save(image, det["bbox"], crop_path, pad=args.crop_padding)
            matched_id, matched_iou = best_match(det, page["columns"])
            pred_text = recognizer.predict(Image.open(crop_path).convert("RGB")) if recognizer and bbox else ""
            detected_rows.append(
                {
                    "page_id": page_id,
                    "detected_reading_order": order,
                    "score": det.get("score", ""),
                    "bbox": bbox,
                    "detected_crop_path": str(crop_path),
                    "matched_gt_column_id": matched_id,
                    "matched_iou": matched_iou,
                    "expert_transcript_for_matched_gt": "",
                    "ocr_prediction_optional": pred_text,
                    "notes": "",
                }
            )
            det_crop_paths.append(crop_path)
            det_labels.append(f"{page_id} d#{order}")
        make_contact_sheet(det_crop_paths, det_labels, args.output_dir / "contact_sheets" / f"{safe_stem(page_id)}_detected_contact.jpg")

    for name, rows in (("expert_transcription_template_oracle.csv", oracle_rows), ("detected_crop_review_template.csv", detected_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "purpose": "Small expert-transcription OCR case study package",
        "selected_pages": page_ids,
        "num_oracle_columns_to_transcribe": len(oracle_rows),
        "num_detected_crops_to_review": len(detected_rows),
        "layout_summary_on_selected_pages": aggregate_page_metrics(page_metrics),
        "recognizer_status": getattr(recognizer, "status", None).__dict__ if recognizer else "not_run",
        "next_step": "Fill expert_transcript in expert_transcription_template_oracle.csv; then run evaluate_ocr_case_study.py.",
    }
    (args.output_dir / "case_study_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README_小规模OCR案例研究.md").write_text(
        f"""# 小规模 OCR case study 标注包

## 目的

本目录用于补充论文中的小规模 OCR case study。当前不会虚报 CER/WER；需要先由蒙古文专家填写真实转写。

## 选中页面

{', '.join(page_ids)}

## 需要填写的文件

请优先填写：

`expert_transcription_template_oracle.csv`

其中 `expert_transcript` 列留空，请专家根据 `oracle_crop_path` 中的裁切图逐列填写真实转写。

## 可选复核文件

`detected_crop_review_template.csv`

用于检查自动检测裁切是否匹配对应 GT 列，后续可比较 oracle crop 与 detected crop。

## 图片目录

- `pages/`：整页图。
- `oracle_crops/`：人工 GT 文本列框裁切图，适合专家转写。
- `detected_crops/`：YOLO 自动检测裁切图。
- `contact_sheets/`：每页裁切图总览。

## 当前规模

- oracle columns to transcribe: {len(oracle_rows)}
- detected crops to review: {len(detected_rows)}

## 下一步

填好 `expert_transcription_template_oracle.csv` 后运行：

```bash
python page_level_ocr/evaluate_ocr_case_study.py \\
  --case-dir page_level_ocr/results/ocr_case_study_3pages
```
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
