#!/usr/bin/env python3
"""Compare GT, rule-based detections, and YOLO predictions on page images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate_layout import aggregate_page_metrics, evaluate_page
from export_yolo_dataset import safe_stem


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def scale_image(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image.copy(), 1.0
    return image.resize((int(width * scale), int(height * scale))), scale


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    scale: float,
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = [int(v * scale) for v in box]
    line_width = max(2, int(4 * scale))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    pad = max(2, int(3 * scale))
    bg = [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad]
    draw.rectangle(bg, fill=color)
    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)


def draw_panel(
    image: Image.Image,
    boxes: list[dict],
    title: str,
    color: tuple[int, int, int],
    max_side: int,
    label_key: str,
) -> Image.Image:
    preview, scale = scale_image(image, max_side)
    draw = ImageDraw.Draw(preview)
    font = load_font(max(12, int(18 * scale)))
    title_font = load_font(max(18, int(28 * scale)))
    title_bbox = draw.textbbox((8, 8), title, font=title_font)
    draw.rectangle(
        [0, 0, title_bbox[2] + 18, title_bbox[3] + 14],
        fill=(255, 255, 255),
        outline=(40, 40, 40),
        width=max(1, int(2 * scale)),
    )
    draw.text((8, 8), title, fill=(20, 20, 20), font=title_font)
    for idx, box in enumerate(boxes):
        if box.get("ignore", False):
            box_color = (120, 120, 120)
            label = f"I {box.get(label_key, idx)}"
        else:
            box_color = color
            label = str(box.get(label_key, idx))
            if "score" in box:
                label = f"{label}:{float(box['score']):.2f}"
        draw_box(draw, box["bbox"], scale, box_color, label, font)
    return preview


def concat_panels(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def yolo_xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def load_yolo_predictions(path: Path, score_threshold: float, page_id_by_stem: dict[str, str] | None = None) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_page: dict[str, list[dict]] = {}
    for item in raw:
        score = float(item.get("score", 0.0))
        if score < score_threshold:
            continue
        raw_id = str(item["image_id"])
        file_stem = Path(str(item.get("file_name", ""))).stem
        page_id = raw_id
        if page_id_by_stem:
            page_id = page_id_by_stem.get(raw_id, page_id_by_stem.get(file_stem, raw_id))
        by_page.setdefault(page_id, []).append(
            {
                "bbox": yolo_xywh_to_xyxy(item["bbox"]),
                "score": score,
                "method": "yolov8n",
                "orientation": "correct",
            }
        )
    for detections in by_page.values():
        detections.sort(key=lambda det: (det["bbox"][0], det["bbox"][1]))
        for order, det in enumerate(detections):
            det["reading_order"] = order
    return by_page


def assign_reading_order(detections: list[dict], page: dict) -> list[dict]:
    ordered = [dict(det) for det in detections]
    image_path = str(page.get("image_path", "")).lower()
    if "rot90" in image_path:
        ordered.sort(key=lambda det: (-(det["bbox"][0] + det["bbox"][2]) / 2.0, det["bbox"][1]))
    else:
        ordered.sort(key=lambda det: ((det["bbox"][0] + det["bbox"][2]) / 2.0, det["bbox"][1]))
    for order, det in enumerate(ordered):
        det["reading_order"] = order
    return ordered


def load_rule_detections(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {page["page_id"]: page.get("detections", []) for page in raw.get("pages", [])}


def load_annotations(path: Path, split: str | None) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pages = raw.get("pages", [])
    if split:
        pages = [page for page in pages if page.get("split") == split]
    return {page["page_id"]: page for page in pages}


def evaluate_detector(name: str, detections: dict[str, list[dict]], gt_by_page: dict[str, dict]) -> dict:
    page_metrics: list[dict] = []
    for page_id, page in gt_by_page.items():
        page_detections = detections.get(page_id, [])
        if name == "yolov8n":
            page_detections = assign_reading_order(page_detections, page)
        metric = evaluate_page(page_detections, page.get("columns", []), iou_threshold=0.5)
        metric.update({"page_id": page_id, "method": name, "split": page.get("split")})
        page_metrics.append(metric)
    return {"summary": aggregate_page_metrics(page_metrics), "pages": page_metrics}


def write_outputs(
    gt_by_page: dict[str, dict],
    rule_by_page: dict[str, list[dict]],
    yolo_by_page: dict[str, list[dict]],
    output_dir: Path,
    max_side: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for page_id, page in gt_by_page.items():
        image_path = Path(page["image_path"])
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        gt_boxes = page.get("columns", [])
        panels = [
            draw_panel(image, gt_boxes, f"{page_id} GT", (22, 163, 74), max_side, "reading_order"),
            draw_panel(image, rule_by_page.get(page_id, []), "Rule proposed", (220, 38, 38), max_side, "reading_order"),
            draw_panel(image, assign_reading_order(yolo_by_page.get(page_id, []), page), "YOLOv8n", (37, 99, 235), max_side, "reading_order"),
        ]
        comparison = concat_panels(panels)
        comparison.save(output_dir / f"{page_id}_gt_rule_yolo.jpg", quality=92)
        count += 1
    print(f"Wrote {count} comparison images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--rule-detections", required=True, type=Path)
    parser.add_argument("--yolo-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["train_unlabeled", "val", "test"])
    parser.add_argument("--score-threshold", default=0.25, type=float)
    parser.add_argument("--max-side", default=1400, type=int)
    args = parser.parse_args()

    gt_by_page = load_annotations(args.annotations, args.split)
    rule_by_page = load_rule_detections(args.rule_detections)
    page_id_by_stem = {safe_stem(page_id): page_id for page_id in gt_by_page}
    yolo_by_page = load_yolo_predictions(args.yolo_predictions, args.score_threshold, page_id_by_stem)

    rule_metrics = evaluate_detector("rule_proposed", rule_by_page, gt_by_page)
    yolo_metrics = evaluate_detector("yolov8n", yolo_by_page, gt_by_page)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_rule_vs_yolo.json").write_text(
        json.dumps({"rule_proposed": rule_metrics, "yolov8n": yolo_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_outputs(gt_by_page, rule_by_page, yolo_by_page, args.output_dir / "images", args.max_side)
    print(json.dumps({"rule_proposed": rule_metrics["summary"], "yolov8n": yolo_metrics["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
