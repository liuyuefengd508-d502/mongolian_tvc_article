#!/usr/bin/env python3
"""Render GT and detection overlays for layout debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], scale: float, color: tuple[int, int, int], label: str, font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = [int(v * scale) for v in box]
    line_width = max(2, int(4 * scale))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    pad = max(2, int(3 * scale))
    bg = [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad]
    draw.rectangle(bg, fill=color)
    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-side", default=2200, type=int)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    gt_by_page = {page["page_id"]: page for page in annotations.get("pages", [])}
    det_by_page = {page["page_id"]: page for page in detections.get("pages", [])}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for page_id, gt_page in gt_by_page.items():
        image_path = Path(gt_page["image_path"])
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        preview, scale = scale_image(image, args.max_side)
        draw = ImageDraw.Draw(preview)
        font = load_font(max(12, int(18 * scale)))

        for column in gt_page.get("columns", []):
            if column.get("ignore", False):
                color = (120, 120, 120)
                label = f"GT-I {column.get('reading_order', '?')}"
            else:
                color = (22, 163, 74)
                label = f"GT {column.get('reading_order', '?')}"
            draw_box(draw, column["bbox"], scale, color, label, font)

        for idx, det in enumerate(det_by_page.get(page_id, {}).get("detections", [])):
            draw_box(draw, det["bbox"], scale, (220, 38, 38), f"P {idx}", font)

        out_path = args.output_dir / f"{page_id}_gt_pred.jpg"
        preview.save(out_path, quality=92)
        count += 1
    print(f"Wrote {count} layout comparison images to {args.output_dir}")


if __name__ == "__main__":
    main()
