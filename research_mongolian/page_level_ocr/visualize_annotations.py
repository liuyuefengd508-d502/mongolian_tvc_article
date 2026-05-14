#!/usr/bin/env python3
"""Render page-level column annotations as review images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "correct": (24, 132, 72),
    "rotated_90_ccw": (214, 119, 0),
    "rotated_90_cw": (214, 119, 0),
    "rotated_180": (168, 85, 247),
    "ambiguous": (220, 38, 38),
}


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
    resized = image.resize((int(width * scale), int(height * scale)))
    return resized, scale


def draw_columns(image: Image.Image, columns: list[dict], scale: float) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    font = load_font(max(12, int(20 * scale)))
    line_width = max(2, int(5 * scale))

    for column in columns:
        x1, y1, x2, y2 = [int(v * scale) for v in column["bbox"]]
        orientation = column.get("orientation", "correct")
        color = COLORS.get(orientation, (37, 99, 235))
        if column.get("ignore", False):
            color = (120, 120, 120)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = f"{column.get('reading_order', '?')} {orientation}"
        if column.get("ignore", False):
            label += " ignore"

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        pad = max(2, int(4 * scale))
        bg = [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad]
        draw.rectangle(bg, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    return output


def render_annotations(annotation_path: Path, output_dir: Path, max_side: int, split: str | None) -> int:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for page in pages:
        if split and page.get("split") != split:
            continue
        image_path = Path(page["image_path"])
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue
        image = Image.open(image_path)
        preview, scale = scale_image(image, max_side=max_side)
        rendered = draw_columns(preview, page.get("columns", []), scale)
        out_path = output_dir / f"{page['page_id']}_annotations.jpg"
        rendered.save(out_path, quality=92)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-side", default=1800, type=int)
    parser.add_argument("--split", choices=["train_unlabeled", "val", "test"])
    args = parser.parse_args()

    count = render_annotations(args.annotations, args.output_dir, args.max_side, args.split)
    print(f"Wrote {count} annotation preview images to {args.output_dir}")


if __name__ == "__main__":
    main()
