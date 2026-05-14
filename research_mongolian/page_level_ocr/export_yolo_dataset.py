#!/usr/bin/env python3
"""Export page-level column annotations to a YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def safe_stem(page_id: str) -> str:
    return (
        page_id.replace("/", "_")
        .replace("\\", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
    )


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_line(bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def image_size(image_path: Path, annotated_width: int | None, annotated_height: int | None) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    if annotated_width and annotated_height and (width, height) != (annotated_width, annotated_height):
        raise ValueError(
            f"Image size mismatch for {image_path}: file={(width, height)}, "
            f"annotation={(annotated_width, annotated_height)}"
        )
    return width, height


def copy_or_convert_image(image_path: Path, target_image: Path) -> None:
    if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        shutil.copy2(image_path, target_image)
        return
    with Image.open(image_path) as image:
        image.convert("RGB").save(target_image, quality=95)


def export_dataset(annotations_path: Path, output_dir: Path, splits: set[str]) -> dict:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "pages": 0,
        "boxes": 0,
        "ignored_boxes": 0,
        "splits": {split: {"pages": 0, "boxes": 0, "ignored_boxes": 0} for split in sorted(splits)},
    }

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for page in annotations.get("pages", []):
        split = page.get("split")
        if split not in splits:
            continue

        image_path = Path(page["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {page.get('page_id')}: {image_path}")

        width, height = image_size(image_path, page.get("width"), page.get("height"))
        stem = safe_stem(page["page_id"])
        image_ext = image_path.suffix.lower()
        target_ext = image_ext if image_ext in {".jpg", ".jpeg", ".png"} else ".jpg"
        target_image = output_dir / "images" / split / f"{stem}{target_ext}"
        target_label = output_dir / "labels" / split / f"{stem}.txt"

        copy_or_convert_image(image_path, target_image)

        lines: list[str] = []
        ignored = 0
        for column in page.get("columns", []):
            if column.get("ignore", False):
                ignored += 1
                continue
            bbox = clamp_bbox(column["bbox"], width, height)
            if bbox is None:
                continue
            lines.append(yolo_line(bbox, width, height))

        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["pages"] += 1
        stats["boxes"] += len(lines)
        stats["ignored_boxes"] += ignored
        stats["splits"][split]["pages"] += 1
        stats["splits"][split]["boxes"] += len(lines)
        stats["splits"][split]["ignored_boxes"] += ignored

    train_split = "train" if "train" in splits else ("train_unlabeled" if "train_unlabeled" in splits else ("val" if "val" in splits else sorted(splits)[0]))
    val_split = "val" if "val" in splits else train_split
    test_split = "test" if "test" in splits else val_split

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                f"train: images/{train_split}",
                f"val: images/{val_split}",
                f"test: images/{test_split}",
                "names:",
                "  0: TextColumn",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "export_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated split names to export. Default: val,test",
    )
    args = parser.parse_args()

    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    stats = export_dataset(args.annotations, args.output_dir, splits)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
