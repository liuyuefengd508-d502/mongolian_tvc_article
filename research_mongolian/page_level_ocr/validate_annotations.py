#!/usr/bin/env python3
"""Validate page-level OCR annotation files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_SPLITS = {"train_unlabeled", "val", "test"}
VALID_ORIENTATIONS = {
    "correct",
    "rotated_90_ccw",
    "rotated_90_cw",
    "rotated_180",
    "ambiguous",
}
VALID_DEGRADATION_TAGS = {
    "severe_fade",
    "bleed_through",
    "red_seal",
    "fold",
    "stain",
    "broken_spine",
    "dense_background",
    "marginalia",
    "overlap",
    "other",
}


@dataclass
class Issue:
    level: str
    location: str
    message: str


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    manifest: dict[str, dict[str, str]] = {}
    for row in rows:
        page_id = row.get("page_id", "")
        if page_id:
            manifest[page_id] = row
    return manifest


def load_annotations(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Annotation root must be a JSON object: {path}")
    return payload


def add(issues: list[Issue], level: str, location: str, message: str) -> None:
    issues.append(Issue(level=level, location=location, message=message))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_bbox(
    issues: list[Issue],
    bbox: Any,
    location: str,
    width: int | None,
    height: int | None,
) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        add(issues, "ERROR", location, "bbox 必须是长度为 4 的数组：[x_min, y_min, x_max, y_max]。")
        return
    if not all(is_number(value) for value in bbox):
        add(issues, "ERROR", location, "bbox 中的 4 个值都必须是数字。")
        return

    x_min, y_min, x_max, y_max = [float(value) for value in bbox]
    if x_min >= x_max or y_min >= y_max:
        add(issues, "ERROR", location, "bbox 坐标顺序不合法，必须满足 x_min < x_max 且 y_min < y_max。")
    if x_min < 0 or y_min < 0:
        add(issues, "ERROR", location, "bbox 不能出现负坐标。")
    if width is not None and x_max > width:
        add(issues, "ERROR", location, f"bbox 的 x_max={x_max:g} 超出页面宽度 width={width}。")
    if height is not None and y_max > height:
        add(issues, "ERROR", location, f"bbox 的 y_max={y_max:g} 超出页面高度 height={height}。")

    if width is not None and height is not None:
        box_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        page_area = float(width * height)
        if page_area > 0 and box_area / page_area > 0.85:
            add(issues, "WARN", location, "bbox 覆盖超过页面面积的 85%，请确认是否把整页或过多空白框进去了。")


def validate_reading_order(issues: list[Issue], columns: list[Any], page_location: str) -> None:
    active_orders: list[int] = []
    seen: dict[int, int] = {}
    for idx, column in enumerate(columns):
        if not isinstance(column, dict):
            continue
        if bool(column.get("ignore", False)):
            continue
        order = column.get("reading_order")
        if not isinstance(order, int) or isinstance(order, bool):
            add(issues, "ERROR", f"{page_location}.columns[{idx}]", "reading_order 必须是从 0 开始的整数。")
            continue
        active_orders.append(order)
        seen[order] = seen.get(order, 0) + 1

    if not active_orders:
        add(issues, "WARN", page_location, "该页没有参与评估的列框；如果不是空白页，请补充 columns。")
        return

    expected = list(range(len(active_orders)))
    actual = sorted(active_orders)
    if actual != expected:
        add(
            issues,
            "ERROR",
            page_location,
            f"reading_order 必须从 0 开始且连续。当前为 {actual}，期望为 {expected}。",
        )
    duplicates = sorted(order for order, count in seen.items() if count > 1)
    if duplicates:
        add(issues, "ERROR", page_location, f"reading_order 存在重复值：{duplicates}。")


def validate_annotations(payload: dict[str, Any], manifest: dict[str, dict[str, str]]) -> list[Issue]:
    issues: list[Issue] = []
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return [Issue("ERROR", "pages", "annotations 文件必须包含 pages 数组。")]

    seen_pages: set[str] = set()
    for page_idx, page in enumerate(pages):
        page_location = f"pages[{page_idx}]"
        if not isinstance(page, dict):
            add(issues, "ERROR", page_location, "每个 page 必须是 JSON object。")
            continue

        page_id = page.get("page_id")
        if not isinstance(page_id, str) or not page_id:
            add(issues, "ERROR", page_location, "page_id 必须是非空字符串。")
            continue
        page_location = f"page:{page_id}"
        if page_id in seen_pages:
            add(issues, "ERROR", page_location, "page_id 在标注文件中重复。")
        seen_pages.add(page_id)

        manifest_row = manifest.get(page_id)
        if manifest_row is None:
            add(issues, "ERROR", page_location, "page_id 不存在于 page_split_manifest.csv。")
        else:
            split = page.get("split")
            if split != manifest_row.get("split"):
                add(
                    issues,
                    "ERROR",
                    page_location,
                    f"split 与 manifest 不一致：标注为 {split!r}，manifest 为 {manifest_row.get('split')!r}。",
                )
            image_path = page.get("image_path")
            if isinstance(image_path, str) and image_path != manifest_row.get("image_path"):
                add(issues, "WARN", page_location, "image_path 与 manifest 不一致，请确认是否移动或复制过数据。")

        split = page.get("split")
        if split not in VALID_SPLITS:
            add(issues, "ERROR", page_location, f"split 取值非法：{split!r}。")

        width = page.get("width")
        height = page.get("height")
        width_value = width if isinstance(width, int) and not isinstance(width, bool) and width > 0 else None
        height_value = height if isinstance(height, int) and not isinstance(height, bool) and height > 0 else None
        if width_value is None:
            add(issues, "WARN", page_location, "缺少合法 width；将无法检查 bbox 是否超出页面宽度。")
        if height_value is None:
            add(issues, "WARN", page_location, "缺少合法 height；将无法检查 bbox 是否超出页面高度。")

        columns = page.get("columns")
        if not isinstance(columns, list):
            add(issues, "ERROR", page_location, "columns 必须是数组。")
            continue

        seen_columns: set[str] = set()
        for col_idx, column in enumerate(columns):
            col_location = f"{page_location}.columns[{col_idx}]"
            if not isinstance(column, dict):
                add(issues, "ERROR", col_location, "column 必须是 JSON object。")
                continue

            column_id = column.get("column_id")
            if not isinstance(column_id, str) or not column_id:
                add(issues, "ERROR", col_location, "column_id 必须是非空字符串。")
            elif column_id in seen_columns:
                add(issues, "ERROR", col_location, f"column_id 重复：{column_id}。")
            else:
                seen_columns.add(column_id)

            validate_bbox(issues, column.get("bbox"), col_location, width_value, height_value)

            orientation = column.get("orientation")
            if orientation not in VALID_ORIENTATIONS:
                add(issues, "ERROR", col_location, f"orientation 取值非法：{orientation!r}。")

            tags = column.get("degradation_tags", [])
            if not isinstance(tags, list):
                add(issues, "ERROR", col_location, "degradation_tags 必须是数组。")
            else:
                invalid_tags = [tag for tag in tags if tag not in VALID_DEGRADATION_TAGS]
                if invalid_tags:
                    add(issues, "ERROR", col_location, f"degradation_tags 含非法标签：{invalid_tags}。")

            ignore = column.get("ignore", False)
            if not isinstance(ignore, bool):
                add(issues, "ERROR", col_location, "ignore 必须是 true 或 false。")

        validate_reading_order(issues, columns, page_location)

    missing_val_test = [
        page_id
        for page_id, row in manifest.items()
        if row.get("split") in {"val", "test"} and page_id not in seen_pages
    ]
    if missing_val_test:
        preview = ", ".join(missing_val_test[:10])
        suffix = " ..." if len(missing_val_test) > 10 else ""
        add(
            issues,
            "WARN",
            "manifest",
            f"有 {len(missing_val_test)} 个 val/test 页面尚未出现在标注文件中：{preview}{suffix}",
        )

    return issues


def print_issues(issues: list[Issue]) -> None:
    for issue in issues:
        print(f"[{issue.level}] {issue.location}: {issue.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="Annotation JSON file.")
    parser.add_argument("--manifest", required=True, type=Path, help="Page split manifest CSV.")
    parser.add_argument("--max-issues", default=200, type=int, help="Maximum number of issues to print.")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    payload = load_annotations(args.annotations)
    issues = validate_annotations(payload, manifest)

    errors = [issue for issue in issues if issue.level == "ERROR"]
    warnings = [issue for issue in issues if issue.level == "WARN"]
    print(f"检查文件：{args.annotations}")
    print(f"页面数：{len(payload.get('pages', [])) if isinstance(payload.get('pages'), list) else 0}")
    print(f"错误：{len(errors)}；警告：{len(warnings)}")

    if issues:
        print_issues(issues[: args.max_issues])
        if len(issues) > args.max_issues:
            print(f"... 还有 {len(issues) - args.max_issues} 条问题未显示，可调大 --max-issues。")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
