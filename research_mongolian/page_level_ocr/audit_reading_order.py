#!/usr/bin/env python3
"""Audit page-level reading_order annotations and render review assets.

This script is intentionally conservative: it validates structural reading-order
properties and highlights pages whose existing order strongly disagrees with
simple geometric ordering. It does not infer a new reading order from geometry,
because several archive pages contain sparse blocks, marginal text, and rotated
views where a purely x-sorted order can be wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SPLIT_ORDER = {"test": 0, "val": 1, "train_unlabeled": 2}
PRIORITY_PAGES = {
    "80-48-63-1",
    "80-48-64-1(1)",
    "80-48-69-1",
    "80-48-71-1",
    "80-48-72-1",
    "80-48-74-1",
    "80-48-75-1",
    "80-48-80-1",
    "80-48-65-1(2)",
    "80-48-61-1",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def active_columns(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in page.get("columns", []) if not c.get("ignore", False)]


def center_x(column: dict[str, Any]) -> float:
    x1, _, x2, _ = column["bbox"]
    return (float(x1) + float(x2)) / 2.0


def center_y(column: dict[str, Any]) -> float:
    _, y1, _, y2 = column["bbox"]
    return (float(y1) + float(y2)) / 2.0


def rotated_state(page: dict[str, Any]) -> str:
    joined = " ".join(str(page.get(k, "")) for k in ("image_path", "original_image_path", "notes")).lower()
    if "rot90ccw_rot180" in joined or "rotated_90_ccw_then_180" in joined:
        return "rotated_90_ccw_then_180_for_labeling"
    if "rot90ccw" in joined or "rotated_90_ccw" in joined:
        return "rotated_90_ccw_for_labeling"
    if "rot90cw" in joined or "rotated_90_cw" in joined:
        return "rotated_90_cw_for_labeling"
    if "rot180" in joined or "rotated_180" in joined:
        return "rotated_180_for_labeling"
    return "normal"


def inversion_rates(columns_by_order: list[dict[str, Any]]) -> tuple[float, float, int, int, int]:
    xs = [center_x(c) for c in columns_by_order]
    pairs = 0
    inv_asc = 0
    inv_desc = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            pairs += 1
            if xs[i] > xs[j]:
                inv_asc += 1
            if xs[i] < xs[j]:
                inv_desc += 1
    inc = sum(1 for a, b in zip(xs, xs[1:]) if b > a)
    dec = sum(1 for a, b in zip(xs, xs[1:]) if b < a)
    return (inv_asc / pairs if pairs else 0.0, inv_desc / pairs if pairs else 0.0, inc, dec, pairs)


def geometry_order(columns: list[dict[str, Any]], direction: str) -> list[str]:
    reverse = direction == "x_desc"
    ordered = sorted(columns, key=lambda c: ((-1 if reverse else 1) * center_x(c), center_y(c)))
    return [c.get("column_id", "") for c in ordered]


def summarize_page(page: dict[str, Any]) -> dict[str, Any]:
    valid = active_columns(page)
    ignore_count = len(page.get("columns", [])) - len(valid)
    orders = [c.get("reading_order") for c in valid]
    structural_issues: list[str] = []
    missing_ids = [c.get("column_id", "") for c in valid if c.get("reading_order") is None]
    nonint = [f"{c.get('column_id')}={c.get('reading_order')!r}" for c in valid if not isinstance(c.get("reading_order"), int) or isinstance(c.get("reading_order"), bool)]
    counts = Counter(o for o in orders if isinstance(o, int) and not isinstance(o, bool))
    duplicate_orders = sorted(o for o, count in counts.items() if count > 1)
    expected = set(range(len(valid)))
    actual = set(o for o in orders if isinstance(o, int) and not isinstance(o, bool))
    missing_numbers = sorted(expected - actual)
    extra_numbers = sorted(actual - expected)
    if missing_ids:
        structural_issues.append("missing_reading_order")
    if nonint:
        structural_issues.append("non_integer_reading_order")
    if duplicate_orders:
        structural_issues.append("duplicate_reading_order")
    if missing_numbers or extra_numbers:
        structural_issues.append("non_consecutive_reading_order")

    by_order = sorted(valid, key=lambda c: c.get("reading_order", 10**9) if isinstance(c.get("reading_order"), int) else 10**9)
    asc_rate, desc_rate, inc, dec, pairs = inversion_rates(by_order)
    rot = rotated_state(page)
    geom_suspicious = pairs >= 10 and asc_rate > 0.25 and desc_rate > 0.25
    priority = page.get("page_id") in PRIORITY_PAGES
    issue_types = list(structural_issues)
    if geom_suspicious:
        issue_types.append("geometry_zigzag_review")
    if priority:
        issue_types.append("priority_from_initial_audit")
    if rot != "normal" and not page.get("notes"):
        issue_types.append("missing_page_rotation_note")

    return {
        "page_id": page.get("page_id", ""),
        "split": page.get("split", ""),
        "valid_columns": len(valid),
        "ignore_regions": ignore_count,
        "rotated_state": rot,
        "image_path": page.get("image_path", ""),
        "original_image_path": page.get("original_image_path", ""),
        "page_notes": page.get("notes") or "",
        "reading_order_sequence": " ".join(str(o) for o in orders),
        "missing_column_ids": ";".join(missing_ids),
        "non_integer_orders": ";".join(nonint),
        "duplicate_orders": " ".join(str(o) for o in duplicate_orders),
        "missing_numbers": " ".join(str(o) for o in missing_numbers),
        "extra_numbers": " ".join(str(o) for o in extra_numbers),
        "adjacent_x_increases": inc,
        "adjacent_x_decreases": dec,
        "asc_inversion_rate": f"{asc_rate:.3f}",
        "desc_inversion_rate": f"{desc_rate:.3f}",
        "geometry_suspicious": geom_suspicious,
        "initial_priority_page": priority,
        "issue_types": ";".join(issue_types) if issue_types else "none",
        "auto_review_status": "needs_visual_review" if issue_types else "structural_pass_low_priority",
        "x_asc_order_column_ids": " ".join(geometry_order(valid, "x_asc")),
        "x_desc_order_column_ids": " ".join(geometry_order(valid, "x_desc")),
    }


def font(size: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def scale_image(im: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    scale = min(1.0, max_side / max(im.size))
    if scale >= 1.0:
        return im.convert("RGB"), 1.0
    return im.resize((int(im.width * scale), int(im.height * scale))).convert("RGB"), scale


def draw_review_image(page: dict[str, Any], out_path: Path, max_side: int = 1800) -> None:
    im = Image.open(page["image_path"])
    preview, s = scale_image(im, max_side)
    draw = ImageDraw.Draw(preview)
    fnt = font(max(13, int(22 * s)))
    fnt_small = font(max(11, int(16 * s)))
    lw = max(2, int(5 * s))
    valid = active_columns(page)
    valid_ids = {id(c) for c in valid}
    ordered_ids = {c.get("column_id"): c.get("reading_order") for c in valid}
    for c in page.get("columns", []):
        x1, y1, x2, y2 = [int(v * s) for v in c["bbox"]]
        if c.get("ignore"):
            color = (128, 128, 128)
            label = "Ignore"
        else:
            color = (22, 163, 74)
            ro = c.get("reading_order", "?")
            label = f"#{ro}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        tb = draw.textbbox((x1, y1), label, font=fnt)
        draw.rectangle([tb[0]-3, tb[1]-3, tb[2]+3, tb[3]+3], fill=color)
        draw.text((x1, y1), label, fill=(255,255,255), font=fnt)
    # connect reading-order centers
    ordered = sorted(valid, key=lambda c: c.get("reading_order", 10**9) if isinstance(c.get("reading_order"), int) else 10**9)
    centers = [(int(center_x(c) * s), int(center_y(c) * s)) for c in ordered]
    for (x1,y1),(x2,y2) in zip(centers, centers[1:]):
        draw.line([x1,y1,x2,y2], fill=(37,99,235), width=max(1, int(3*s)))
    title = f"{page['page_id']} | {page['split']} | valid={len(valid)} ignore={len(page.get('columns', []))-len(valid)} | {rotated_state(page)}"
    tb = draw.textbbox((10,10), title, font=fnt_small)
    draw.rectangle([tb[0]-4,tb[1]-4,tb[2]+4,tb[3]+4], fill=(255,255,255))
    draw.text((10,10), title, fill=(0,0,0), font=fnt_small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_w: int = 420, cols: int = 4) -> None:
    thumbs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        scale = thumb_w / im.width
        h = int(im.height * scale)
        thumbs.append((p, im.resize((thumb_w, h))))
    rows = math.ceil(len(thumbs) / cols) if thumbs else 1
    row_heights = []
    for r in range(rows):
        row_heights.append(max((im.height for _, im in thumbs[r*cols:(r+1)*cols]), default=1))
    sheet = Image.new("RGB", (thumb_w*cols, sum(row_heights)), "white")
    y = 0
    for r in range(rows):
        x = 0
        for p, im in thumbs[r*cols:(r+1)*cols]:
            sheet.paste(im, (x,y))
            x += thumb_w
        y += row_heights[r]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def maybe_add_rotation_notes(payload: dict[str, Any]) -> list[dict[str, str]]:
    changes = []
    audit_tag = f"reading_order_audit_{date.today().isoformat()}: rotation metadata checked"
    for page in payload.get("pages", []):
        rot = rotated_state(page)
        if rot == "normal":
            continue
        old = page.get("notes") or ""
        additions = []
        if rot not in old:
            additions.append(rot)
        if audit_tag not in old:
            additions.append(audit_tag)
        if additions:
            new = "; ".join([part for part in [old, *additions] if part])
            page["notes"] = new
            changes.append({"page_id": page["page_id"], "old_notes": old, "new_notes": new})
    dataset = payload.setdefault("dataset", {})
    ds_notes = dataset.get("notes", "")
    if audit_tag not in ds_notes:
        dataset["notes"] = (ds_notes + "; " if ds_notes else "") + audit_tag
    return changes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fix-rotation-notes", action="store_true")
    args = ap.parse_args()

    payload = load_json(args.annotations)
    pages = payload.get("pages", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_page(p) for p in pages]
    rows.sort(key=lambda r: (SPLIT_ORDER.get(r["split"], 99), r["page_id"]))

    write_csv(args.output_dir / "reading_order_audit_62_pages.csv", rows)
    checklist_rows = []
    for i, r in enumerate(rows, start=1):
        checklist_rows.append({
            "review_index": i,
            "page_id": r["page_id"],
            "split": r["split"],
            "valid_columns": r["valid_columns"],
            "ignore_regions": r["ignore_regions"],
            "rotated_state": r["rotated_state"],
            "priority": "P0" if r["initial_priority_page"] or r["issue_types"] != "none" else "P1",
            "review_status": "pending_human_visual_confirmation",
            "action_needed": r["issue_types"],
            "notes": "",
        })
    write_csv(args.output_dir / "reading_order_manual_review_checklist_62_pages.csv", checklist_rows)

    summary = {
        "annotation_file": str(args.annotations),
        "pages": len(pages),
        "valid_columns": sum(int(r["valid_columns"]) for r in rows),
        "ignore_regions": sum(int(r["ignore_regions"]) for r in rows),
        "structural_issue_pages": [r["page_id"] for r in rows if any(x in r["issue_types"] for x in ["missing_reading_order", "non_integer_reading_order", "duplicate_reading_order", "non_consecutive_reading_order"])],
        "geometry_suspicious_pages": [r["page_id"] for r in rows if r["geometry_suspicious"]],
        "initial_priority_pages": sorted(PRIORITY_PAGES),
        "rotation_note_missing_pages": [r["page_id"] for r in rows if "missing_page_rotation_note" in r["issue_types"]],
        "outputs": {
            "audit_csv": str(args.output_dir / "reading_order_audit_62_pages.csv"),
            "manual_checklist_csv": str(args.output_dir / "reading_order_manual_review_checklist_62_pages.csv"),
            "review_images_dir": str(args.output_dir / "review_images") if args.render else None,
        },
    }

    if args.render:
        img_dir = args.output_dir / "review_images"
        by_id = {p["page_id"]: p for p in pages}
        rendered = []
        for r in rows:
            out = img_dir / f"{r['review_index'] if 'review_index' in r else ''}{r['page_id']}_reading_order.jpg"
            # Avoid blank prefixes; use split ordering from checklist for stable names.
            idx = next(c["review_index"] for c in checklist_rows if c["page_id"] == r["page_id"])
            out = img_dir / f"{idx:02d}_{r['split']}_{r['page_id']}_reading_order.jpg"
            draw_review_image(by_id[r["page_id"]], out)
            rendered.append(out)
        summary["rendered_review_images"] = len(rendered)
        for split in ("test", "val", "train_unlabeled"):
            split_imgs = [p for p in rendered if f"_{split}_" in p.name]
            make_contact_sheet(split_imgs, args.output_dir / f"contact_sheet_{split}.jpg")
        priority_imgs = [p for p in rendered if any(pid in p.name for pid in PRIORITY_PAGES)]
        make_contact_sheet(priority_imgs, args.output_dir / "contact_sheet_priority_10.jpg", cols=2)

    changes = []
    if args.fix_rotation_notes:
        backup = args.annotations.with_suffix(args.annotations.suffix + f".before_reading_order_audit_{date.today().isoformat().replace('-', '')}.bak")
        if not backup.exists():
            shutil.copy2(args.annotations, backup)
        changes = maybe_add_rotation_notes(payload)
        write_json(args.annotations, payload)
        summary["annotation_backup"] = str(backup)
        summary["rotation_note_changes"] = changes

    (args.output_dir / "reading_order_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Reading-order audit for old 62 pages",
        "",
        f"- Annotation file: `{args.annotations}`",
        f"- Pages: {summary['pages']}",
        f"- Valid TextColumn boxes: {summary['valid_columns']}",
        f"- Ignore regions: {summary['ignore_regions']}",
        f"- Structural issue pages: {len(summary['structural_issue_pages'])}",
        f"- Geometry-suspicious pages: {len(summary['geometry_suspicious_pages'])}",
        f"- Rotation-note-missing pages before fix: {len(summary['rotation_note_missing_pages'])}",
        "",
        "## Geometry-suspicious / priority pages",
        "",
    ]
    for pid in sorted(set(summary["geometry_suspicious_pages"]) | PRIORITY_PAGES):
        rec = next(r for r in rows if r["page_id"] == pid)
        report_lines.append(f"- `{pid}` ({rec['split']}): {rec['issue_types']}; asc_inv={rec['asc_inversion_rate']}, desc_inv={rec['desc_inversion_rate']}")
    report_lines += [
        "",
        "## Outputs",
        "",
        f"- Audit CSV: `{args.output_dir / 'reading_order_audit_62_pages.csv'}`",
        f"- Manual checklist: `{args.output_dir / 'reading_order_manual_review_checklist_62_pages.csv'}`",
        f"- Summary JSON: `{args.output_dir / 'reading_order_audit_summary.json'}`",
    ]
    if args.render:
        report_lines += [
            f"- Review images: `{args.output_dir / 'review_images'}`",
            f"- Priority contact sheet: `{args.output_dir / 'contact_sheet_priority_10.jpg'}`",
        ]
    if changes:
        report_lines += ["", "## Applied metadata changes", ""]
        for c in changes:
            report_lines.append(f"- `{c['page_id']}`: `{c['old_notes']}` -> `{c['new_notes']}`")
    (args.output_dir / "Reading_Order_Audit_2026-05-12.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
