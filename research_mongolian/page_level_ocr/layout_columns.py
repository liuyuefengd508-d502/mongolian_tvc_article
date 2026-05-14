#!/usr/bin/env python3
"""Column extraction methods for page-level traditional Mongolian OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import binary_closing, binary_opening, label
from skimage import filters, io, measure, morphology


@dataclass
class ColumnDetection:
    bbox: list[int]
    score: float
    method: str
    reading_order: int
    orientation: str = "correct"
    diagnostics: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if data["diagnostics"] is None:
            data["diagnostics"] = {}
        return data


def load_gray(image_path: str | Path) -> np.ndarray:
    img = io.imread(str(image_path))
    img = np.asarray(img)
    if img.ndim == 3:
        if img.dtype.kind in {"u", "i"}:
            img_float = img.astype(np.float32)
            img_float = np.nan_to_num(img_float, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = float(np.nanmax(img_float)) or 1.0
            img_float = np.clip(img_float / max_val, 0.0, 1.0)
        else:
            img_float = np.nan_to_num(img.astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0)
            img_float = np.clip(img_float, 0.0, 1.0)
        if img_float.shape[-1] >= 3:
            gray = (
                0.2125 * img_float[..., 0]
                + 0.7154 * img_float[..., 1]
                + 0.0721 * img_float[..., 2]
            )
        else:
            gray = img_float[..., 0]
        gray = np.nan_to_num(gray, nan=1.0, posinf=1.0, neginf=0.0)
        gray = (gray * 255).astype(np.uint8)
    else:
        if img.dtype == np.uint8:
            gray = img
        else:
            img_float = img.astype(np.float32)
            min_val = float(np.nanmin(img_float))
            max_val = float(np.nanmax(img_float))
            if max_val > min_val:
                img_float = (img_float - min_val) / (max_val - min_val)
            else:
                img_float = np.zeros_like(img_float)
            gray = (np.nan_to_num(img_float, nan=1.0) * 255).astype(np.uint8)
    return gray


def sauvola_mask(gray: np.ndarray, window_size: int = 25) -> np.ndarray:
    threshold = filters.threshold_sauvola(gray, window_size=window_size)
    mask = gray < threshold
    mask = morphology.remove_small_objects(mask, min_size=100)
    return binary_opening(mask, structure=np.ones((2, 2))).astype(bool)


def smooth_projection(projection: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width))
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(projection.astype(float), kernel, mode="same")


def intervals_from_mask(mask: np.ndarray, min_width: int, max_width: int, pad: int, image_width: int) -> list[tuple[int, int]]:
    labeled, num = label(mask.astype(bool))
    intervals: list[tuple[int, int]] = []
    for idx in range(1, num + 1):
        xs = np.where(labeled == idx)[0]
        if xs.size == 0:
            continue
        start, end = int(xs[0]), int(xs[-1] + 1)
        width = end - start
        if min_width <= width <= max_width:
            intervals.append((max(0, start - pad), min(image_width, end + pad)))
    return intervals


def merge_close_intervals(intervals: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def estimate_orientation(crop_mask: np.ndarray) -> tuple[str, float]:
    if crop_mask.size == 0 or crop_mask.max() == 0:
        return "ambiguous", 0.0
    h_proj = crop_mask.sum(axis=0).astype(float)
    v_proj = crop_mask.sum(axis=1).astype(float)
    h_norm = h_proj / max(1.0, h_proj.max())
    v_norm = v_proj / max(1.0, v_proj.max())
    h_var = float(np.var(h_norm))
    v_var = float(np.var(v_norm))
    margin = abs(h_var - v_var)
    if margin < 0.002:
        return "ambiguous", margin
    return ("correct" if h_var >= v_var else "rotated_90_ccw"), margin


def score_interval(binary: np.ndarray, start: int, end: int) -> float:
    crop = binary[:, start:end]
    if crop.size == 0:
        return 0.0
    density = float(crop.mean())
    height_coverage = float(np.mean(crop.sum(axis=1) > 0))
    return density * 0.5 + height_coverage * 0.5


def detections_from_intervals(binary: np.ndarray, intervals: list[tuple[int, int]], method: str) -> list[ColumnDetection]:
    detections: list[ColumnDetection] = []
    for order, (start, end) in enumerate(sorted(intervals)):
        crop = binary[:, start:end]
        orientation, orientation_score = estimate_orientation(crop)
        detections.append(
            ColumnDetection(
                bbox=[int(start), 0, int(end), int(binary.shape[0])],
                score=score_interval(binary, start, end),
                method=method,
                reading_order=order,
                orientation=orientation,
                diagnostics={"orientation_margin": orientation_score},
            )
        )
    return detections


def tight_bbox_for_vertical_interval(binary: np.ndarray, start: int, end: int, pad: int) -> list[int] | None:
    crop = binary[:, start:end]
    ys, xs = np.where(crop)
    if ys.size == 0:
        return None
    x1 = max(0, start + int(xs.min()) - pad)
    x2 = min(binary.shape[1], start + int(xs.max()) + 1 + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def tight_bbox_for_horizontal_interval(binary: np.ndarray, start: int, end: int, pad: int) -> list[int] | None:
    crop = binary[start:end, :]
    ys, xs = np.where(crop)
    if ys.size == 0:
        return None
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + 1 + pad)
    y1 = max(0, start + int(ys.min()) - pad)
    y2 = min(binary.shape[0], start + int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def split_tall_boxes_by_dark_ink(
    gray: np.ndarray,
    boxes: list[list[int]],
    dark_percentile: float = 12.0,
    min_segment_height: int = 180,
    merge_gap: int = 500,
) -> list[list[int]]:
    dark = gray < np.percentile(gray, dark_percentile)
    dark = morphology.remove_small_objects(dark, min_size=20)
    image_h, image_w = gray.shape
    split_boxes: list[list[int]] = []
    for x1, y1, x2, y2 in boxes:
        crop = dark[y1:y2, x1:x2]
        if crop.size == 0 or (y2 - y1) < image_h * 0.35:
            split_boxes.append([x1, y1, x2, y2])
            continue
        row_projection = smooth_projection(crop.sum(axis=1).astype(float), width=max(5, crop.shape[0] // 420))
        if float(row_projection.max()) <= 0.0:
            split_boxes.append([x1, y1, x2, y2])
            continue
        threshold = max(float(row_projection.max()) * 0.05, 1.0)
        intervals = intervals_from_projection(
            row_projection,
            threshold,
            min_width=min_segment_height,
            max_width=crop.shape[0],
            pad=12,
            axis_size=crop.shape[0],
        )
        intervals = merge_close_intervals(intervals, max_gap=merge_gap)

        segments: list[list[int]] = []
        for start, end in intervals:
            segment = crop[start:end, :]
            ys, xs = np.where(segment)
            if ys.size == 0:
                continue
            sx1 = max(0, x1 + int(xs.min()) - 25)
            sx2 = min(image_w, x1 + int(xs.max()) + 1 + 25)
            sy1 = max(0, y1 + start + int(ys.min()) - 35)
            sy2 = min(image_h, y1 + start + int(ys.max()) + 1 + 35)
            if sx2 > sx1 and sy2 > sy1 and sy2 - sy1 >= min_segment_height:
                segments.append([sx1, sy1, sx2, sy2])
        split_boxes.extend(segments if 1 < len(segments) <= 3 else [[x1, y1, x2, y2]])
    return split_boxes


def intervals_from_projection(
    projection: np.ndarray,
    threshold: float,
    min_width: int,
    max_width: int,
    pad: int,
    axis_size: int,
) -> list[tuple[int, int]]:
    return intervals_from_mask(
        projection > threshold,
        min_width=min_width,
        max_width=max_width,
        pad=pad,
        image_width=axis_size,
    )


def split_wide_interval_by_valleys(
    projection: np.ndarray,
    interval: tuple[int, int],
    target_width: int,
    min_width: int,
) -> list[tuple[int, int]]:
    start, end = interval
    width = end - start
    if width <= target_width * 1.8:
        return [interval]

    expected = max(2, int(round(width / max(1, target_width))))
    expected = min(expected, max(2, width // max(1, min_width)))
    if expected <= 1:
        return [interval]

    segment = projection[start:end]
    cuts: list[int] = []
    for idx in range(1, expected):
        center = start + int(round(width * idx / expected))
        radius = max(min_width // 2, target_width // 3)
        left = max(start + min_width, center - radius)
        right = min(end - min_width, center + radius)
        if right <= left:
            continue
        local = projection[left:right]
        cut = left + int(np.argmin(local))
        if all(abs(cut - old) >= min_width for old in cuts):
            cuts.append(cut)

    if not cuts:
        return [interval]
    points = [start] + sorted(cuts) + [end]
    pieces = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    return [(a, b) for a, b in pieces if b - a >= min_width]


def merge_overlapping_intervals(intervals: list[tuple[int, int]], overlap_ratio: float = 0.55) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        overlap = max(0, min(last_end, end) - max(last_start, start))
        smaller = max(1, min(last_end - last_start, end - start))
        if overlap / smaller >= overlap_ratio:
            merged[-1] = (min(last_start, start), max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def count_prominent_peaks(projection: np.ndarray, min_distance: int, prominence_scale: float) -> int:
    if projection.size == 0:
        return 0
    prominence = max(float(np.std(projection)) * prominence_scale, float(np.max(projection)) * 0.01, 1.0)
    peaks, _ = signal.find_peaks(projection, distance=max(1, int(min_distance)), prominence=prominence)
    return int(len(peaks))


def detections_from_bboxes(
    binary: np.ndarray,
    bboxes: list[list[int]],
    method: str,
    orientation: str = "correct",
    order_axis: str = "x",
) -> list[ColumnDetection]:
    detections: list[ColumnDetection] = []
    if order_axis == "y":
        ordered = sorted(bboxes, key=lambda box: (box[1], box[0]))
    elif order_axis == "x_desc":
        ordered = sorted(bboxes, key=lambda box: (-(box[0] + box[2]) / 2.0, box[1]))
    else:
        ordered = sorted(bboxes, key=lambda box: (box[0], box[1]))
    for order, bbox in enumerate(ordered):
        x1, y1, x2, y2 = bbox
        crop = binary[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else np.zeros((0, 0), dtype=bool)
        ink_density = float(crop.mean()) if crop.size else 0.0
        row_coverage = float(np.mean(crop.sum(axis=1) > 0)) if crop.size else 0.0
        column_coverage = float(np.mean(crop.sum(axis=0) > 0)) if crop.size else 0.0
        detections.append(
            ColumnDetection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                score=score_interval(binary[y1:y2, :], x1, x2) if y2 > y1 else 0.0,
                method=method,
                reading_order=order,
                orientation=orientation,
                diagnostics={
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "ink_density": ink_density,
                    "row_coverage": row_coverage,
                    "column_coverage": column_coverage,
                },
            )
        )
    return detections


def should_keep_layout_detection(det: ColumnDetection) -> bool:
    """Keep high-confidence columns and recover a few strong low-confidence candidates."""
    if det.score >= 0.30:
        if det.diagnostics is not None:
            det.diagnostics["filter_reason"] = "score_ge_0.30"
        return True

    x1, y1, x2, y2 = det.bbox
    width = x2 - x1
    height = y2 - y1
    aspect = height / max(1, width)
    diagnostics = det.diagnostics or {}
    ink_density = float(diagnostics.get("ink_density", 0.0))
    row_coverage = float(diagnostics.get("row_coverage", 0.0))
    column_coverage = float(diagnostics.get("column_coverage", 0.0))
    if (
        0.18 <= det.score < 0.30
        and ink_density >= 0.023
        and row_coverage >= 0.36
        and column_coverage >= 0.69
        and 150 <= width <= 380
        and height >= 500
        and aspect <= 8.0
    ):
        if det.diagnostics is not None:
            det.diagnostics["filter_reason"] = "low_score_strong_column_recovery"
            det.diagnostics["aspect"] = float(aspect)
        return True
    return False


def cluster_boxes_by_axis(boxes: list[list[int]], axis: str, max_gap: int) -> list[list[list[int]]]:
    if not boxes:
        return []
    if axis == "y":
        key = lambda b: (b[1] + b[3]) / 2.0
    else:
        key = lambda b: (b[0] + b[2]) / 2.0
    ordered = sorted(boxes, key=key)
    clusters = [[ordered[0]]]
    last_center = key(ordered[0])
    for box in ordered[1:]:
        center = key(box)
        if abs(center - last_center) <= max_gap:
            clusters[-1].append(box)
        else:
            clusters.append([box])
        last_center = center
    return clusters


def merge_boxes_to_band(boxes: list[list[int]], pad: int, image_size: tuple[int, int]) -> list[int]:
    image_h, image_w = image_size
    x1 = max(0, min(box[0] for box in boxes) - pad)
    y1 = max(0, min(box[1] for box in boxes) - pad)
    x2 = min(image_w, max(box[2] for box in boxes) + pad)
    y2 = min(image_h, max(box[3] for box in boxes) + pad)
    return [int(x1), int(y1), int(x2), int(y2)]


def split_boxes_by_axis_gap(boxes: list[list[int]], axis: str, gap: int) -> list[list[list[int]]]:
    if not boxes:
        return []
    if axis == "x":
        ordered = sorted(boxes, key=lambda b: b[0])
        end_of = lambda b: b[2]
        start_of = lambda b: b[0]
    else:
        ordered = sorted(boxes, key=lambda b: b[1])
        end_of = lambda b: b[3]
        start_of = lambda b: b[1]
    groups = [[ordered[0]]]
    current_end = end_of(ordered[0])
    for box in ordered[1:]:
        if start_of(box) - current_end > gap:
            groups.append([box])
            current_end = end_of(box)
        else:
            groups[-1].append(box)
            current_end = max(current_end, end_of(box))
    return groups


def detect_horizontal_special_case(binary: np.ndarray, image_h: int, image_w: int) -> list[list[int]]:
    comps = measure.regionprops(measure.label(binary))
    candidates: list[list[int]] = []
    for comp in comps:
        minr, minc, maxr, maxc = comp.bbox
        area = comp.area
        width = maxc - minc
        height = maxr - minr
        if area < 1000 or height < 40 or height > 600 or width < 50 or width > 6000:
            continue
        candidates.append([int(minc), int(minr), int(maxc), int(maxr)])

    if image_h < 6500 or image_w < 8500:
        return []
    if len(candidates) < 120:
        return []

    horizontal_projection = smooth_projection(binary.sum(axis=1), width=max(15, image_h // 220))
    peaks, _ = signal.find_peaks(
        horizontal_projection,
        distance=max(120, image_h // 32),
        prominence=max(float(np.std(horizontal_projection)) * 0.12, float(np.max(horizontal_projection)) * 0.005, 1.0),
    )
    peaks = sorted(int(p) for p in peaks)
    if len(peaks) < 19:
        return []

    bounds = [0]
    for idx, peak in enumerate(peaks[:-1]):
        bounds.append((peak + peaks[idx + 1]) // 2)
    bounds.append(image_h)
    intervals = list(zip(bounds[:-1], bounds[1:]))

    refined_boxes: list[list[int]] = []
    for start, end in intervals:
        band_candidates = [box for box in candidates if not (box[3] < start or box[1] > end)]
        if not band_candidates:
            continue
        for box in band_candidates:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width < image_w * 0.005 or height < image_h * 0.002:
                continue
            crop = binary[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            density = float(crop.mean())
            if density < 0.001:
                continue
            refined_boxes.append([x1, y1, x2, y2])
    if len(refined_boxes) < 8:
        return []
    refined_boxes = sorted(refined_boxes, key=lambda box: (box[1], box[0]))
    return refined_boxes


def detect_sauvola_projection(image_path: str | Path) -> list[ColumnDetection]:
    gray = load_gray(image_path)
    binary = sauvola_mask(gray, window_size=25)
    projection = smooth_projection(binary.sum(axis=0), width=30)
    threshold = projection.mean() * 0.15
    intervals = intervals_from_mask(
        projection > threshold,
        min_width=max(30, gray.shape[1] // 100),
        max_width=max(80, gray.shape[1] // 2),
        pad=15,
        image_width=gray.shape[1],
    )
    return detections_from_intervals(binary, intervals, "sauvola_projection")


def detect_connected_components(image_path: str | Path) -> list[ColumnDetection]:
    gray = load_gray(image_path)
    binary = sauvola_mask(gray, window_size=31)
    binary = binary_closing(binary, structure=np.ones((11, 3))).astype(bool)
    components = measure.regionprops(measure.label(binary))
    intervals: list[tuple[int, int]] = []
    for comp in components:
        minr, minc, maxr, maxc = comp.bbox
        width = maxc - minc
        height = maxr - minr
        aspect = height / max(1, width)
        if height > gray.shape[0] * 0.15 and 20 <= width <= gray.shape[1] * 0.6 and aspect > 1.5:
            intervals.append((minc, maxc))
    intervals = merge_close_intervals(intervals, max_gap=max(10, gray.shape[1] // 100))
    return detections_from_intervals(binary, intervals, "connected_components")


def detect_layout_aware(image_path: str | Path) -> list[ColumnDetection]:
    gray = load_gray(image_path)
    binary = sauvola_mask(gray, window_size=25)
    binary = morphology.remove_small_objects(binary, min_size=60)

    image_h, image_w = gray.shape
    special_horizontal_boxes = detect_horizontal_special_case(binary, image_h, image_w)
    if len(special_horizontal_boxes) >= 20:
        return detections_from_bboxes(
            binary,
            special_horizontal_boxes,
            "layout_aware",
            orientation="rotated_90_ccw",
            order_axis="y",
        )

    pad = max(8, image_w // 600)
    min_col_width = max(90, image_w // 75)
    target_col_width = max(220, image_w // 28)
    max_col_width = max(520, image_w // 8)

    vertical_projection = smooth_projection(binary.sum(axis=0), width=max(15, image_w // 220))
    horizontal_projection = smooth_projection(binary.sum(axis=1), width=max(15, image_h // 220))

    vertical_peak_count = count_prominent_peaks(
        vertical_projection,
        min_distance=max(min_col_width, image_w // 24),
        prominence_scale=0.20,
    )
    horizontal_peak_count = count_prominent_peaks(
        horizontal_projection,
        min_distance=max(max(45, image_h // 120), image_h // 32),
        prominence_scale=0.16,
    )

    vertical_thresholds = [
        max(vertical_projection.mean() * factor, vertical_projection.max() * peak_factor)
        for factor, peak_factor in ((0.11, 0.010), (0.14, 0.012), (0.18, 0.014))
    ]
    vertical_intervals: list[tuple[int, int]] = []
    for threshold in vertical_thresholds:
        raw = intervals_from_projection(
            vertical_projection,
            threshold,
            min_width=min_col_width,
            max_width=max(image_w // 2, max_col_width * 4),
            pad=pad,
            axis_size=image_w,
        )
        for interval in raw:
            vertical_intervals.extend(
                split_wide_interval_by_valleys(
                    vertical_projection,
                    interval,
                    target_width=target_col_width,
                    min_width=min_col_width,
                )
            )
    vertical_intervals = merge_overlapping_intervals(vertical_intervals)

    vertical_boxes: list[list[int]] = []
    for start, end in vertical_intervals:
        width = end - start
        if width < min_col_width or width > max_col_width * 1.35:
            continue
        bbox = tight_bbox_for_vertical_interval(binary, start, end, pad=pad)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        height_coverage = (y2 - y1) / max(1, image_h)
        ink_density = float(crop.mean())
        if height_coverage < 0.13 or ink_density < 0.002:
            continue
        if (y2 - y1) < image_h * 0.18 and (x2 - x1) < target_col_width * 0.9:
            continue
        vertical_boxes.append(bbox)

    horizontal_thresholds = [
        max(horizontal_projection.mean() * factor, horizontal_projection.max() * peak_factor)
        for factor, peak_factor in ((0.07, 0.006), (0.09, 0.008), (0.12, 0.010))
    ]
    horizontal_intervals: list[tuple[int, int]] = []
    min_band_height = max(40, image_h // 120)
    target_band_height = max(180, image_h // 18)
    max_band_height = max(650, image_h // 5)
    for threshold in horizontal_thresholds:
        raw = intervals_from_projection(
            horizontal_projection,
            threshold,
            min_width=min_band_height,
            max_width=max_band_height,
            pad=max(6, image_h // 800),
            axis_size=image_h,
        )
        for interval in raw:
            horizontal_intervals.extend(
                split_wide_interval_by_valleys(
                    horizontal_projection,
                    interval,
                    target_width=target_band_height,
                    min_width=min_band_height,
                )
            )
    horizontal_intervals = merge_overlapping_intervals(horizontal_intervals)

    horizontal_boxes: list[list[int]] = []
    for start, end in horizontal_intervals:
        bbox = tight_bbox_for_horizontal_interval(binary, start, end, pad=max(6, image_h // 800))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width_ratio = (x2 - x1) / max(1, image_w)
        height_ratio = (y2 - y1) / max(1, image_h)
        if width_ratio < 0.10 or height_ratio < 0.005:
            continue
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if float(crop.mean()) < 0.0015:
            continue
        horizontal_boxes.append(bbox)

    vertical_aspect = float(np.median([(box[3] - box[1]) / max(1, box[2] - box[0]) for box in vertical_boxes])) if vertical_boxes else 0.0
    horizontal_aspect = float(np.median([(box[2] - box[0]) / max(1, box[3] - box[1]) for box in horizontal_boxes])) if horizontal_boxes else 0.0
    horizontal_score = (
        horizontal_peak_count * 1.4
        + len(horizontal_boxes) * 1.2
        + max(0.0, horizontal_aspect - 1.5) * 2.0
    )
    vertical_score = (
        vertical_peak_count * 1.4
        + len(vertical_boxes) * 1.0
        + max(0.0, vertical_aspect - 1.5) * 1.5
    )
    tall_ratio = (
        sum(1 for box in vertical_boxes if (box[3] - box[1]) > image_h * 0.55) / len(vertical_boxes)
        if vertical_boxes
        else 0.0
    )
    if image_h > image_w * 1.2 and 12 <= len(vertical_boxes) <= 20 and tall_ratio >= 0.70:
        vertical_boxes = split_tall_boxes_by_dark_ink(gray, vertical_boxes)
        return detections_from_bboxes(binary, vertical_boxes, "layout_aware", orientation="correct", order_axis="x_desc")
    if len(vertical_boxes) <= 5:
        return detections_from_bboxes(binary, vertical_boxes, "layout_aware", orientation="correct", order_axis="y")
    return detections_from_bboxes(binary, vertical_boxes, "layout_aware", orientation="correct", order_axis="x")


def detect_columns(image_path: str | Path, method: str) -> list[dict]:
    if method == "sauvola_projection":
        detections = detect_sauvola_projection(image_path)
    elif method == "connected_components":
        detections = detect_connected_components(image_path)
    elif method == "proposed":
        detections = detect_layout_aware(image_path)
    else:
        raise ValueError(f"Unknown method: {method}")
    if method == "proposed":
        detections = [det for det in detections if should_keep_layout_detection(det)]
        for order, det in enumerate(detections):
            det.reading_order = order
    return [det.to_dict() for det in detections]
