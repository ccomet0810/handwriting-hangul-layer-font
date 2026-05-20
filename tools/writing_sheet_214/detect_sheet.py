from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


GridBBox = tuple[int, int, int, int]


def parse_grid_bbox(value: str) -> GridBBox:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--grid_bbox must be x0,y0,x1,y1")
    x0, y0, x1, y1 = [int(part) for part in parts]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid grid bbox: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def clamp_bbox(bbox: GridBBox, image_shape: tuple[int, ...]) -> GridBBox:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def detect_fixed(image: np.ndarray, grid_bbox: str | GridBBox) -> GridBBox:
    bbox = parse_grid_bbox(grid_bbox) if isinstance(grid_bbox, str) else grid_bbox
    return clamp_bbox(bbox, image.shape)


def _bbox_from_mask(mask: np.ndarray, min_area_ratio: float = 0.02) -> GridBBox | None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = mask.shape[:2]
    min_area = height * width * min_area_ratio
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        if 0.35 <= aspect <= 1.25:
            candidates.append((area, (x, y, x + w, y + h)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def detect_auto(image: np.ndarray) -> GridBBox:
    """Best-effort sheet/grid detection.

    The production path for this tool is fixed mode. Auto mode first tries a blue
    border mask, then falls back to dark grid-line projection similar to
    handwrite2350's simple grid bbox detector.
    """
    bgr = image
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (85, 35, 35), (140, 255, 255))
    bbox = _bbox_from_mask(blue_mask, min_area_ratio=0.015)
    if bbox is not None:
        return clamp_bbox(bbox, image.shape)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = gray < 175
    height, width = binary.shape
    row_counts = binary.sum(axis=1)
    col_counts = binary.sum(axis=0)
    row_indexes = np.where(row_counts > width * 0.25)[0]
    col_indexes = np.where(col_counts > height * 0.25)[0]
    if len(row_indexes) >= 2 and len(col_indexes) >= 2:
        return clamp_bbox(
            (
                int(col_indexes.min()),
                int(row_indexes.min()),
                int(col_indexes.max()) + 1,
                int(row_indexes.max()) + 1,
            ),
            image.shape,
        )

    raise ValueError(
        "auto detection failed. Re-run with --mode fixed --grid_bbox x0,y0,x1,y1."
    )


def detect_sheet(image_path: str | Path, mode: str, grid_bbox: str | None = None) -> tuple[np.ndarray, GridBBox]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read input image: {image_path}")

    if mode == "fixed":
        if not grid_bbox:
            raise ValueError("fixed mode requires --grid_bbox x0,y0,x1,y1")
        return image, detect_fixed(image, grid_bbox)
    if mode == "auto":
        return image, detect_auto(image)
    raise ValueError(f"unsupported mode: {mode}")
