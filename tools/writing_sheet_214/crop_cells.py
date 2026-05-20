from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


COLS = 12
ROWS = 18
GLYPH_COUNT = 214
DEFAULT_CELL_MARGIN_RATIO = 0.03


def code_name(char: str) -> str:
    return f"U{ord(char):04X}"


def cell_bounds(
    image_shape: tuple[int, ...],
    grid_bbox: tuple[int, int, int, int],
    row: int,
    col: int,
    margin_ratio: float = DEFAULT_CELL_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = grid_bbox
    grid_width = x1 - x0
    grid_height = y1 - y0

    cell_x0 = x0 + int(round(col * grid_width / COLS))
    cell_x1 = x0 + int(round((col + 1) * grid_width / COLS))
    cell_y0 = y0 + int(round(row * grid_height / ROWS))
    cell_y1 = y0 + int(round((row + 1) * grid_height / ROWS))

    cell_width = cell_x1 - cell_x0
    cell_height = cell_y1 - cell_y0
    margin_x = int(round(cell_width * margin_ratio))
    margin_y = int(round(cell_height * margin_ratio))

    height, width = image_shape[:2]
    crop_x0 = min(max(cell_x0 + margin_x, 0), width)
    crop_x1 = min(max(cell_x1 - margin_x, 0), width)
    crop_y0 = min(max(cell_y0 + margin_y, 0), height)
    crop_y1 = min(max(cell_y1 - margin_y, 0), height)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError(f"invalid crop box for row={row}, col={col}")
    return crop_x0, crop_y0, crop_x1, crop_y1


def crop_cells(
    image: np.ndarray,
    grid_bbox: tuple[int, int, int, int],
    chars: list[str],
    output_dir: str | Path,
    user_id: str,
    margin_ratio: float = DEFAULT_CELL_MARGIN_RATIO,
) -> list[dict[str, object]]:
    if len(chars) != GLYPH_COUNT:
        raise ValueError(f"expected {GLYPH_COUNT} chars, got {len(chars)}")

    cell_dir = Path(output_dir) / "cropped_214" / user_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, char in enumerate(chars, start=1):
        row = (index - 1) // COLS
        col = (index - 1) % COLS
        x0, y0, x1, y1 = cell_bounds(image.shape, grid_bbox, row, col, margin_ratio)
        cell = image[y0:y1, x0:x1]
        filename = f"{index:03d}_{code_name(char)}_cell.png"
        cell_path = cell_dir / filename
        if not cv2.imwrite(str(cell_path), cell):
            raise ValueError(f"failed to write cell image: {cell_path}")
        records.append(
            {
                "index": index,
                "char": char,
                "unicode": f"U+{ord(char):04X}",
                "row": row,
                "col": col,
                "cell_x0": x0,
                "cell_y0": y0,
                "cell_x1": x1,
                "cell_y1": y1,
                "cell_path": str(cell_path),
                "ink_bbox_x0": "",
                "ink_bbox_y0": "",
                "ink_bbox_x1": "",
                "ink_bbox_y1": "",
                "ink_area": "",
                "warning": "",
            }
        )
    return records


def write_crop_metadata(records: list[dict[str, object]], output_dir: str | Path, user_id: str) -> Path:
    metadata_path = Path(output_dir) / "cropped_214" / user_id / "crop_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "char",
        "unicode",
        "row",
        "col",
        "cell_x0",
        "cell_y0",
        "cell_x1",
        "cell_y1",
        "ink_bbox_x0",
        "ink_bbox_y0",
        "ink_bbox_x1",
        "ink_bbox_y1",
        "ink_area",
        "warning",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return metadata_path
