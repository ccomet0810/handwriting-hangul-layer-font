from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from crop_cells import code_name


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_for_ink(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.25) if min(gray.shape[:2]) >= 3 else gray
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    black_pixels = int(np.count_nonzero(binary == 0))
    white_pixels = int(np.count_nonzero(binary == 255))
    if black_pixels > white_pixels:
        gray = cv2.bitwise_not(gray)
        binary = cv2.bitwise_not(binary)
    return gray, binary


def threshold_final(gray: np.ndarray) -> np.ndarray:
    if min(gray.shape[:2]) >= 3:
        gray = cv2.GaussianBlur(gray, (3, 3), 0.35)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    black_pixels = int(np.count_nonzero(binary == 0))
    white_pixels = int(np.count_nonzero(binary == 255))
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)
    return binary


def ink_bbox(binary: np.ndarray, min_ink_area: int) -> tuple[tuple[int, int, int, int] | None, int]:
    ys, xs = np.where(binary == 0)
    ink_area = int(len(xs))
    if ink_area < min_ink_area:
        return None, ink_area
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), ink_area


def normalize_cell(
    image: np.ndarray,
    image_size: int = 512,
    padding_ratio: float = 0.18,
    min_ink_area: int = 12,
) -> tuple[np.ndarray, dict[str, object]]:
    gray, binary = threshold_for_ink(to_gray(image))
    bbox, area = ink_bbox(binary, min_ink_area=min_ink_area)
    metrics: dict[str, object] = {
        "ink_bbox_x0": "",
        "ink_bbox_y0": "",
        "ink_bbox_x1": "",
        "ink_bbox_y1": "",
        "ink_area": area,
        "warning": "",
    }
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    if bbox is None:
        metrics["warning"] = f"empty_or_too_little_ink(area={area})"
        return canvas, metrics

    x0, y0, x1, y1 = bbox
    metrics.update(
        {
            "ink_bbox_x0": x0,
            "ink_bbox_y0": y0,
            "ink_bbox_x1": x1,
            "ink_bbox_y1": y1,
        }
    )

    bbox_width = x1 - x0
    bbox_height = y1 - y0
    pad = max(4, int(round(max(bbox_width, bbox_height) * 0.08)))
    crop_x0 = max(0, x0 - pad)
    crop_y0 = max(0, y0 - pad)
    crop_x1 = min(gray.shape[1], x1 + pad)
    crop_y1 = min(gray.shape[0], y1 + pad)
    crop = gray[crop_y0:crop_y1, crop_x0:crop_x1]

    target_size = int(round(image_size * (1.0 - padding_ratio * 2.0)))
    target_size = max(1, min(image_size, target_size))
    scale = min(target_size / max(1, crop.shape[1]), target_size / max(1, crop.shape[0]))
    new_width = max(1, int(round(crop.shape[1] * scale)))
    new_height = max(1, int(round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(crop, (new_width, new_height), interpolation=interpolation)
    resized = threshold_final(resized)

    x_offset = (image_size - new_width) // 2
    y_offset = (image_size - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas, metrics


def normalize_glyphs(
    records: list[dict[str, object]],
    output_dir: str | Path,
    user_id: str,
    image_size: int = 512,
    padding_ratio: float = 0.18,
    min_ink_area: int = 12,
) -> list[dict[str, object]]:
    glyph_dir = Path(output_dir) / "glyphs_214" / user_id
    glyph_dir.mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "masks_layer_214" / user_id).mkdir(parents=True, exist_ok=True)

    for record in records:
        cell_path = Path(str(record["cell_path"]))
        image = cv2.imread(str(cell_path), cv2.IMREAD_COLOR)
        if image is None:
            record["warning"] = "failed_to_read_cell"
            continue
        normalized, metrics = normalize_cell(
            image,
            image_size=image_size,
            padding_ratio=padding_ratio,
            min_ink_area=min_ink_area,
        )
        record.update(metrics)
        index = int(record["index"])
        char = str(record["char"])
        glyph_path = glyph_dir / f"{index:03d}_{code_name(char)}.png"
        if not cv2.imwrite(str(glyph_path), normalized):
            raise ValueError(f"failed to write normalized glyph: {glyph_path}")
        record["glyph_path"] = str(glyph_path)
    return records
