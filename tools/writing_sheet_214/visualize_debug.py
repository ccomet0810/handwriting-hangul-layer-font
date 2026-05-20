from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crop_cells import COLS, ROWS


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save_detected_sheet_preview(
    image: np.ndarray,
    grid_bbox: tuple[int, int, int, int],
    output_dir: str | Path,
    user_id: str,
) -> Path:
    debug_dir = Path(output_dir) / "cropped_214" / user_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    x0, y0, x1, y1 = grid_bbox
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 3)
    path = debug_dir / "detected_sheet_preview.png"
    if not cv2.imwrite(str(path), preview):
        raise ValueError(f"failed to write debug preview: {path}")
    return path


def save_crop_grid_preview(
    image: np.ndarray,
    grid_bbox: tuple[int, int, int, int],
    records: list[dict[str, object]],
    output_dir: str | Path,
    user_id: str,
) -> Path:
    debug_dir = Path(output_dir) / "cropped_214" / user_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _font(max(10, int((grid_bbox[3] - grid_bbox[1]) / ROWS * 0.18)))
    x0, y0, x1, y1 = grid_bbox
    grid_w = x1 - x0
    grid_h = y1 - y0

    for row in range(ROWS + 1):
        y = y0 + round(row * grid_h / ROWS)
        draw.line((x0, y, x1, y), fill=(40, 120, 230), width=1)
    for col in range(COLS + 1):
        x = x0 + round(col * grid_w / COLS)
        draw.line((x, y0, x, y1), fill=(40, 120, 230), width=1)

    for record in records:
        label = f"{int(record['index']):03d} {record['char']}"
        tx = int(record["cell_x0"]) + 3
        ty = int(record["cell_y0"]) + 3
        draw.rectangle((tx - 1, ty - 1, tx + 54, ty + 18), fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(210, 20, 20), font=font)

    path = debug_dir / "crop_grid_preview.png"
    output = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), output):
        raise ValueError(f"failed to write crop grid preview: {path}")
    return path


def save_normalized_contact_sheet(
    records: list[dict[str, object]],
    output_dir: str | Path,
    user_id: str,
    thumb_size: int = 96,
) -> Path:
    debug_dir = Path(output_dir) / "glyphs_214" / user_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (COLS * thumb_size, ROWS * thumb_size), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(10, thumb_size // 8))

    for record in records:
        index = int(record["index"])
        row = int(record["row"])
        col = int(record["col"])
        glyph_path = record.get("glyph_path")
        x = col * thumb_size
        y = row * thumb_size
        if glyph_path and Path(str(glyph_path)).exists():
            glyph = Image.open(str(glyph_path)).convert("L").resize(
                (thumb_size, thumb_size),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(Image.merge("RGB", (glyph, glyph, glyph)), (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(210, 210, 210))
        draw.text((x + 3, y + 3), f"{index:03d} {record['char']}", fill=(210, 20, 20), font=font)

    path = debug_dir / "normalized_contact_sheet.png"
    sheet.save(path)
    return path
