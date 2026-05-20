from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from crop_cells import GLYPH_COUNT, crop_cells, write_crop_metadata
from detect_sheet import detect_sheet
from normalize_glyphs import normalize_glyphs
from visualize_debug import (
    save_crop_grid_preview,
    save_detected_sheet_preview,
    save_normalized_contact_sheet,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
CHARS_PATH = CONFIG_DIR / "chars_214.txt"
MAPPING_PATH = CONFIG_DIR / "mapping_214.csv"


def load_chars(path: str | Path = CHARS_PATH) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    chars = [char for char in text if not char.isspace()]
    if len(chars) != GLYPH_COUNT:
        raise ValueError(f"expected {GLYPH_COUNT} chars in {path}, got {len(chars)}")
    return chars


def ensure_dataset_dirs(out_dir: str | Path) -> None:
    for name in ["raw_sheets", "cropped_214", "glyphs_214", "masks_layer_214"]:
        (Path(out_dir) / name).mkdir(parents=True, exist_ok=True)


def copy_mapping_snapshot(out_dir: str | Path) -> Path:
    dst = Path(out_dir) / "mapping_214.csv"
    if MAPPING_PATH.exists():
        shutil.copy2(MAPPING_PATH, dst)
    return dst


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop and normalize a 12x18 / 214-char writing sheet.")
    parser.add_argument("--input", required=True, help="Path to one scanned/photo writing sheet image.")
    parser.add_argument("--out_dir", default="datasets", help="Dataset root directory.")
    parser.add_argument("--user_id", required=True, help="Output source id, e.g. user_001.")
    parser.add_argument("--mode", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--grid_bbox", help="Fixed grid bbox as x0,y0,x1,y1.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--cell_margin", type=float, default=0.03)
    parser.add_argument("--padding_ratio", type=float, default=0.18)
    parser.add_argument("--min_ink_area", type=int, default=12)
    parser.add_argument("--chars", default=str(CHARS_PATH))
    return parser


def run(args: argparse.Namespace) -> dict[str, Path | int | tuple[int, int, int, int]]:
    chars = load_chars(args.chars)
    ensure_dataset_dirs(args.out_dir)
    mapping_path = copy_mapping_snapshot(args.out_dir)

    image, grid_bbox = detect_sheet(args.input, args.mode, args.grid_bbox)
    records = crop_cells(
        image,
        grid_bbox,
        chars,
        args.out_dir,
        args.user_id,
        margin_ratio=args.cell_margin,
    )
    save_detected_sheet_preview(image, grid_bbox, args.out_dir, args.user_id)
    save_crop_grid_preview(image, grid_bbox, records, args.out_dir, args.user_id)

    records = normalize_glyphs(
        records,
        args.out_dir,
        args.user_id,
        image_size=args.image_size,
        padding_ratio=args.padding_ratio,
        min_ink_area=args.min_ink_area,
    )
    metadata_path = write_crop_metadata(records, args.out_dir, args.user_id)
    save_normalized_contact_sheet(records, args.out_dir, args.user_id)
    (Path(args.out_dir) / "masks_layer_214" / args.user_id).mkdir(parents=True, exist_ok=True)
    return {
        "chars": len(chars),
        "mapping_path": mapping_path,
        "metadata_path": metadata_path,
        "grid_bbox": grid_bbox,
        "cell_count": len(records),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(f"chars: {result['chars']}")
    print(f"grid_bbox: {result['grid_bbox']}")
    print(f"cell_count: {result['cell_count']}")
    print(f"mapping_csv: {result['mapping_path']}")
    print(f"crop_metadata: {result['metadata_path']}")
    print(f"layer_mask_dir: {Path(args.out_dir) / 'masks_layer_214' / args.user_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
