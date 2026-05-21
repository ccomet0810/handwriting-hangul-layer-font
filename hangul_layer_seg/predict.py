from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_predict_deps():
    import numpy as np
    import torch
    from PIL import Image

    try:
        from .dataset import IMAGE_EXTS
        from .models import UNet
        from .utils.layer_mask_io import LAYERS, make_layer_overlay, save_prediction_layers
    except ImportError:
        from dataset import IMAGE_EXTS
        from models import UNet
        from utils.layer_mask_io import LAYERS, make_layer_overlay, save_prediction_layers
    return np, torch, Image, IMAGE_EXTS, UNet, LAYERS, make_layer_overlay, save_prediction_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict independent L/V/T layer masks.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mapping_csv", default=None, help="Optional mapping_214.csv path for glyph metadata lookup.")
    parser.add_argument(
        "--suppress_no_final_t",
        action="store_true",
        help="Set the T channel probability to 0 for glyphs with TIndex == 0 in --mapping_csv.",
    )
    parser.add_argument("--restrict_to_ink", action="store_true", help="Limit L/V/T probabilities to the source glyph ink mask.")
    parser.add_argument("--ink_threshold", type=int, default=245, help="Pixels darker than this value are treated as ink.")
    parser.add_argument("--ink_dilate", type=int, default=1, help="Dilate the ink mask by this many 3x3 iterations.")
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=0,
        help="Remove per-layer connected components smaller than this area after thresholding. 0 disables.",
    )
    return parser.parse_args()


def resolve_device(torch, name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(input_path: Path, image_exts: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)


def mapping_key_from_row(row: dict[str, str]) -> str:
    index = int(row["index"])
    unicode_value = row["unicode"].replace("U+", "").upper()
    return f"{index:03d}_U{unicode_value}".lower()


def load_mapping_by_glyph_id(mapping_csv: Path) -> dict[str, dict[str, str]]:
    rows_by_glyph_id: dict[str, dict[str, str]] = {}
    with mapping_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows_by_glyph_id[mapping_key_from_row(row)] = row
    return rows_by_glyph_id


def find_mapping_row(rows_by_glyph_id: dict[str, dict[str, str]], glyph_id: Path) -> dict[str, str] | None:
    return rows_by_glyph_id.get(glyph_id.name.lower())


def maybe_suppress_no_final_t(probs, mapping_row: dict[str, str] | None) -> bool:
    if mapping_row is None:
        return False
    if int(mapping_row["TIndex"]) != 0:
        return False
    probs[2] = 0.0
    return True


def dilate_mask(mask, iterations: int, np):
    out = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def make_ink_mask(original, ink_threshold: int, ink_dilate: int, np):
    gray = np.array(original, dtype=np.uint8)
    mask = gray < ink_threshold
    return dilate_mask(mask, ink_dilate, np)


def restrict_probs_to_ink(probs, ink_mask) -> int:
    before = int((probs > 0).sum())
    probs[:, ~ink_mask] = 0.0
    return before - int((probs > 0).sum())


def remove_small_components_from_mask(mask, min_area: int, np):
    visited = np.zeros(mask.shape, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)
    removed_components = 0
    removed_pixels = 0
    height, width = mask.shape

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        component = []
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(component) >= min_area:
            ys, xs = zip(*component)
            keep[ys, xs] = True
        else:
            removed_components += 1
            removed_pixels += len(component)

    return keep, removed_components, removed_pixels


def remove_small_layer_components(probs, threshold: float, min_area: int, np) -> dict[str, int]:
    stats = {"removed_components": 0, "removed_pixels": 0}
    if min_area <= 1:
        return stats

    for idx in range(probs.shape[0]):
        binary = probs[idx] >= threshold
        keep, removed_components, removed_pixels = remove_small_components_from_mask(binary, min_area, np)
        probs[idx, binary & ~keep] = 0.0
        stats["removed_components"] += removed_components
        stats["removed_pixels"] += removed_pixels
    return stats


def predict_one(model, path: Path, image_size: int, device, np, torch, Image) -> tuple:
    original = Image.open(path).convert("L")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()
    resized_probs = []
    for idx in range(probs.shape[0]):
        img = Image.fromarray((probs[idx] * 255).clip(0, 255).astype("uint8"), mode="L")
        img = img.resize(original.size, Image.BILINEAR)
        resized_probs.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(resized_probs, axis=0), original


def main() -> None:
    args = parse_args()
    np, torch, Image, IMAGE_EXTS, UNet, LAYERS, make_layer_overlay, save_prediction_layers = import_predict_deps()
    device = resolve_device(torch, args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet(**ckpt.get("model_kwargs", {"in_channels": 1, "num_classes": 3, "base_channels": 32}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    image_size = args.image_size or ckpt.get("image_size", 128)

    input_path = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    mapping_rows = None
    if args.suppress_no_final_t:
        if args.mapping_csv is None:
            raise SystemExit("--suppress_no_final_t requires --mapping_csv")
        mapping_rows = load_mapping_by_glyph_id(Path(args.mapping_csv))
    paths = list_images(input_path, IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No input images found under {input_path}")

    for path in paths:
        rel = path.relative_to(input_path) if input_path.is_dir() else Path(path.name)
        glyph_id = rel.with_suffix("")
        probs, original = predict_one(model, path, image_size, device, np, torch, Image)
        mapping_row = find_mapping_row(mapping_rows, glyph_id) if mapping_rows is not None else None
        if args.suppress_no_final_t and mapping_row is None:
            raise SystemExit(f"No mapping row found for glyph_id {glyph_id.name} in {args.mapping_csv}")
        t_suppressed = maybe_suppress_no_final_t(probs, mapping_row) if args.suppress_no_final_t else False
        ink_stats = None
        if args.restrict_to_ink:
            ink_mask = make_ink_mask(original, args.ink_threshold, args.ink_dilate, np)
            ink_stats = {
                "ink_threshold": args.ink_threshold,
                "ink_dilate": args.ink_dilate,
                "ink_pixels": int(ink_mask.sum()),
                "masked_probability_pixels": restrict_probs_to_ink(probs, ink_mask),
            }
        component_stats = remove_small_layer_components(probs, args.threshold, args.min_component_area, np)
        glyph_out = out_dir / glyph_id
        save_prediction_layers(glyph_out, probs, threshold=args.threshold)
        overlay_dir = glyph_out / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        binary = probs >= args.threshold
        make_layer_overlay(original, binary).save(overlay_dir / "composite.png")
        metadata = {
            "glyph": rel.as_posix(),
            "layers": list(LAYERS),
            "threshold": args.threshold,
            "activation": "sigmoid",
        }
        if args.suppress_no_final_t:
            metadata["postprocess"] = {
                "suppress_no_final_t": True,
                "mapping_csv": str(args.mapping_csv),
                "TIndex": int(mapping_row["TIndex"]),
                "t_suppressed": t_suppressed,
            }
        if args.restrict_to_ink or args.min_component_area > 1:
            metadata.setdefault("postprocess", {})
            if ink_stats is not None:
                metadata["postprocess"]["restrict_to_ink"] = True
                metadata["postprocess"]["ink"] = ink_stats
            if args.min_component_area > 1:
                metadata["postprocess"]["min_component_area"] = args.min_component_area
                metadata["postprocess"]["components"] = component_stats
        (glyph_out / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {glyph_out}")


if __name__ == "__main__":
    main()
