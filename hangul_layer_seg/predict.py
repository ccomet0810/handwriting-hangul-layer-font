from __future__ import annotations

import argparse
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
    paths = list_images(input_path, IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No input images found under {input_path}")

    for path in paths:
        rel = path.relative_to(input_path) if input_path.is_dir() else Path(path.name)
        glyph_id = rel.with_suffix("")
        probs, original = predict_one(model, path, image_size, device, np, torch, Image)
        glyph_out = out_dir / glyph_id
        save_prediction_layers(glyph_out, probs, threshold=args.threshold)
        overlay_dir = glyph_out / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        binary = probs >= args.threshold
        make_layer_overlay(original, binary).save(overlay_dir / "composite.png")
        (glyph_out / "metadata.json").write_text(
            json.dumps(
                {
                    "glyph": rel.as_posix(),
                    "layers": list(LAYERS),
                    "threshold": args.threshold,
                    "activation": "sigmoid",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {glyph_out}")


if __name__ == "__main__":
    main()
