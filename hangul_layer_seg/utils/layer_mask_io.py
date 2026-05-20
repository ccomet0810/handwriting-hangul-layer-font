from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


LAYERS = ("L", "V", "T")
LAYER_COLORS = {
    "L": np.array([255, 60, 60], dtype=np.float32),
    "V": np.array([40, 210, 90], dtype=np.float32),
    "T": np.array([80, 130, 255], dtype=np.float32),
    "ignore": np.array([160, 160, 160], dtype=np.float32),
}


def _resize_gray(path: Path, size: int | None, default_shape: tuple[int, int] | None = None) -> np.ndarray:
    if path.exists():
        image = Image.open(path).convert("L")
        if size is not None:
            image = image.resize((size, size), Image.NEAREST)
        return np.array(image, dtype=np.uint8)
    if default_shape is None:
        if size is None:
            raise ValueError(f"missing mask and no size/default shape supplied: {path}")
        default_shape = (size, size)
    return np.zeros(default_shape, dtype=np.uint8)


def load_layer_mask_dir(mask_dir: str | Path, image_size: int | None = None) -> dict[str, np.ndarray]:
    root = Path(mask_dir)
    first_shape: tuple[int, int] | None = None
    layers: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        arr = _resize_gray(root / f"{layer}.png", image_size, first_shape)
        first_shape = arr.shape
        layers[layer] = (arr >= 128).astype(np.uint8)
    ignore_path = root / "ignore.png"
    if ignore_path.exists():
        layers["ignore"] = (_resize_gray(ignore_path, image_size, first_shape) >= 128).astype(np.uint8)
    else:
        layers["ignore"] = np.zeros(first_shape or (image_size or 0, image_size or 0), dtype=np.uint8)
    return layers


def save_layer_mask_dir(
    mask_dir: str | Path,
    layers: dict[str, np.ndarray],
    metadata: dict | None = None,
    save_ignore: bool = False,
) -> None:
    root = Path(mask_dir)
    root.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        if layer not in layers:
            raise ValueError(f"missing layer {layer}")
        arr = (np.asarray(layers[layer]) > 0).astype(np.uint8) * 255
        Image.fromarray(arr, mode="L").save(root / f"{layer}.png")
    if save_ignore or "ignore" in layers:
        arr = (np.asarray(layers.get("ignore", 0)) > 0).astype(np.uint8) * 255
        Image.fromarray(arr, mode="L").save(root / "ignore.png")
    if metadata is not None:
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_target_arrays(mask_dir: str | Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    loaded = load_layer_mask_dir(mask_dir, image_size)
    target = np.stack([loaded[layer].astype(np.float32) for layer in LAYERS], axis=0)
    valid = (1.0 - loaded["ignore"].astype(np.float32))[None, ...]
    return target, valid


def save_prediction_layers(
    out_dir: str | Path,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> None:
    root = Path(out_dir)
    prob_dir = root / "probs"
    binary_dir = root / "binary"
    prob_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    for idx, layer in enumerate(LAYERS):
        prob = np.clip(probs[idx] * 255.0, 0, 255).astype(np.uint8)
        binary = (probs[idx] >= threshold).astype(np.uint8) * 255
        Image.fromarray(prob, mode="L").save(prob_dir / f"{layer}.png")
        Image.fromarray(binary, mode="L").save(binary_dir / f"{layer}.png")


def make_layer_overlay(
    glyph: Image.Image | np.ndarray,
    layers: dict[str, np.ndarray] | np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    if isinstance(glyph, Image.Image):
        base = np.array(glyph.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(glyph)
        base = np.stack([arr, arr, arr], axis=-1).astype(np.float32) if arr.ndim == 2 else arr[..., :3].astype(np.float32)

    if isinstance(layers, np.ndarray):
        layer_map = {layer: layers[idx] for idx, layer in enumerate(LAYERS)}
    else:
        layer_map = layers

    h, w = base.shape[:2]
    color_sum = np.zeros_like(base, dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for layer in LAYERS:
        mask = np.asarray(layer_map[layer])
        if mask.shape != (h, w):
            mask = np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        else:
            mask = mask > 0
        color_sum[mask] += LAYER_COLORS[layer]
        count[mask] += 1.0
    active = count > 0
    color = base.copy()
    color[active] = color_sum[active] / count[active][:, None]
    out = base.copy()
    out[active] = base[active] * (1.0 - alpha) + color[active] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
