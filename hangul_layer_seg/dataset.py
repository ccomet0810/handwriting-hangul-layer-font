from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

try:
    from .utils.layer_mask_io import load_target_arrays
except ImportError:  # pragma: no cover
    from utils.layer_mask_io import load_target_arrays


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LayerSegSample:
    glyph_path: Path
    mask_dir: Path
    rel_path: Path


class HangulLayerDataset(Dataset):
    def __init__(
        self,
        glyph_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 128,
        samples: Sequence[LayerSegSample] | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.glyph_dir = Path(glyph_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.samples = list(samples) if samples is not None else find_samples(self.glyph_dir, self.mask_dir)
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, dict] = {}
        if cache_in_memory:
            for idx in range(len(self.samples)):
                self._cache[idx] = self._load_item(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.cache_in_memory:
            return self._cache[idx]
        return self._load_item(idx)

    def _load_item(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample.glyph_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_arr = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr).unsqueeze(0)

        target, valid = load_target_arrays(sample.mask_dir, self.image_size)
        return {
            "image": image_tensor,
            "target": torch.from_numpy(target),
            "valid": torch.from_numpy(valid),
            "glyph_path": str(sample.glyph_path),
            "mask_dir": str(sample.mask_dir),
            "rel_path": sample.rel_path.as_posix(),
        }


def glyph_id_from_path(path: Path) -> str:
    return path.stem


def find_samples(glyph_dir: str | Path, mask_dir: str | Path) -> list[LayerSegSample]:
    glyph_root = Path(glyph_dir)
    mask_root = Path(mask_dir)
    samples: list[LayerSegSample] = []
    if not glyph_root.exists() or not mask_root.exists():
        return samples
    for glyph_path in sorted(glyph_root.rglob("*")):
        if not glyph_path.is_file() or glyph_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = glyph_path.relative_to(glyph_root)
        mask_path = mask_root / rel.with_suffix("")
        if all((mask_path / f"{layer}.png").exists() for layer in ("L", "V", "T")):
            samples.append(LayerSegSample(glyph_path=glyph_path, mask_dir=mask_path, rel_path=rel.with_suffix("")))
    return samples


def train_val_split(
    dataset: HangulLayerDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_n = int(round(n * val_ratio))
    if n > 1:
        val_n = max(1, min(val_n, n - 1))
    return Subset(dataset, indices[val_n:]), Subset(dataset, indices[:val_n])
