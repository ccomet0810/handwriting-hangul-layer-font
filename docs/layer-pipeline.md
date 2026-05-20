# Layer Pipeline Design

## Goals

- Accept a 214-character handwriting sheet or rendered font glyph set.
- Normalize each syllable glyph into `datasets/glyphs_214/{source_id}/{glyph_id}.png`.
- Decompose each glyph into independent `L`, `V`, and `T` binary masks.
- Allow overlapping pixels across layers.
- Train a multi-label segmentation model that predicts three sigmoid channels.
- Support a browser editor that can toggle, paint, fill, and save individual layers.

## Canonical Mask Format

Each glyph has a directory:

```text
datasets/masks_layer_214/{source_id}/{glyph_id}/
  L.png
  V.png
  T.png
  ignore.png
  metadata.json
```

Rules:

- `L.png`, `V.png`, and `T.png` are grayscale PNGs.
- Pixel value `0` means not in this layer.
- Pixel value `255` means included in this layer.
- `T.png` exists for every glyph for shape consistency; it may be all black when `TIndex == 0`.
- `ignore.png` is optional. `255` means ignore this pixel during loss computation.
- `metadata.json` is optional but should include source, glyph id, Unicode, character, decomposition, label state, and provenance.

Example metadata:

```json
{
  "source_id": "user_001",
  "glyph_id": "001_UAC00",
  "char": "가",
  "unicode": "U+AC00",
  "LIndex": 0,
  "VIndex": 0,
  "TIndex": 0,
  "layers": {
    "L": {"path": "L.png", "empty": false},
    "V": {"path": "V.png", "empty": false},
    "T": {"path": "T.png", "empty": true}
  },
  "status": "draft"
}
```

## Proposed Modules

```text
tools/writing_sheet_214/
  run_pipeline.py          # crop + normalize sheet into glyphs_214
  detect_sheet.py
  crop_cells.py
  normalize_glyphs.py
  clova_font_pipeline.py   # render fonts and prepare layer-label rounds
  clova_auto_loop.py

tools/layer_mask_editor/
  server.py                # serves glyphs and per-layer masks
  static/
    index.html
    app.js
    style.css

hangul_layer_seg/
  dataset.py               # returns image [1,H,W], target [3,H,W], valid mask
  losses.py                # BCEWithLogits + optional multilabel Dice/Focal
  train.py                 # trains sigmoid 3-channel model
  predict.py               # writes L/V/T probability and binary masks
  visualize.py             # overlay combinations without becoming source data
  models/
    unet.py
  utils/
    layer_mask_io.py
    hangul.py
```

## Layer Mask IO

`layer_mask_io.py` should provide:

- `load_layer_mask_dir(path) -> dict`
- `save_layer_mask_dir(path, layers, metadata=None)`
- `load_target_tensor(mask_dir, size) -> target[3,H,W], valid[1,H,W]`
- `save_prediction_layers(out_dir, probs, threshold=0.5)`
- `make_layer_overlay(glyph, layers)`

It should never collapse `L/V/T` into an argmax class map for storage.

## Editor Behavior

The editor should be layer-first:

- layer toggles: `L`, `V`, `T`, `ignore`
- one active paint target at a time
- visibility can show any combination of layers
- painting a pixel in one layer does not erase it from another layer
- optional erase mode affects only the active layer
- fill mode operates only on the active layer
- save writes `L.png`, `V.png`, `T.png`, optional `ignore.png`, and status metadata

Suggested shortcuts:

```text
1     active layer L
2     active layer V
3     active layer T
4     active layer ignore
Q     brush
W     fill
E     erase active layer
S     save
Enter save and next
[ ]   brush size
Ctrl+Z / Ctrl+Y undo / redo
```

## Training Contract

Dataset sample:

```python
{
    "image": FloatTensor[1, H, W],
    "target": FloatTensor[3, H, W],
    "valid": FloatTensor[1, H, W],
    "glyph_path": str,
    "mask_dir": str,
    "rel_id": str,
}
```

Loss:

```text
loss = BCEWithLogits(logits, target, reduction=none)
loss = (loss * valid).sum() / valid.sum().clamp_min(1)
loss += dice_weight * multilabel_dice_loss(sigmoid(logits), target, valid)
```

For `TIndex == 0`, `T.png` should be all zero and still participate in training unless metadata marks it ignored.

## Prediction Output

```text
runs/{run_id}/predictions/{source_id}/{glyph_id}/
  probs/
    L.png
    V.png
    T.png
  binary/
    L.png
    V.png
    T.png
  overlays/
    composite.png
    L.png
    V.png
    T.png
  metadata.json
```

Draft predictions copied into editor workspaces should use the canonical `L/V/T` layout directly.

## First Implementation Order

1. Port 214 mapping helpers and Hangul decomposition utilities.
2. Port writing-sheet crop/normalize into `tools/writing_sheet_214`.
3. Add `layer_mask_io.py` and tests for overlapping pixels.
4. Build a minimal layer editor server API and save/load format.
5. Port U-Net backbone with a 3-channel sigmoid prediction path.
6. Add train/predict scripts for multi-label masks.
7. Adapt Clova render and one-font loop to prepare layer drafts.

