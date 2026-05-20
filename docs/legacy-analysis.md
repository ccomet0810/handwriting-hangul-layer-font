# Legacy Repository Analysis

Source repository analyzed:

```text
C:\Users\ccomet0810\Documents\GitHub\handwriting-to-ai-font
```

Target repository:

```text
C:\Users\ccomet0810\Documents\GitHub\handwriting-hangul-layer-font
```

## Reuse

The old repo has several parts that are still valuable because they operate before jamo ownership is introduced:

- `tools/writing_sheet_214/chars_214.txt` and `mapping_214.csv`: keep the 214-character order, grid position, Unicode decomposition, `LIndex`, `VIndex`, `TIndex`, and vowel grouping.
- `detect_sheet.py`, `crop_cells.py`, `normalize_glyphs.py`, `visualize_debug.py`, `run_pipeline.py`: reuse the writing-sheet input path with light renaming from `masks_214` to `masks_layer_214`.
- `clova_font_pipeline.py`: reuse the one-font-at-a-time manifest, render, prepare, skip, complete, and sync workflow.
- `clova_auto_loop.py`: reuse the orchestration idea after the layer editor and multi-label predictor exist.
- Small U-Net backbone shape from `ai_jamo_seg/models/unet.py`: reusable as an encoder-decoder baseline if its head outputs 3 independent logits instead of 4 mutually exclusive classes.

## Do Not Reuse As-Is

These modules encode the old single-owner semantic segmentation assumption:

- `ai_jamo_seg/utils/mask_io.py`: converts RGB/index PNG to one class index per pixel.
- `ai_jamo_seg/dataset.py`: loads one mask file per glyph and returns an integer target `[H, W]`.
- `ai_jamo_seg/losses.py`: uses cross entropy and softmax Dice.
- `ai_jamo_seg/train.py`: saves checkpoints with `num_classes=4` and visualizes argmax predictions.
- `ai_jamo_seg/predict.py`: computes `logits.argmax(dim=1)` and writes `index_masks`/`rgb_masks`.
- `tools/jamo_mask_editor/*`: paints one RGB mask canvas where each brush action replaces the pixel class.
- `create_draft_masks.py`: produces single RGB draft masks rather than independent `L/V/T` drafts.

## Main Architectural Change

Old contract:

```text
glyph.png -> model -> [background, L, V, T] softmax -> argmax class per pixel -> one RGB PNG
```

New contract:

```text
glyph.png -> model -> [L, V, T] sigmoid -> three independent layer masks
```

This lets a pixel be simultaneously part of `L` and `V`, or any other combination, without destroying either jamo shape.

## Practical Migration Notes

- Keep filenames and relative paths compatible with the old 214 pipeline wherever possible.
- Store layer masks in a directory per glyph instead of one PNG per glyph.
- Use previews/overlays for human review, but keep separate binary layer PNGs as the canonical labels.
- The editor should maintain three editable alpha/binary canvases, not one RGB palette canvas.
- Existing completed RGB masks can be imported only as non-overlap drafts, not as authoritative labels.

