# handwriting-hangul-layer-font

Layer-mask Hangul glyph decomposition pipeline for handwriting-to-font generation.

This repository is a reset of the older RGB semantic-mask pipeline. The old project stored one PNG where each pixel had exactly one class: background, choseong, jungseong, or jongseong. That is useful for a first baseline, but it breaks down when handwritten jamo touch or overlap. This project treats each Hangul syllable as three independently editable binary layers:

- `L.png`: choseong layer
- `V.png`: jungseong layer
- `T.png`: jongseong layer, empty when the syllable has no final consonant

A pixel may belong to more than one layer at the same time.

## Current Stage

The initial layer pipeline is in place:

- writing sheet crop/normalize into `datasets/glyphs_214`
- Clova 214-character font rendering and one-font preparation
- browser layer editor that saves independent `L/V/T` masks
- multi-label U-Net train/predict scripts using sigmoid outputs

The old RGB mask editor, RGB mask IO, and 4-class argmax segmentation code were intentionally not copied as the data source of truth.

## Data Layout

```text
configs/
  chars_214.txt
  mapping_214.csv
datasets/
  raw_sheets/{user_id}.png
  cropped_214/{user_id}/...
  glyphs_214/{user_or_font}/{glyph_id}.png
  masks_layer_214/{user_or_font}/{glyph_id}/
    L.png
    V.png
    T.png
    ignore.png        # optional
    metadata.json     # optional
  clova_214/
runs/
tools/
  writing_sheet_214/
  layer_mask_editor/
hangul_layer_seg/
```

`glyph_id` should stay ASCII-safe, for example `001_UAC00`.

## Model Direction

The segmentation model should be multi-label, not multi-class:

- input: grayscale glyph tensor `[B, 1, H, W]`
- output: logits `[B, 3, H, W]` for `L`, `V`, `T`
- activation: sigmoid per channel
- loss: `BCEWithLogitsLoss`, optionally plus per-channel Dice/Focal loss
- ignore handling: optional `ignore.png` masks pixels excluded from all channel losses

Prediction should write separate binary/probability layer files and optional color overlays for inspection. It should not write an argmax class mask as the source of truth.

## Basic Commands

Install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Crop and normalize one writing sheet:

```powershell
python tools/writing_sheet_214/run_pipeline.py `
  --input datasets/raw_sheets/user_001.png `
  --out_dir datasets `
  --user_id user_001 `
  --mode fixed `
  --grid_bbox 64,48,731,1076 `
  --image_size 512
```

Open the layer editor:

```powershell
python tools/layer_mask_editor/server.py `
  --glyph_dir datasets/glyphs_214/user_001 `
  --mask_dir datasets/masks_layer_214/user_001 `
  --port 7860
```

Train the multi-label model:

```powershell
python hangul_layer_seg/train.py `
  --glyph_dir datasets/glyphs_214 `
  --mask_dir datasets/masks_layer_214 `
  --save_dir runs/layer_unet_baseline `
  --image_size 128 `
  --cache_in_memory `
  --val_interval 5 `
  --sample_interval 5
```

On CUDA, add these options for faster training:

```powershell
  --device cuda `
  --amp `
  --channels_last
```

Predict layer masks:

```powershell
python hangul_layer_seg/predict.py `
  --ckpt runs/layer_unet_baseline/checkpoint_best.pt `
  --input_dir datasets/glyphs_214/user_001 `
  --out_dir runs/layer_unet_baseline/predictions/user_001
```

Prepare the next Clova font for editing:

```powershell
python tools/writing_sheet_214/clova_font_pipeline.py prepare-next
```

Prepare every Clova/Nanum font into 214 glyphs:

```powershell
python tools/writing_sheet_214/clova_font_pipeline.py prepare-all `
  --dataset_root datasets/clova_214
```

Open the multi-font layer editor:

```powershell
python tools/layer_mask_editor/server.py `
  --dataset_root datasets/clova_214 `
  --port 7860
```

The multi-font editor lets you choose a prepared font, label its 214 glyphs,
mark usable fonts complete, skip weak fonts, or move bad fonts to `_deleted`.

## Design Docs

- [Legacy Analysis](docs/legacy-analysis.md)
- [Layer Pipeline Design](docs/layer-pipeline.md)
