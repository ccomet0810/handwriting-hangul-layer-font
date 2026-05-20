from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from crop_cells import code_name
from normalize_glyphs import normalize_cell
from run_pipeline import CHARS_PATH, load_chars


FONT_EXTS = {".ttf", ".otf", ".ttc"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLOVA_ROOT = ROOT.parent / "handwriting-to-ai-font" / "clova-all"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "clova_214"
DEFAULT_STATE_PATH = DEFAULT_DATASET_ROOT / "_state.json"


def safe_id(text: str, index: int) -> str:
    ascii_part = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "font"
    return f"{index:03d}_{ascii_part}"


def find_fonts(clova_root: str | Path) -> list[Path]:
    root = Path(clova_root)
    fonts = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FONT_EXTS)
    if not fonts:
        raise ValueError(f"no font files found under {root}")
    return fonts


def build_manifest(clova_root: str | Path, dataset_root: str | Path) -> list[dict[str, str]]:
    rows = []
    for idx, font_path in enumerate(find_fonts(clova_root), start=1):
        display_name = font_path.parent.name or font_path.stem
        rows.append(
            {
                "order": str(idx),
                "font_id": safe_id(display_name, idx),
                "display_name": display_name,
                "font_path": str(font_path),
            }
        )
    manifest_path = Path(dataset_root) / "font_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order", "font_id", "display_name", "font_path"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_manifest(clova_root: str | Path, dataset_root: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(dataset_root) / "font_manifest.csv"
    if not manifest_path.exists():
        return build_manifest(clova_root, dataset_root)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_state(path: str | Path = DEFAULT_STATE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {"prepared": [], "completed": [], "skipped": [], "current": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    for key in ("prepared", "completed", "skipped"):
        state.setdefault(key, [])
    state.setdefault("current", None)
    return state


def save_state(state: dict, path: str | Path = DEFAULT_STATE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_char_to_image(char: str, font: ImageFont.FreeTypeFont, canvas_size: int) -> Image.Image:
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw.textbbox((0, 0), char, font=font)
    x = (canvas_size - (x1 - x0)) / 2 - x0
    y = (canvas_size - (y1 - y0)) / 2 - y0
    draw.text((x, y), char, fill=0, font=font)
    return image


def render_font_214(
    font_path: str | Path,
    glyph_dir: str | Path,
    chars: list[str],
    image_size: int = 512,
    render_size: int = 420,
    oversample: int = 2,
) -> list[dict[str, str]]:
    glyph_dir = Path(glyph_dir)
    glyph_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(font_path), size=render_size * oversample)
    rows = []
    for index, char in enumerate(chars, start=1):
        rendered = render_char_to_image(char, font, image_size * oversample)
        normalized, metrics = normalize_cell(
            image=__import__("numpy").array(rendered),
            image_size=image_size,
            padding_ratio=0.18,
            min_ink_area=4,
        )
        glyph_id = f"{index:03d}_{code_name(char)}"
        out_path = glyph_dir / f"{glyph_id}.png"
        Image.fromarray(normalized).save(out_path)
        rows.append({"index": str(index), "char": char, "unicode": f"U+{ord(char):04X}", "glyph_id": glyph_id, "glyph_path": str(out_path), "warning": str(metrics.get("warning", ""))})
    return rows


def write_render_metadata(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "char", "unicode", "glyph_id", "glyph_path", "warning"])
        writer.writeheader()
        writer.writerows(rows)


def run_predict(ckpt: str | Path, input_dir: str | Path, out_dir: str | Path, device: str = "auto") -> None:
    cmd = [
        sys.executable,
        str(ROOT / "hangul_layer_seg" / "predict.py"),
        "--ckpt",
        str(ckpt),
        "--input_dir",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def copy_binary_predictions_to_draft(pred_dir: Path, draft_dir: Path) -> int:
    copied = 0
    for binary_dir in sorted((pred_dir).rglob("binary")):
        glyph_rel = binary_dir.parent.relative_to(pred_dir)
        dst = draft_dir / glyph_rel
        dst.mkdir(parents=True, exist_ok=True)
        for layer in ("L", "V", "T"):
            src = binary_dir / f"{layer}.png"
            if src.exists():
                shutil.copy2(src, dst / f"{layer}.png")
                copied += 1
    return copied


def prepare_font(
    row: dict[str, str],
    dataset_root: str | Path,
    chars: list[str],
    ckpt: str | Path | None,
    device: str,
    image_size: int,
) -> dict[str, str]:
    dataset_root = Path(dataset_root)
    font_id = row["font_id"]
    glyph_dir = dataset_root / "glyphs" / font_id
    mask_dir = dataset_root / "masks_layer" / font_id
    draft_dir = dataset_root / "draft_masks_layer" / font_id
    pred_dir = dataset_root / "predictions" / font_id
    meta_dir = dataset_root / "metadata" / font_id

    render_rows = render_font_214(row["font_path"], glyph_dir, chars, image_size=image_size)
    write_render_metadata(render_rows, meta_dir / "render_metadata.csv")
    mask_dir.mkdir(parents=True, exist_ok=True)

    if ckpt and Path(ckpt).exists():
        run_predict(ckpt, glyph_dir, pred_dir, device=device)
        copy_binary_predictions_to_draft(pred_dir, draft_dir)

    info = {
        "font_id": font_id,
        "display_name": row["display_name"],
        "font_path": row["font_path"],
        "glyph_dir": str(glyph_dir),
        "mask_dir": str(mask_dir),
        "draft_dir": str(draft_dir),
        "pred_dir": str(pred_dir),
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "font_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def next_unprepared(manifest: list[dict[str, str]], state: dict) -> dict[str, str] | None:
    done = set(state.get("prepared", [])) | set(state.get("skipped", [])) | set(state.get("completed", []))
    return next((row for row in manifest if row["font_id"] not in done), None)


def sync_training_root(
    dataset_root: str | Path,
    out_root: str | Path,
    include_handwritten: bool = True,
    completed_only: bool = False,
) -> tuple[int, int]:
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    glyph_out = out_root / "glyphs"
    mask_out = out_root / "masks_layer"
    glyph_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    state = load_state(dataset_root / "_state.json")
    skipped = set(state.get("skipped", []))
    completed = set(state.get("completed", []))

    def copy_tree(
        src_root: Path,
        dst_root: Path,
        exclude_top_names: set[str] | None = None,
        include_top_names: set[str] | None = None,
    ) -> int:
        count = 0
        if not src_root.exists():
            return count
        for src in sorted(src_root.rglob("*")):
            if not src.is_file() or src.name.startswith("_"):
                continue
            rel = src.relative_to(src_root)
            if exclude_top_names and rel.parts and rel.parts[0] in exclude_top_names:
                continue
            if include_top_names is not None and (not rel.parts or rel.parts[0] not in include_top_names):
                continue
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
        return count

    glyph_count = copy_tree(ROOT / "datasets" / "glyphs_214", glyph_out / "handwritten") if include_handwritten else 0
    mask_count = copy_tree(ROOT / "datasets" / "masks_layer_214", mask_out / "handwritten") if include_handwritten else 0
    if completed_only:
        allowed = completed
        glyph_count += copy_tree(dataset_root / "glyphs", glyph_out / "clova", exclude_top_names=None, include_top_names=allowed)
        mask_count += copy_tree(dataset_root / "masks_layer", mask_out / "clova", exclude_top_names=None, include_top_names=allowed)
    else:
        glyph_count += copy_tree(dataset_root / "glyphs", glyph_out / "clova", skipped)
        mask_count += copy_tree(dataset_root / "masks_layer", mask_out / "clova", skipped)
    return glyph_count, mask_count


def print_editor_command(info: dict[str, str], port: int) -> None:
    print("editor command:")
    print(
        "python tools/layer_mask_editor/server.py `\n"
        f"  --glyph_dir {info['glyph_dir']} `\n"
        f"  --mask_dir {info['mask_dir']} `\n"
        f"  --draft_dir {info['draft_dir']} `\n"
        f"  --port {port}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-font-at-a-time Clova 214 layer-mask workflow.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--clova_root", default=str(DEFAULT_CLOVA_ROOT))
    common.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    common.add_argument("--chars", default=str(CHARS_PATH))
    sub.add_parser("list", parents=[common])

    prepare_next = sub.add_parser("prepare-next", parents=[common])
    prepare_next.add_argument("--ckpt", default="")
    prepare_next.add_argument("--device", default="auto")
    prepare_next.add_argument("--image_size", type=int, default=512)
    prepare_next.add_argument("--port", type=int, default=7860)

    prepare_all = sub.add_parser("prepare-all", parents=[common])
    prepare_all.add_argument("--ckpt", default="")
    prepare_all.add_argument("--device", default="auto")
    prepare_all.add_argument("--image_size", type=int, default=512)
    prepare_all.add_argument("--force", action="store_true", help="Re-render fonts that already have 214 glyphs.")

    complete = sub.add_parser("mark-complete", parents=[common])
    complete.add_argument("--font_id", required=True)
    skip = sub.add_parser("mark-skip", parents=[common])
    skip.add_argument("--font_id", required=True)
    sync = sub.add_parser("sync-training-root", parents=[common])
    sync.add_argument("--out_root", default=str(ROOT / "datasets" / "train_214"))
    sync.add_argument("--no_handwritten", action="store_true")
    sync.add_argument("--completed_only", action="store_true", help="Include only fonts marked Complete in _state.json.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chars = load_chars(args.chars)
    manifest = load_manifest(args.clova_root, args.dataset_root)
    if args.cmd == "list":
        for row in manifest:
            print(f"{row['order']}. {row['font_id']} | {row['display_name']} | {row['font_path']}")
        return 0
    if args.cmd == "prepare-next":
        state = load_state(Path(args.dataset_root) / "_state.json")
        row = next_unprepared(manifest, state)
        if row is None:
            print("all fonts have been prepared")
            return 0
        info = prepare_font(row, args.dataset_root, chars, args.ckpt or None, args.device, args.image_size)
        state.setdefault("prepared", []).append(info["font_id"])
        state["current"] = info["font_id"]
        save_state(state, Path(args.dataset_root) / "_state.json")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print_editor_command(info, args.port)
        return 0
    if args.cmd == "prepare-all":
        state = load_state(Path(args.dataset_root) / "_state.json")
        prepared = set(state.get("prepared", []))
        completed = set(state.get("completed", []))
        skipped = set(state.get("skipped", []))
        made = 0
        reused = 0
        failed = 0
        for row in manifest:
            font_id = row["font_id"]
            if font_id in skipped:
                continue
            glyph_dir = Path(args.dataset_root) / "glyphs" / font_id
            already_ready = len(list(glyph_dir.glob("*.png"))) >= len(chars)
            if already_ready and not args.force:
                reused += 1
                if font_id not in prepared and font_id not in completed:
                    state.setdefault("prepared", []).append(font_id)
                continue
            try:
                prepare_font(row, args.dataset_root, chars, args.ckpt or None, args.device, args.image_size)
            except Exception as exc:
                failed += 1
                error_dir = Path(args.dataset_root) / "metadata" / font_id
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / "prepare_error.json").write_text(
                    json.dumps(
                        {
                            "font_id": font_id,
                            "display_name": row.get("display_name", font_id),
                            "font_path": row.get("font_path", ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if font_id not in state.setdefault("skipped", []):
                    state["skipped"].append(font_id)
                print(f"skipped {font_id} | {row['display_name']} | {type(exc).__name__}: {exc}")
                continue
            made += 1
            if font_id not in state.setdefault("prepared", []) and font_id not in completed:
                state["prepared"].append(font_id)
            print(f"prepared {font_id} | {row['display_name']}")
        save_state(state, Path(args.dataset_root) / "_state.json")
        print(f"prepared newly: {made}")
        print(f"already ready: {reused}")
        print(f"failed/skipped: {failed}")
        print(f"dataset_root: {args.dataset_root}")
        return 0
    if args.cmd in {"mark-complete", "mark-skip"}:
        state = load_state(Path(args.dataset_root) / "_state.json")
        key = "completed" if args.cmd == "mark-complete" else "skipped"
        other = "skipped" if key == "completed" else "completed"
        if args.font_id not in state.setdefault(key, []):
            state[key].append(args.font_id)
        if args.font_id in state.setdefault(other, []):
            state[other].remove(args.font_id)
        if state.get("current") == args.font_id:
            state["current"] = None
        save_state(state, Path(args.dataset_root) / "_state.json")
        print(f"marked {key}: {args.font_id}")
        return 0
    if args.cmd == "sync-training-root":
        glyph_count, mask_count = sync_training_root(
            args.dataset_root,
            args.out_root,
            include_handwritten=not args.no_handwritten,
            completed_only=args.completed_only,
        )
        print(f"synced glyph files: {glyph_count}")
        print(f"synced mask files: {mask_count}")
        print(f"train glyph_dir: {Path(args.out_root) / 'glyphs'}")
        print(f"train mask_dir: {Path(args.out_root) / 'masks_layer'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
