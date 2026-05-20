from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import shutil
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LAYERS = {"L", "V", "T", "ignore"}
SKIP_DIRS = {"debug", "_debug", "_metadata", "_overlay_previews", "overlay_previews"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local L/V/T layer mask annotation tool.")
    parser.add_argument("--glyph_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--draft_dir", default=None)
    parser.add_argument("--dataset_root", default=None, help="Optional Clova dataset root for multi-font selection.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def safe_rel_path(raw: str) -> Path:
    rel = Path(unquote(raw).replace("\\", "/"))
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("invalid relative path")
    return rel


def layer_from_query(query: str) -> str:
    layer = parse_qs(query).get("layer", [""])[0]
    if layer not in LAYERS:
        raise ValueError("invalid layer")
    return layer


def code_to_char(stem: str) -> str:
    code = stem.split("_")[-1]
    if not code.startswith("U"):
        return ""
    try:
        return chr(int(code[1:], 16))
    except ValueError:
        return ""


def list_glyphs(glyph_dir: Path, mask_dir: Path) -> list[dict]:
    items = []
    for path in sorted(glyph_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(glyph_dir)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        mask_base = mask_dir / rel.with_suffix("")
        saved = all((mask_base / f"{layer}.png").exists() for layer in ("L", "V", "T"))
        items.append({"file": rel.as_posix(), "glyphId": rel.with_suffix("").as_posix(), "code": path.stem, "char": code_to_char(path.stem), "saved": saved})
    return items


def load_manifest(dataset_root: Path) -> list[dict]:
    path = dataset_root / "font_manifest.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_state(dataset_root: Path) -> dict:
    path = dataset_root / "_state.json"
    if not path.exists():
        return {"prepared": [], "completed": [], "skipped": [], "deleted": [], "current": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        state = {}
    for key in ("prepared", "completed", "skipped", "deleted"):
        state.setdefault(key, [])
    state.setdefault("current", None)
    return state


def save_state(dataset_root: Path, state: dict) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def completed_mask_count(mask_dir: Path) -> int:
    if not mask_dir.exists():
        return 0
    count = 0
    for path in mask_dir.iterdir():
        if path.is_dir() and all((path / f"{layer}.png").exists() for layer in ("L", "V", "T")):
            count += 1
    return count


def font_summary(dataset_root: Path) -> list[dict]:
    state = load_state(dataset_root)
    rows = load_manifest(dataset_root)
    out = []
    for row in rows:
        font_id = row["font_id"]
        glyph_dir = dataset_root / "glyphs" / font_id
        mask_dir = dataset_root / "masks_layer" / font_id
        glyph_count = len(list(glyph_dir.glob("*.png"))) if glyph_dir.exists() else 0
        saved_count = completed_mask_count(mask_dir)
        status = "pending"
        if font_id in state.get("deleted", []):
            status = "deleted"
        elif font_id in state.get("skipped", []):
            status = "skipped"
        elif font_id in state.get("completed", []):
            status = "completed"
        elif glyph_count:
            status = "prepared"
        out.append(
            {
                "fontId": font_id,
                "displayName": row.get("display_name", font_id),
                "fontPath": row.get("font_path", ""),
                "glyphCount": glyph_count,
                "savedCount": saved_count,
                "status": status,
            }
        )
    return out


def safe_font_id(value: str) -> str:
    if not value or any(ch in value for ch in "\\/.:"):
        raise ValueError("invalid font_id")
    return value


def decode_png_data_url(data_url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("layer must be a PNG data URL")
    payload = base64.b64decode(data_url[len(prefix) :], validate=True)
    return Image.open(BytesIO(payload)).convert("L")


def quantize_binary(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    return gray.point(lambda px: 255 if px >= 128 else 0, mode="L")


class LayerMaskEditorHandler(BaseHTTPRequestHandler):
    server: "LayerMaskEditorServer"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[layer_mask_editor] " + fmt % args + "\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_static("index.html")
            elif parsed.path.startswith("/static/"):
                self.send_static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/list":
                glyph_dir, mask_dir, draft_dir = self.resolve_roots(parsed.query)
                self.send_json({"glyphDir": str(glyph_dir), "maskDir": str(mask_dir), "hasDraftDir": draft_dir is not None, "files": list_glyphs(glyph_dir, mask_dir)})
            elif parsed.path == "/api/fonts":
                if self.server.dataset_root is None:
                    self.send_json({"managed": False, "fonts": []})
                else:
                    self.send_json({"managed": True, "fonts": font_summary(self.server.dataset_root)})
            elif parsed.path == "/api/glyph":
                glyph_dir, _mask_dir, _draft_dir = self.resolve_roots(parsed.query)
                self.send_image_from_query(parsed.query, glyph_dir, required=True)
            elif parsed.path == "/api/layer":
                _glyph_dir, mask_dir, _draft_dir = self.resolve_roots(parsed.query)
                self.send_layer_from_query(parsed.query, mask_dir, required=False)
            elif parsed.path == "/api/draft-layer":
                _glyph_dir, _mask_dir, draft_dir = self.resolve_roots(parsed.query)
                if draft_dir is None:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_layer_from_query(parsed.query, draft_dir, required=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/save":
                self.handle_save()
            elif parsed.path == "/api/font-action":
                self.handle_font_action()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected JSON object")
        return payload

    def handle_save(self) -> None:
        payload = self.read_json()
        rel = safe_rel_path(str(payload.get("file", "")))
        _glyph_dir, mask_dir, _draft_dir = self.resolve_roots("", payload)
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            raise ValueError("layers must be an object")
        out_dir = mask_dir / rel.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for layer in ("L", "V", "T", "ignore"):
            data_url = layers.get(layer)
            if not data_url:
                continue
            quantize_binary(decode_png_data_url(str(data_url))).save(out_dir / f"{layer}.png", format="PNG")
            saved.append(layer)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update({"glyph": rel.as_posix(), "saved_layers": saved})
        (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.send_json({"ok": True, "path": str(out_dir), "savedLayers": saved})

    def handle_font_action(self) -> None:
        if self.server.dataset_root is None:
            raise ValueError("font actions require --dataset_root")
        payload = self.read_json()
        font_id = safe_font_id(str(payload.get("fontId", "")))
        action = str(payload.get("action", ""))
        state = load_state(self.server.dataset_root)
        for key in ("prepared", "completed", "skipped", "deleted"):
            state.setdefault(key, [])
        if action == "skip":
            if font_id not in state["skipped"]:
                state["skipped"].append(font_id)
            if font_id in state["completed"]:
                state["completed"].remove(font_id)
        elif action == "unskip":
            if font_id in state["skipped"]:
                state["skipped"].remove(font_id)
        elif action == "complete":
            if font_id not in state["completed"]:
                state["completed"].append(font_id)
            if font_id in state["skipped"]:
                state["skipped"].remove(font_id)
        elif action == "delete":
            deleted_root = self.server.dataset_root / "_deleted" / font_id
            deleted_root.mkdir(parents=True, exist_ok=True)
            for name in ("glyphs", "masks_layer", "draft_masks_layer", "predictions", "metadata"):
                src = self.server.dataset_root / name / font_id
                if src.exists():
                    dst = deleted_root / name
                    if dst.exists():
                        shutil.rmtree(dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
            if font_id not in state["deleted"]:
                state["deleted"].append(font_id)
            if font_id not in state["skipped"]:
                state["skipped"].append(font_id)
            for key in ("prepared", "completed"):
                if font_id in state[key]:
                    state[key].remove(font_id)
        else:
            raise ValueError("unsupported font action")
        save_state(self.server.dataset_root, state)
        self.send_json({"ok": True, "fonts": font_summary(self.server.dataset_root)})

    def resolve_roots(self, query: str, payload: dict | None = None) -> tuple[Path, Path, Path | None]:
        font_id = ""
        if query:
            font_id = parse_qs(query).get("font_id", [""])[0]
        if not font_id and payload:
            font_id = str(payload.get("fontId", ""))
        if self.server.dataset_root is not None and font_id:
            font_id = safe_font_id(font_id)
            return (
                self.server.dataset_root / "glyphs" / font_id,
                self.server.dataset_root / "masks_layer" / font_id,
                self.server.dataset_root / "draft_masks_layer" / font_id,
            )
        if self.server.glyph_dir is None or self.server.mask_dir is None:
            raise ValueError("font_id is required")
        return self.server.glyph_dir, self.server.mask_dir, self.server.draft_dir

    def send_image_from_query(self, query: str, root: Path, required: bool) -> None:
        rel = safe_rel_path(parse_qs(query).get("file", [""])[0])
        path = root / rel
        self.send_file(path, required)

    def send_layer_from_query(self, query: str, root: Path, required: bool) -> None:
        qs = parse_qs(query)
        rel = safe_rel_path(qs.get("file", [""])[0])
        layer = layer_from_query(query)
        path = root / rel.with_suffix("") / f"{layer}.png"
        self.send_file(path, required)

    def send_file(self, path: Path, required: bool) -> None:
        if not path.exists():
            if required:
                self.send_error(HTTPStatus.NOT_FOUND, "image not found")
            else:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_static(self, rel_name: str) -> None:
        rel = safe_rel_path(rel_name)
        path = STATIC_DIR / rel
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_file(path, required=True)

    def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class LayerMaskEditorServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], glyph_dir: Path, mask_dir: Path, draft_dir: Path | None) -> None:
        super().__init__(address, LayerMaskEditorHandler)
        self.glyph_dir = glyph_dir.resolve() if glyph_dir else None
        self.mask_dir = mask_dir.resolve() if mask_dir else None
        self.draft_dir = draft_dir.resolve() if draft_dir else None
        self.dataset_root: Path | None = None


def main() -> None:
    args = parse_args()
    glyph_dir = Path(args.glyph_dir) if args.glyph_dir else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    draft_dir = Path(args.draft_dir) if args.draft_dir else None
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is None and args.glyph_dir is None:
        raise SystemExit("either --glyph_dir/--mask_dir or --dataset_root is required")
    if dataset_root is None and glyph_dir is not None and not glyph_dir.exists():
        raise SystemExit(f"glyph_dir does not exist: {glyph_dir}")
    if dataset_root is None and mask_dir is None:
        raise SystemExit("--mask_dir is required with --glyph_dir")
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if draft_dir is not None and not draft_dir.exists():
        raise SystemExit(f"draft_dir does not exist: {draft_dir}")
    server = LayerMaskEditorServer((args.host, args.port), glyph_dir if args.glyph_dir else None, mask_dir, draft_dir)
    server.dataset_root = dataset_root.resolve() if dataset_root else None
    print(f"Layer mask editor running at http://{args.host}:{args.port}")
    if server.dataset_root:
        print(f"dataset_root: {server.dataset_root}")
    else:
        print(f"glyph_dir: {server.glyph_dir}")
        print(f"mask_dir:  {server.mask_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
