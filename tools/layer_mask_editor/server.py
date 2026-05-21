from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
STATIC_DIR = ROOT / "static"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LAYERS = {"L", "V", "T", "ignore"}
SKIP_DIRS = {"debug", "_debug", "_metadata", "_overlay_previews", "overlay_previews"}
RUN_PREVIEW_FILES = {
    "composite": Path("overlays") / "composite.png",
    "L": Path("binary") / "L.png",
    "V": Path("binary") / "V.png",
    "T": Path("binary") / "T.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local L/V/T layer mask annotation tool.")
    parser.add_argument("--glyph_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--draft_dir", default=None)
    parser.add_argument("--dataset_root", default=None, help="Optional Clova dataset root for multi-font selection.")
    parser.add_argument("--runs_dir", default="runs", help="Optional runs root for prediction preview browsing.")
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


def safe_run_rel(raw: str) -> Path:
    rel = safe_rel_path(raw)
    if len(rel.parts) > 8:
        raise ValueError("run path is too deep")
    return rel


def safe_job_name(value: str, fallback: str) -> str:
    value = (value or fallback).strip().replace("\\", "/").strip("/")
    if not value:
        value = fallback
    rel = safe_rel_path(value)
    if len(rel.parts) > 4:
        raise ValueError("output path is too deep")
    return rel.as_posix()


def repo_path(rel: str) -> Path:
    path = REPO_ROOT / safe_rel_path(rel)
    return path


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def list_run_prediction_roots(runs_dir: Path) -> list[dict]:
    if not runs_dir.exists():
        return []
    roots: list[dict] = []
    seen: set[Path] = set()
    for composite in sorted(runs_dir.rglob("overlays/composite.png")):
        glyph_dir = composite.parent.parent
        prediction_root = glyph_dir.parent
        if prediction_root in seen:
            continue
        seen.add(prediction_root)
        rel = prediction_root.relative_to(runs_dir)
        glyph_count = sum(1 for child in prediction_root.iterdir() if child.is_dir() and (child / "overlays" / "composite.png").exists())
        roots.append({"id": rel.as_posix(), "label": rel.as_posix(), "glyphCount": glyph_count})
    return roots


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
            elif parsed.path == "/api/runs":
                self.send_json({"runsDir": str(self.server.runs_dir), "runs": list_run_prediction_roots(self.server.runs_dir)})
            elif parsed.path == "/api/run-files":
                self.send_run_files(parsed.query)
            elif parsed.path == "/api/run-preview":
                self.send_run_preview(parsed.query)
            elif parsed.path == "/api/jobs":
                self.send_json({"jobs": self.server.job_summaries()})
            elif parsed.path == "/api/job":
                self.send_job(parsed.query)
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
            elif parsed.path == "/api/start-train":
                self.handle_start_train()
            elif parsed.path == "/api/start-predict":
                self.handle_start_predict()
            elif parsed.path == "/api/stop-job":
                self.handle_stop_job()
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

    def handle_start_train(self) -> None:
        payload = self.read_json()
        if self.server.dataset_root is None:
            raise ValueError("training from the web UI requires --dataset_root")
        source = str(payload.get("source", "completed"))
        save_dir = safe_job_name(str(payload.get("saveDir", "runs/web_train")), "runs/web_train")
        image_size = int(payload.get("imageSize", 128))
        epochs = int(payload.get("epochs", 20))
        batch_size = int(payload.get("batchSize", 8))
        device = str(payload.get("device", "cuda"))
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        cmd = [
            sys.executable,
            "hangul_layer_seg/train.py",
            "--glyph_dir",
            str(self.server.dataset_root / "glyphs"),
            "--mask_dir",
            str(self.server.dataset_root / "masks_layer"),
            "--save_dir",
            save_dir,
            "--image_size",
            str(image_size),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--device",
            device,
        ]
        if source == "completed":
            font_ids = load_state(self.server.dataset_root).get("completed", [])
            if not font_ids:
                raise ValueError("no completed fonts found")
            cmd += ["--font_ids", ",".join(font_ids)]
        elif source == "current":
            font_id = safe_font_id(str(payload.get("fontId", "")))
            cmd += ["--font_ids", font_id]
        elif source != "all":
            raise ValueError("unsupported train source")
        if payload.get("amp"):
            cmd.append("--amp")
        if payload.get("channelsLast"):
            cmd.append("--channels_last")
        if payload.get("conditional"):
            cmd += ["--mapping_csv", "configs/mapping_214.csv", "--conditional", "--condition_embed_dim", str(int(payload.get("conditionEmbedDim", 8)))]
        penalty = float(payload.get("noFinalTPenalty", 0) or 0)
        if penalty > 0:
            if "--mapping_csv" not in cmd:
                cmd += ["--mapping_csv", "configs/mapping_214.csv"]
            cmd += ["--no_final_t_fp_penalty", str(penalty)]
        job = self.server.start_job("train", f"train {save_dir}", cmd)
        self.send_json({"ok": True, "job": job})

    def handle_start_predict(self) -> None:
        payload = self.read_json()
        if self.server.dataset_root is None:
            raise ValueError("prediction from the web UI requires --dataset_root")
        font_id = safe_font_id(str(payload.get("fontId", "")))
        ckpt = repo_path(str(payload.get("checkpoint", "")))
        if not ckpt.exists():
            raise ValueError(f"checkpoint does not exist: {ckpt.relative_to(REPO_ROOT)}")
        out_name = safe_job_name(str(payload.get("outputRun", "runs/web_predict")), "runs/web_predict")
        out_dir = REPO_ROOT / out_name / "predictions" / font_id
        device = str(payload.get("device", "cuda"))
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        cmd = [
            sys.executable,
            "hangul_layer_seg/predict.py",
            "--ckpt",
            str(ckpt),
            "--input_dir",
            str(self.server.dataset_root / "glyphs" / font_id),
            "--out_dir",
            str(out_dir),
            "--mapping_csv",
            "configs/mapping_214.csv",
            "--threshold",
            str(float(payload.get("threshold", 0.5))),
            "--device",
            device,
        ]
        if payload.get("suppressNoFinalT", True):
            cmd.append("--suppress_no_final_t")
        if payload.get("restrictToInk", True):
            cmd += ["--restrict_to_ink", "--ink_threshold", str(int(payload.get("inkThreshold", 245))), "--ink_dilate", str(int(payload.get("inkDilate", 1)))]
        min_area = int(payload.get("minComponentArea", 16))
        if min_area > 0:
            cmd += ["--min_component_area", str(min_area)]
        job = self.server.start_job("predict", f"predict {font_id}", cmd)
        self.send_json({"ok": True, "job": job})

    def handle_stop_job(self) -> None:
        payload = self.read_json()
        job_id = str(payload.get("jobId", ""))
        self.send_json({"ok": self.server.stop_job(job_id)})

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

    def send_run_preview(self, query: str) -> None:
        qs = parse_qs(query)
        run_rel = safe_run_rel(qs.get("run", [""])[0])
        glyph_id = safe_rel_path(qs.get("glyph_id", [""])[0])
        kind = qs.get("kind", ["composite"])[0]
        if kind not in RUN_PREVIEW_FILES:
            raise ValueError("invalid run preview kind")
        path = self.server.runs_dir / run_rel / glyph_id / RUN_PREVIEW_FILES[kind]
        self.send_file(path, required=False)

    def send_run_files(self, query: str) -> None:
        run_rel = safe_run_rel(parse_qs(query).get("run", [""])[0])
        run_root = self.server.runs_dir / run_rel
        if not run_root.exists():
            self.send_json({"files": []})
            return
        files = []
        for path in sorted(run_root.iterdir()):
            if not path.is_dir() or not (path / "overlays" / "composite.png").exists():
                continue
            files.append({"glyphId": path.name, "code": path.name, "char": code_to_char(path.name)})
        self.send_json({"files": files})

    def send_job(self, query: str) -> None:
        job_id = parse_qs(query).get("id", [""])[0]
        job = self.server.job_detail(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND, "job not found")
            return
        self.send_json(job)

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
    def __init__(self, address: tuple[str, int], glyph_dir: Path, mask_dir: Path, draft_dir: Path | None, runs_dir: Path) -> None:
        super().__init__(address, LayerMaskEditorHandler)
        self.glyph_dir = glyph_dir.resolve() if glyph_dir else None
        self.mask_dir = mask_dir.resolve() if mask_dir else None
        self.draft_dir = draft_dir.resolve() if draft_dir else None
        self.runs_dir = runs_dir.resolve()
        self.dataset_root: Path | None = None
        self.jobs: dict[str, dict] = {}

    def start_job(self, kind: str, name: str, cmd: list[str]) -> dict:
        job_id = uuid.uuid4().hex[:10]
        log_dir = REPO_ROOT / "runs" / "_web_jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        handle = log_path.open("w", encoding="utf-8", errors="replace")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        handle.close()
        job = {
            "id": job_id,
            "kind": kind,
            "name": name,
            "cmd": " ".join(str(part) for part in cmd),
            "logPath": str(log_path),
            "startedAt": time.time(),
            "returncode": None,
            "pid": process.pid,
            "process": process,
        }
        self.jobs[job_id] = job
        return self.job_public(job)

    def refresh_job(self, job: dict) -> None:
        process = job.get("process")
        if process is not None and job["returncode"] is None:
            job["returncode"] = process.poll()

    def job_public(self, job: dict) -> dict:
        self.refresh_job(job)
        status = "running" if job["returncode"] is None else ("done" if job["returncode"] == 0 else "failed")
        return {
            "id": job["id"],
            "kind": job["kind"],
            "name": job["name"],
            "status": status,
            "returncode": job["returncode"],
            "pid": job["pid"],
            "startedAt": job["startedAt"],
            "cmd": job["cmd"],
        }

    def job_summaries(self) -> list[dict]:
        return [self.job_public(job) for job in sorted(self.jobs.values(), key=lambda item: item["startedAt"], reverse=True)]

    def job_detail(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        out = self.job_public(job)
        out["log"] = tail_text(Path(job["logPath"]))
        return out

    def stop_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        self.refresh_job(job)
        if job["returncode"] is not None:
            return True
        process = job.get("process")
        if process is not None:
            process.terminate()
        return True


def main() -> None:
    args = parse_args()
    glyph_dir = Path(args.glyph_dir) if args.glyph_dir else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    draft_dir = Path(args.draft_dir) if args.draft_dir else None
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir
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
    server = LayerMaskEditorServer((args.host, args.port), glyph_dir if args.glyph_dir else None, mask_dir, draft_dir, runs_dir)
    server.dataset_root = dataset_root.resolve() if dataset_root else None
    print(f"Layer mask editor running at http://{args.host}:{args.port}")
    if server.dataset_root:
        print(f"dataset_root: {server.dataset_root}")
    else:
        print(f"glyph_dir: {server.glyph_dir}")
        print(f"mask_dir:  {server.mask_dir}")
    print(f"runs_dir:  {server.runs_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
