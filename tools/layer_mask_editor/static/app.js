const COLORS = {
  L: [255, 60, 60],
  V: [40, 210, 90],
  T: [80, 130, 255],
  ignore: [160, 160, 160],
};
const state = {
  files: [],
  fonts: [],
  managedFonts: false,
  fontId: "",
  index: 0,
  layer: "L",
  mode: "brush",
  size: 512,
  layers: {},
  glyphImage: null,
  glyphImageData: null,
  dirty: false,
  undo: [],
  redo: [],
  zoom: 1,
  panX: 0,
  panY: 0,
  panning: false,
  spaceDown: false,
  panStart: null,
  panFrame: null,
  canvasFrame: null,
  lastPaintPoint: null,
  lastPointerEvent: null,
  pointerInsideEdit: false,
};
const els = {
  files: document.getElementById("files"),
  fontPanel: document.getElementById("fontPanel"),
  fontSelect: document.getElementById("fontSelect"),
  fontMeta: document.getElementById("fontMeta"),
  progress: document.getElementById("progress"),
  glyphName: document.getElementById("glyphName"),
  glyphChar: document.getElementById("glyphChar"),
  glyph: document.getElementById("glyph"),
  overlay: document.getElementById("overlay"),
  edit: document.getElementById("edit"),
  previewL: document.getElementById("previewL"),
  previewV: document.getElementById("previewV"),
  previewT: document.getElementById("previewT"),
  brush: document.getElementById("brush"),
  brushValue: document.getElementById("brushValue"),
  brushCursor: document.getElementById("brushCursor"),
  viewport: document.getElementById("viewport"),
  stage: document.getElementById("stage"),
  strokeOnly: document.getElementById("strokeOnly"),
  fillRespectLayers: document.getElementById("fillRespectLayers"),
  message: document.getElementById("message"),
};
const ctx = {
  glyph: els.glyph.getContext("2d", { willReadFrequently: true }),
  overlay: els.overlay.getContext("2d", { willReadFrequently: true }),
  edit: els.edit.getContext("2d", { willReadFrequently: true }),
  previewL: els.previewL.getContext("2d", { willReadFrequently: true }),
  previewV: els.previewV.getContext("2d", { willReadFrequently: true }),
  previewT: els.previewT.getContext("2d", { willReadFrequently: true }),
};

function current() { return state.files[state.index]; }
function currentFont() { return state.fonts.find((font) => font.fontId === state.fontId); }
function msg(text) { els.message.textContent = text; }
function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  if (state.fontId) url.searchParams.set("font_id", state.fontId);
  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
  return url.toString();
}
function snapshotLayers() {
  const snapshot = {};
  for (const layer of ["L", "V", "T", "ignore"]) {
    snapshot[layer] = state.layers[layer]
      .getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, state.size, state.size);
  }
  return snapshot;
}
function restoreLayers(snapshot) {
  for (const layer of ["L", "V", "T", "ignore"]) {
    state.layers[layer]
      .getContext("2d", { willReadFrequently: true })
      .putImageData(snapshot[layer], 0, 0);
  }
  state.dirty = true;
  render();
}
function pushUndo() {
  if (!state.layers.L) return;
  state.undo.push(snapshotLayers());
  if (state.undo.length > 50) state.undo.shift();
  state.redo = [];
}
function undo() {
  if (!state.undo.length) return;
  state.redo.push(snapshotLayers());
  restoreLayers(state.undo.pop());
  msg("undo");
}
function redo() {
  if (!state.redo.length) return;
  state.undo.push(snapshotLayers());
  restoreLayers(state.redo.pop());
  msg("redo");
}
function blankLayer() {
  const c = document.createElement("canvas");
  c.width = state.size; c.height = state.size;
  return c;
}
function setCanvasSize(size) {
  state.size = size;
  [els.glyph, els.overlay, els.edit].forEach((c) => { c.width = size; c.height = size; });
  applyZoom();
}
function applyViewportTransform() {
  els.stage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  updateBrushCursor();
}
function scheduleViewportTransform() {
  if (state.panFrame) return;
  state.panFrame = requestAnimationFrame(() => {
    state.panFrame = null;
    applyViewportTransform();
  });
}
function applyZoom() {
  msg(`zoom ${Math.round(state.zoom * 100)}%`);
  applyViewportTransform();
}
function renderList() {
  els.files.innerHTML = "";
  els.progress.textContent = `${state.files.filter((f) => f.saved).length}/${state.files.length}`;
  state.files.forEach((file, i) => {
    const b = document.createElement("button");
    b.className = `file ${i === state.index ? "active" : ""}`;
    b.innerHTML = `<span>${file.saved ? "*" : ""}</span><span>${file.code}</span><span>${file.char || ""}</span>`;
    b.onclick = () => go(i);
    els.files.appendChild(b);
  });
}
function renderFonts() {
  if (!state.managedFonts) return;
  els.fontPanel.classList.remove("hidden");
  els.fontSelect.innerHTML = "";
  state.fonts.forEach((font) => {
    const option = document.createElement("option");
    option.value = font.fontId;
    option.textContent = `${font.fontId} | ${font.displayName} | ${font.savedCount}/${font.glyphCount} | ${font.status}`;
    els.fontSelect.appendChild(option);
  });
  els.fontSelect.value = state.fontId;
  const font = currentFont();
  els.fontMeta.textContent = font ? `${font.displayName} - ${font.status} - saved ${font.savedCount}/${font.glyphCount}` : "";
}
function drawGlyph() {
  ctx.glyph.clearRect(0, 0, state.size, state.size);
  ctx.glyph.drawImage(state.glyphImage, 0, 0, state.size, state.size);
}
function layerData(layer) {
  return state.layers[layer].getContext("2d", { willReadFrequently: true }).getImageData(0, 0, state.size, state.size);
}
function renderOverlay() {
  ctx.overlay.drawImage(state.glyphImage, 0, 0, state.size, state.size);
  const base = ctx.overlay.getImageData(0, 0, state.size, state.size);
  const maps = Object.fromEntries(["L", "V", "T", "ignore"].map((l) => [l, layerData(l).data]));
  for (let p = 0; p < base.data.length; p += 4) {
    const colors = [];
    for (const layer of ["L", "V", "T", "ignore"]) {
      if (maps[layer][p] > 127) colors.push(COLORS[layer]);
    }
    if (!colors.length) continue;
    const avg = colors.reduce((a, c) => [a[0] + c[0], a[1] + c[1], a[2] + c[2]], [0, 0, 0]).map((v) => v / colors.length);
    base.data[p] = base.data[p] * 0.55 + avg[0] * 0.45;
    base.data[p + 1] = base.data[p + 1] * 0.55 + avg[1] * 0.45;
    base.data[p + 2] = base.data[p + 2] * 0.55 + avg[2] * 0.45;
  }
  ctx.overlay.putImageData(base, 0, 0);
}
function renderEdit() {
  const composite = buildCompositeImageData(0.24);
  const data = layerData(state.layer).data;
  const color = COLORS[state.layer];
  for (let p = 0; p < composite.data.length; p += 4) {
    if (data[p] > 127) {
      composite.data[p] = composite.data[p] * 0.28 + color[0] * 0.72;
      composite.data[p + 1] = composite.data[p + 1] * 0.28 + color[1] * 0.72;
      composite.data[p + 2] = composite.data[p + 2] * 0.28 + color[2] * 0.72;
      composite.data[p + 3] = 255;
    }
  }
  ctx.edit.putImageData(composite, 0, 0);
}
function buildCompositeImageData(alpha) {
  const baseCanvas = document.createElement("canvas");
  baseCanvas.width = state.size;
  baseCanvas.height = state.size;
  const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
  baseCtx.fillStyle = "#fff";
  baseCtx.fillRect(0, 0, state.size, state.size);
  baseCtx.globalAlpha = 0.38;
  baseCtx.drawImage(state.glyphImage, 0, 0, state.size, state.size);
  baseCtx.globalAlpha = 1;
  const base = baseCtx.getImageData(0, 0, state.size, state.size);
  const maps = Object.fromEntries(["L", "V", "T", "ignore"].map((l) => [l, layerData(l).data]));
  for (let p = 0; p < base.data.length; p += 4) {
    const colors = [];
    for (const layer of ["L", "V", "T", "ignore"]) {
      if (maps[layer][p] > 127) colors.push(COLORS[layer]);
    }
    if (!colors.length) continue;
    const avg = colors.reduce((a, c) => [a[0] + c[0], a[1] + c[1], a[2] + c[2]], [0, 0, 0]).map((v) => v / colors.length);
    base.data[p] = base.data[p] * (1 - alpha) + avg[0] * alpha;
    base.data[p + 1] = base.data[p + 1] * (1 - alpha) + avg[1] * alpha;
    base.data[p + 2] = base.data[p + 2] * (1 - alpha) + avg[2] * alpha;
  }
  return base;
}
function render() {
  renderCanvases();
  renderList();
  renderControls();
}
function renderCanvases() {
  drawGlyph();
  renderOverlay();
  renderEdit();
  renderLayerPreviews();
  updateBrushCursor();
}
function layerPresence() {
  const char = current()?.char || "";
  if (char.length !== 1) return { L: true, V: true, T: true };
  const code = char.charCodeAt(0);
  if (code < 0xac00 || code > 0xd7a3) return { L: true, V: true, T: true };
  const syllableIndex = code - 0xac00;
  const tIndex = syllableIndex % 28;
  return { L: true, V: true, T: tIndex !== 0 };
}
function renderLayerPreviews() {
  if (!state.glyphImage || !state.layers.L) return;
  const presence = layerPresence();
  for (const layer of ["L", "V", "T"]) {
    const canvas = els[`preview${layer}`];
    const previewCtx = ctx[`preview${layer}`];
    const wrapper = canvas.closest(".layer-preview");
    canvas.width = 128;
    canvas.height = 128;
    wrapper.classList.toggle("absent", !presence[layer]);
    previewCtx.fillStyle = "#fff";
    previewCtx.fillRect(0, 0, 128, 128);
    if (!presence[layer]) {
      previewCtx.strokeStyle = "#c7ccd4";
      previewCtx.lineWidth = 2;
      previewCtx.beginPath();
      previewCtx.moveTo(8, 120);
      previewCtx.lineTo(120, 8);
      previewCtx.stroke();
      return;
    }
    const mask = layerData(layer);
    const color = COLORS[layer];
    const data = mask.data;
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = state.size;
    previewCanvas.height = state.size;
    const previewLayerCtx = previewCanvas.getContext("2d", { willReadFrequently: true });
    previewLayerCtx.putImageData(mask, 0, 0);
    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = 128;
    smallCanvas.height = 128;
    const smallCtx = smallCanvas.getContext("2d", { willReadFrequently: true });
    smallCtx.imageSmoothingEnabled = false;
    smallCtx.drawImage(previewCanvas, 0, 0, 128, 128);
    const smallMask = smallCtx.getImageData(0, 0, 128, 128).data;
    const imageData = previewCtx.getImageData(0, 0, 128, 128);
    const out = imageData.data;
    for (let y = 0; y < 128; y += 1) {
      for (let x = 0; x < 128; x += 1) {
        const src = (y * 128 + x) * 4;
        if (smallMask[src] <= 127) continue;
        const dst = (y * 128 + x) * 4;
        out[dst] = color[0];
        out[dst + 1] = color[1];
        out[dst + 2] = color[2];
        out[dst + 3] = 255;
      }
    }
    previewCtx.putImageData(imageData, 0, 0);
  }
}
function scheduleRenderCanvases() {
  if (state.canvasFrame) return;
  state.canvasFrame = requestAnimationFrame(() => {
    state.canvasFrame = null;
    renderCanvases();
  });
}
function renderControls() {
  document.querySelectorAll("[data-layer]").forEach((b) => b.classList.toggle("active", b.dataset.layer === state.layer));
  document.getElementById("brushMode").classList.toggle("active", state.mode === "brush");
  document.getElementById("fillMode").classList.toggle("active", state.mode === "fill");
  document.getElementById("erase").classList.toggle("active", state.mode === "erase");
  els.edit.classList.toggle("brush-cursor", state.mode === "brush" || state.mode === "erase");
  els.viewport.classList.toggle("panning", state.spaceDown);
  els.viewport.classList.toggle("dragging", state.panning);
  updateBrushValue();
  updateBrushCursor();
}
function loadImg(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url.startsWith("blob:") ? url : `${url}&cache=${Date.now()}`;
  });
}
async function loadLayer(layer, draft = false) {
  const endpoint = draft ? "/api/draft-layer" : "/api/layer";
  const res = await fetch(apiUrl(endpoint, { file: current().file, layer }));
  const canvas = blankLayer();
  if (res.status === 204 || res.status === 404) return canvas;
  if (!res.ok) throw new Error(`Could not load ${layer}`);
  const blob = await res.blob();
  const img = await loadImg(URL.createObjectURL(blob));
  canvas.getContext("2d").drawImage(img, 0, 0, state.size, state.size);
  return canvas;
}
async function go(index) {
  state.index = Math.max(0, Math.min(state.files.length - 1, index));
  const file = current();
  els.glyphName.textContent = file.code;
  els.glyphChar.textContent = file.char || "-";
  state.glyphImage = await loadImg(apiUrl("/api/glyph", { file: file.file }));
  setCanvasSize(state.glyphImage.naturalWidth || state.glyphImage.width);
  if (state.index === 0 && state.panX === 0 && state.panY === 0) {
    state.panX = 0;
    state.panY = 0;
    applyViewportTransform();
  }
  const native = document.createElement("canvas");
  native.width = state.size;
  native.height = state.size;
  native.getContext("2d", { willReadFrequently: true }).drawImage(state.glyphImage, 0, 0, state.size, state.size);
  state.glyphImageData = native.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, state.size, state.size);
  state.layers = {};
  for (const layer of ["L", "V", "T", "ignore"]) state.layers[layer] = await loadLayer(layer, false);
  state.undo = [];
  state.redo = [];
  state.dirty = false;
  render();
}
async function save() {
  const layers = {};
  for (const layer of ["L", "V", "T", "ignore"]) layers[layer] = state.layers[layer].toDataURL("image/png");
  const res = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fontId: state.fontId, file: current().file, layers, metadata: { status: "saved" } }) });
  if (!res.ok) throw new Error(await res.text());
  current().saved = true;
  state.dirty = false;
  if (state.managedFonts) {
    const font = currentFont();
    if (font) font.savedCount = state.files.filter((file) => file.saved).length;
    renderFonts();
  }
  msg("saved");
  renderList();
  return true;
}
async function loadDraft() {
  pushUndo();
  for (const layer of ["L", "V", "T", "ignore"]) state.layers[layer] = await loadLayer(layer, true);
  state.dirty = true;
  render();
}
function canvasToNativePoint(event) {
  const rect = els.edit.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(state.size - 1, Math.floor(((event.clientX - rect.left) / rect.width) * state.size))),
    y: Math.max(0, Math.min(state.size - 1, Math.floor(((event.clientY - rect.top) / rect.height) * state.size))),
  };
}
function updateBrushCursor(event = null) {
  if (!els.brushCursor) return;
  const show = state.pointerInsideEdit && !state.panning && !state.spaceDown && (state.mode === "brush" || state.mode === "erase");
  if (!show) {
    els.brushCursor.style.display = "none";
    return;
  }
  const rect = els.edit.getBoundingClientRect();
  const cssDiameter = Math.max(2, Number(els.brush.value) * (rect.width / state.size));
  els.brushCursor.style.width = `${cssDiameter}px`;
  els.brushCursor.style.height = `${cssDiameter}px`;
  if (event) {
    state.lastPointerEvent = { clientX: event.clientX, clientY: event.clientY };
  }
  const point = state.lastPointerEvent;
  if (point) {
    els.brushCursor.style.left = `${point.clientX}px`;
    els.brushCursor.style.top = `${point.clientY}px`;
  }
  els.brushCursor.style.borderColor = state.mode === "erase" ? "#d93025" : "#1a73e8";
  els.brushCursor.style.display = "block";
}
function isForegroundAt(x, y) {
  if (!state.glyphImageData) return true;
  const i = (y * state.size + x) * 4;
  const d = state.glyphImageData.data;
  return (d[i] + d[i + 1] + d[i + 2]) / 3 < 245;
}
function activeLayerImageData() {
  const c = state.layers[state.layer].getContext("2d");
  return [c, c.getImageData(0, 0, state.size, state.size)];
}
function pixelOn(data, x, y) {
  return data[(y * state.size + x) * 4] > 127;
}
function otherLayerMaps() {
  if (!els.fillRespectLayers.checked) return false;
  const maps = {};
  for (const layer of ["L", "V", "T"]) {
    if (layer === state.layer) continue;
    maps[layer] = layerData(layer).data;
  }
  return maps;
}
function otherLayerOn(x, y, maps = null) {
  const layerMaps = maps === null ? otherLayerMaps() : maps;
  if (!layerMaps) return false;
  const i = (y * state.size + x) * 4;
  for (const data of Object.values(layerMaps)) {
    if (data[i] > 127) return true;
  }
  return false;
}
function setPixel(data, x, y, on) {
  const i = (y * state.size + x) * 4;
  const v = on ? 255 : 0;
  data[i] = v;
  data[i + 1] = v;
  data[i + 2] = v;
  data[i + 3] = 255;
}
function stampBrush(data, x, y) {
  const radius = Math.max(1, Math.floor(Number(els.brush.value) / 2));
  const r2 = radius * radius;
  const on = state.mode !== "erase";
  const constrainToGlyph = els.strokeOnly.checked && on;
  for (let yy = Math.max(0, y - radius); yy <= Math.min(state.size - 1, y + radius); yy += 1) {
    for (let xx = Math.max(0, x - radius); xx <= Math.min(state.size - 1, x + radius); xx += 1) {
      const dx = xx - x;
      const dy = yy - y;
      if (dx * dx + dy * dy > r2) continue;
      if (constrainToGlyph && !isForegroundAt(xx, yy)) continue;
      setPixel(data, xx, yy, on);
    }
  }
}
function paintAt(x, y) {
  const [layerCtx, imageData] = activeLayerImageData();
  stampBrush(imageData.data, x, y);
  layerCtx.putImageData(imageData, 0, 0);
  state.dirty = true;
  scheduleRenderCanvases();
}
function paintStrokeTo(x, y) {
  const [layerCtx, imageData] = activeLayerImageData();
  const data = imageData.data;
  if (!state.lastPaintPoint) {
    stampBrush(data, x, y);
    layerCtx.putImageData(imageData, 0, 0);
    state.dirty = true;
    scheduleRenderCanvases();
    state.lastPaintPoint = { x, y };
    return;
  }
  const start = state.lastPaintPoint;
  const dx = x - start.x;
  const dy = y - start.y;
  const distance = Math.hypot(dx, dy);
  const radius = Math.max(1, Math.floor(Number(els.brush.value) / 2));
  const step = Math.max(1, radius * 0.45);
  const steps = Math.max(1, Math.ceil(distance / step));
  for (let i = 1; i <= steps; i += 1) {
    const t = i / steps;
    stampBrush(data, Math.round(start.x + dx * t), Math.round(start.y + dy * t));
  }
  layerCtx.putImageData(imageData, 0, 0);
  state.dirty = true;
  scheduleRenderCanvases();
  state.lastPaintPoint = { x, y };
}
function fillAt(x, y) {
  const [layerCtx, imageData] = activeLayerImageData();
  const data = imageData.data;
  const sourceOn = pixelOn(data, x, y);
  const targetOn = state.mode === "erase" ? false : true;
  const layerBarriers = otherLayerMaps();
  if (sourceOn === targetOn) {
    msg("Already filled.");
    return;
  }
  const constrainToGlyph = els.strokeOnly.checked && targetOn;
  if (targetOn && otherLayerOn(x, y, layerBarriers)) {
    msg("Fill stopped by another layer.");
    return;
  }
  const visited = new Uint8Array(state.size * state.size);
  const queue = [[x, y]];
  visited[y * state.size + x] = 1;
  let q = 0;
  let changed = 0;
  while (q < queue.length) {
    const [cx, cy] = queue[q];
    q += 1;
    if (constrainToGlyph && !isForegroundAt(cx, cy)) continue;
    if (targetOn && otherLayerOn(cx, cy, layerBarriers)) continue;
    if (pixelOn(data, cx, cy) !== sourceOn) continue;
    setPixel(data, cx, cy, targetOn);
    changed += 1;
    for (const [nx, ny] of [[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]]) {
      if (nx < 0 || ny < 0 || nx >= state.size || ny >= state.size) continue;
      const idx = ny * state.size + nx;
      if (visited[idx]) continue;
      visited[idx] = 1;
      queue.push([nx, ny]);
    }
  }
  layerCtx.putImageData(imageData, 0, 0);
  if (changed > 0) state.dirty = true;
  msg(`Filled ${changed} pixels.`);
  scheduleRenderCanvases();
}
function updateBrushValue() {
  if (!els.brushValue) return;
  els.brushValue.textContent = `${els.brush.value} px`;
}
async function saveIfDirty() {
  if (!state.dirty) return true;
  return save();
}
async function saveAndGo(delta) {
  const ok = await saveIfDirty();
  if (!ok) return;
  await go(state.index + delta);
}
async function init() {
  await loadFonts();
  bindControls();
  const data = await (await fetch(apiUrl("/api/list"))).json();
  state.files = data.files;
  if (!state.files.length) {
    renderList();
    msg("No glyphs found for selected font.");
    return;
  }
  await go(0);
}
function bindControls() {
  document.querySelectorAll("[data-layer]").forEach((b) => b.onclick = () => { state.layer = b.dataset.layer; render(); });
  document.getElementById("brushMode").onclick = () => { state.mode = "brush"; renderControls(); };
  document.getElementById("fillMode").onclick = () => { state.mode = "fill"; renderControls(); };
  document.getElementById("erase").onclick = () => { state.mode = "erase"; renderControls(); };
  document.getElementById("prev").onclick = () => go(state.index - 1);
  document.getElementById("next").onclick = () => go(state.index + 1);
  document.getElementById("save").onclick = save;
  document.getElementById("loadDraft").onclick = loadDraft;
  els.fontSelect.addEventListener("change", async () => {
    const ok = await saveIfDirty();
    if (!ok) return;
    state.fontId = els.fontSelect.value;
    localStorage.setItem("layerEditor.fontId", state.fontId);
    await reloadCurrentFont();
  });
  document.getElementById("markComplete").onclick = () => fontAction("complete");
  document.getElementById("skipFont").onclick = () => fontAction("skip");
  document.getElementById("deleteFont").onclick = async () => {
    if (!confirm("Move this font to _deleted and exclude it from training?")) return;
    await fontAction("delete");
  };
  els.brush.addEventListener("input", () => {
    updateBrushValue();
    updateBrushCursor();
  });
  els.viewport.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const before = state.zoom;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const next = Math.max(0.35, Math.min(5, state.zoom * factor));
    const rect = els.viewport.getBoundingClientRect();
    const vx = e.clientX - rect.left;
    const vy = e.clientY - rect.top;
    const worldX = (vx - state.panX) / before;
    const worldY = (vy - state.panY) / before;
    state.zoom = next;
    state.panX = vx - worldX * next;
    state.panY = vy - worldY * next;
    applyZoom();
  }, { passive: false });
  els.viewport.addEventListener("pointerdown", (e) => {
    if (e.button !== 1 && !(state.spaceDown && e.button === 0)) return;
    e.preventDefault();
    state.panning = true;
    state.panStart = { x: e.clientX, y: e.clientY, panX: state.panX, panY: state.panY };
    els.viewport.setPointerCapture(e.pointerId);
    render();
  });
  els.viewport.addEventListener("pointermove", (e) => {
    if (!state.panning || !state.panStart) return;
    e.preventDefault();
    state.panX = state.panStart.panX + e.clientX - state.panStart.x;
    state.panY = state.panStart.panY + e.clientY - state.panStart.y;
    scheduleViewportTransform();
  });
  function stopPanning(e) {
    if (!state.panning) return;
    state.panning = false;
    state.panStart = null;
    applyViewportTransform();
    try {
      els.viewport.releasePointerCapture(e.pointerId);
    } catch (_err) {
      // Pointer capture may already be released by the browser.
    }
    render();
  }
  els.viewport.addEventListener("pointerup", stopPanning);
  els.viewport.addEventListener("pointercancel", stopPanning);
  let down = false;
  els.edit.addEventListener("pointerenter", (e) => {
    if (state.spaceDown || state.panning) return;
    state.pointerInsideEdit = true;
    state.lastPointerEvent = { clientX: e.clientX, clientY: e.clientY };
    updateBrushCursor(e);
  });
  els.edit.addEventListener("pointerleave", () => {
    state.pointerInsideEdit = false;
    state.lastPointerEvent = null;
    updateBrushCursor();
  });
  els.edit.addEventListener("pointermove", (e) => {
    if (state.spaceDown || state.panning) return;
    state.pointerInsideEdit = true;
    state.lastPointerEvent = { clientX: e.clientX, clientY: e.clientY };
    updateBrushCursor(e);
  });
  els.edit.addEventListener("pointerdown", (e) => {
    if (e.button === 1 || state.spaceDown) return;
    down = true;
    pushUndo();
    updateBrushCursor(e);
    const { x, y } = canvasToNativePoint(e);
    if (state.mode === "fill") fillAt(x, y);
    else {
      state.lastPaintPoint = null;
      paintStrokeTo(x, y);
    }
  });
  els.edit.addEventListener("pointermove", (e) => {
    if (!down || state.mode === "fill") return;
    const { x, y } = canvasToNativePoint(e);
    paintStrokeTo(x, y);
  });
  window.addEventListener("pointerup", () => {
    down = false;
    state.lastPaintPoint = null;
  });
  window.addEventListener("keydown", async (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      if (state.spaceDown) return;
      state.spaceDown = true;
      updateBrushCursor();
      render();
      return;
    }
    if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      redo();
      return;
    }
    if (e.ctrlKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      undo();
      return;
    }
    if (e.ctrlKey && e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
      return;
    }
    if (e.key === "Tab") {
      e.preventDefault();
      await saveAndGo(1);
      return;
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      await saveAndGo(1);
      return;
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      await saveAndGo(-1);
      return;
    }
    let layerChanged = false;
    if (e.key === "1") { state.layer = "L"; layerChanged = true; }
    if (e.key === "2") { state.layer = "V"; layerChanged = true; }
    if (e.key === "3") { state.layer = "T"; layerChanged = true; }
    if (e.key.toLowerCase() === "b") state.mode = "brush";
    if (e.key.toLowerCase() === "f") state.mode = "fill";
    if (e.key.toLowerCase() === "e") state.mode = "erase";
    if (e.key === "[") {
      els.brush.value = Math.max(Number(els.brush.min), Number(els.brush.value) - 2);
      updateBrushValue();
      updateBrushCursor();
    }
    if (e.key === "]") {
      els.brush.value = Math.min(Number(els.brush.max), Number(els.brush.value) + 2);
      updateBrushValue();
      updateBrushCursor();
    }
    if (e.key.toLowerCase() === "s") {
      e.preventDefault();
      await save();
    }
    if (e.key === "Enter") { await save(); await go(state.index + 1); }
    if (layerChanged) render();
    else renderControls();
  });
  window.addEventListener("keyup", (e) => {
    if (e.code !== "Space") return;
    state.spaceDown = false;
    if (state.panning) {
      state.panning = false;
      state.panStart = null;
    }
    render();
  });
}
async function loadFonts() {
  const data = await (await fetch("/api/fonts")).json();
  state.managedFonts = Boolean(data.managed);
  state.fonts = data.fonts || [];
  if (!state.managedFonts || !state.fonts.length) return;
  const saved = localStorage.getItem("layerEditor.fontId");
  const usableFont = (font) => font.status !== "deleted" && font.glyphCount > 0;
  const preferred = state.fonts.find((font) => font.fontId === saved && usableFont(font));
  const first = state.fonts.find((font) => font.status !== "deleted" && font.glyphCount > 0) || state.fonts[0];
  state.fontId = (preferred || first).fontId;
  localStorage.setItem("layerEditor.fontId", state.fontId);
  renderFonts();
}
async function reloadCurrentFont() {
  const data = await (await fetch(apiUrl("/api/list"))).json();
  state.files = data.files;
  state.index = 0;
  renderFonts();
  if (!state.files.length) {
    els.files.innerHTML = "";
    msg("No glyphs for selected font.");
    return;
  }
  await go(0);
}
async function fontAction(action) {
  if (!state.fontId) return;
  const res = await fetch("/api/font-action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fontId: state.fontId, action }),
  });
  if (!res.ok) {
    msg("font action failed");
    return;
  }
  const data = await res.json();
  state.fonts = data.fonts || state.fonts;
  if (action === "delete") {
    const next = state.fonts.find((font) => font.status !== "deleted" && font.glyphCount > 0);
    state.fontId = next ? next.fontId : "";
  }
  renderFonts();
  if (state.fontId) await reloadCurrentFont();
  msg(action);
}
init().catch((err) => msg(err.message));
