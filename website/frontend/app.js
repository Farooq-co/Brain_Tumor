const API = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const MAX = 15 * 1024 * 1024;
const $ = (id) => document.getElementById(id);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

let file = null, objectUrl = null, dims = "", data = null, view = "overlay", x = 50, classesLoaded = false;

const show = (el, on) => (el.hidden = !on);
const fmtSize = (b) => (b > 1048576 ? (b / 1048576).toFixed(2) + " MB" : (b / 1024).toFixed(0) + " KB");
const pct = (p) => (p > 0 && p < 0.001 ? "<0.1%" : (p * 100).toFixed(1) + "%");
const nice = (s) => { s = s.replace(/_/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); };
const stage = $("stage");

// graticule ticks (empty-state decoration only)
(() => {
  let d = "";
  for (let i = 10; i < 100; i += 10) if (i !== 50) d += `M${i} 48.5v3M48.5 ${i}h3`;
  $("ticks").innerHTML = `<path d="${d}"/>`;
})();

function setState(s) {
  stage.dataset.state = s;
  show($("sIdle"), s === "empty" || s === "loaded");
  show($("sBusy"), s === "busy");
  show($("sDone"), s === "done");
}

function setError(msg, html) {
  const el = $("error");
  if (html) el.innerHTML = msg; else el.textContent = msg || "";
  show(el, !!msg);
}

// ---- server status
async function checkHealth() {
  const el = $("health");
  try {
    const h = await (await fetch(API + "/api/health")).json();
    el.dataset.s = h.models_loaded ? "ok" : "off";
    el.querySelector("b").textContent = h.models_loaded ? "Models ready" : "Models not loaded";
    el.title = h.error || "";
    if (h.models_loaded && !classesLoaded) loadClasses();
  } catch {
    el.dataset.s = "off";
    el.querySelector("b").textContent = "Server offline";
  }
}
async function loadClasses() {
  try {
    const m = await (await fetch(API + "/api/model-info")).json();
    $("classList").textContent = m.class_names.map((c) => nice(c).toLowerCase()).join(", ") + ".";
    classesLoaded = true;
  } catch { /* leave the default text */ }
}

// ---- choosing a file
function pick(f) {
  setError("");
  if (!f) return;
  if (!/\.(jpe?g|png)$/i.test(f.name)) return setError("That file type is not supported. Choose a JPG, JPEG or PNG image.");
  if (f.size === 0) return setError("That file is empty. Choose a different image.");
  if (f.size > MAX) return setError("That file is larger than 15 MB. Choose a smaller image.");
  clearResult();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(f);
  const img = $("imgBase");
  img.onload = () => {
    dims = `${img.naturalWidth} × ${img.naturalHeight} px`;
    $("annTL").textContent = f.name;
    $("annTR").textContent = dims;
    $("meta").innerHTML = "";
    const b = document.createElement("b");
    b.textContent = f.name;
    $("meta").append(b, ` — ${fmtSize(f.size)}, ${dims}`);
  };
  img.onerror = () => { clearFile(); setError("That file could not be read as an image. Choose a different one."); };
  img.src = objectUrl;
  img.alt = "Selected MRI slice";
  show(img, true);
  file = f;
  show($("study"), true);
  setState("loaded");
}

function clearResult() {
  data = null;
  show($("imgTop"), false);
  show($("handle"), false);
  show($("views"), false);
  show($("enlargeBtn"), false);
  show($("saveBtn"), false);
  $("annBL").textContent = "";
  $("annBR").textContent = "";
}

function clearFile() {
  clearResult();
  file = null;
  $("file").value = "";
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  show($("imgBase"), false);
  $("imgBase").removeAttribute("src");
  show($("study"), false);
  $("annTL").textContent = "No study loaded";
  $("annTR").textContent = "";
  setError("");
  setState("empty");
}

$("browse").onclick = (e) => { e.stopPropagation(); $("file").click(); };
stage.addEventListener("click", (e) => { if (stage.dataset.state === "empty" && e.target !== $("browse")) $("file").click(); });
$("file").onchange = (e) => pick(e.target.files[0]);
["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); if (stage.dataset.state !== "busy") stage.classList.add("over-drop"); }));
["dragleave", "drop"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.remove("over-drop"); }));
stage.addEventListener("drop", (e) => { if (stage.dataset.state !== "busy") pick(e.dataTransfer.files[0]); });
$("removeBtn").onclick = clearFile;

// ---- analysis
$("analyzeBtn").onclick = async () => {
  if (!file) return setError("Choose an MRI image first.");
  setError("");
  clearResult();
  $("imgBase").src = objectUrl;
  setState("busy");
  $("analyzeBtn").disabled = $("removeBtn").disabled = true;
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(API + "/api/predict/analyze", { method: "POST", body });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof json.detail === "string" ? json.detail : `The server returned an error (${res.status}).`);
    data = json;
    render();
  } catch (err) {
    setState("loaded");
    if (err instanceof TypeError) setError("The analysis server is not running. Start it with <code>uvicorn backend.main:app --port 8000</code>, then press Analyze slice again.", true);
    else setError(err.message);
  } finally {
    $("analyzeBtn").disabled = $("removeBtn").disabled = false;
  }
};

// ---- results
function render() {
  const c = data.classification, s = data.segmentation || {};
  $("resClass").textContent = nice(c.predicted_class);
  $("resConf").textContent = `${pct(c.confidence)} model confidence`;

  const rows = [...c.probabilities].sort((a, b) => b.probability - a.probability);
  const tb = $("ranks");
  tb.innerHTML = "";
  rows.forEach((p) => {
    const tr = document.createElement("tr");
    if (p.index === c.predicted_index) tr.className = "lead";
    tr.innerHTML = `<th scope="row"></th><td><span class="rule"><i></i></span></td><td></td>`;
    tr.querySelector("th").textContent = nice(p.label);
    tr.querySelector("i").style.setProperty("--p", p.probability);
    tr.lastElementChild.textContent = pct(p.probability);
    tb.appendChild(tr);
  });
  tb.parentElement.classList.remove("in");

  const n = data.image_size;
  if (!s.available) $("resSeg").textContent = "The segmentation model is not loaded, so no region is marked.";
  else if (s.mask_pixels === 0) $("resSeg").textContent = `No pixel reached ${s.threshold}, so no region is marked.`;
  else $("resSeg").textContent = `${s.mask_area_percent}% of image area marked (${s.mask_pixels.toLocaleString()} pixels; not physical size).`;
  $("runTime").textContent = (data.processing_time_ms / 1000).toFixed(2) + " s";

  const views = [["original", "Original", data.original_image]];
  if (s.available) views.push(["mask", "Mask", s.mask_image], ["overlay", "Overlay", s.overlay_image]);
  views.push(["gradcam", "Grad-CAM", data.gradcam_image]);
  data.views = Object.fromEntries(views.map(([k, l, src]) => [k, { label: l, src }]));

  const box = $("views");
  box.innerHTML = "";
  views.forEach(([k, l]) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = l; b.dataset.k = k;
    b.onclick = () => setView(k);
    box.appendChild(b);
  });
  show(box, true);
  show($("enlargeBtn"), true);
  show($("saveBtn"), true);

  // use the server's resized original so every layer lines up pixel for pixel
  const base = $("imgBase");
  base.src = data.original_image;
  $("annTR").textContent = `Model input ${n} × ${n} px`;
  $("annBL").textContent = s.available ? `Mask threshold ${s.threshold}` : "";
  stage.dataset.state = "done";
  setState("done");
  x = 100;
  setView(s.available ? "overlay" : "gradcam");
  requestAnimationFrame(() => requestAnimationFrame(() => tb.parentElement.classList.add("in")));
  if (reduce) setX(50); else sweep(100, 50, 750);
}

function setView(k) {
  view = k;
  $("annBR").textContent = data.views[k].label;
  box_pressed(k);
  const top = $("imgTop"), h = $("handle");
  if (k === "original") { show(top, false); show(h, false); stage.style.cursor = "default"; return; }
  top.src = data.views[k].src;
  top.alt = data.views[k].label + " compared with the original";
  show(top, true); show(h, true);
  stage.style.cursor = "";
  if (x >= 98 && reduce === false && !sweeping) setX(50); else setX(x);
}
function box_pressed(k) {
  document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === k)));
}

// ---- wipe handle
let sweeping = false;
function setX(v) {
  x = Math.max(0, Math.min(100, v));
  stage.style.setProperty("--x", x + "%");
  $("handle").setAttribute("aria-valuenow", Math.round(x));
}
function sweep(from, to, ms) {
  sweeping = true;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    setX(from + (to - from) * e);
    if (k < 1 && sweeping) requestAnimationFrame(step); else sweeping = false;
  };
  requestAnimationFrame(step);
}
let dragging = false;
const at = (e) => { const r = stage.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * 100; };
stage.addEventListener("pointerdown", (e) => {
  if (stage.dataset.state !== "done" || view === "original" || e.target.closest("button")) return;
  dragging = true; sweeping = false;
  stage.setPointerCapture(e.pointerId);
  setX(at(e));
});
stage.addEventListener("pointermove", (e) => { if (dragging) setX(at(e)); });
["pointerup", "pointercancel"].forEach((t) => stage.addEventListener(t, () => (dragging = false)));
$("handle").addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 10 : 2;
  const map = { ArrowLeft: x - step, ArrowRight: x + step, Home: 0, End: 100 };
  if (e.key in map) { e.preventDefault(); sweeping = false; setX(map[e.key]); }
});

// ---- enlarge and save act on the view being shown
const shown = () => data.views[view].src;
$("enlargeBtn").onclick = () => { $("lbImg").src = shown(); $("lbImg").alt = data.views[view].label; $("lb").showModal(); };
$("lb").addEventListener("click", () => $("lb").close());
$("lbClose").onclick = () => $("lb").close();
$("saveBtn").onclick = () => {
  const a = document.createElement("a");
  a.href = shown();
  a.download = `${(file?.name || "mri").replace(/\.[^.]+$/, "")}_${view}.png`;
  a.click();
};

setState("empty");
checkHealth();
setInterval(checkHealth, 15000);
