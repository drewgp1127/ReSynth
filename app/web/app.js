import WaveSurfer from "./vendor/wavesurfer-7.8.6.esm.js";
import RegionsPlugin from "./vendor/regions-7.8.6.esm.js";

const $ = (sel) => document.querySelector(sel);
const STAGES = [
  ["research", "Gear research"],
  ["isolate", "Separate stems"],
  ["analyse", "Analyse the sound"],
  ["plan", "Plan the search"],
  ["match", "Match the patch"],
  ["report", "Write the report"],
];
const STEM_LABELS = {
  raw: ["Original selection", "no separation"],
  melodic: ["Melodic mix", "other + piano + guitar"],
  other: ["Other", "synths often land here"],
  piano: ["Piano / keys", ""],
  guitar: ["Guitar", ""],
  bass: ["Bass", ""],
  drums: ["Drums", ""],
  vocals: ["Vocals", ""],
};

let config = { region_s: [1, 3] };
let jobId = null;
let ws = null;
let region = null;
let pollTimer = null;
let stemsShownFor = null;
let resultsShown = false;
const player = new Audio();

// ---------- helpers ----------
function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c !== null && c !== undefined) n.append(c);
  return n;
}
function fileUrl(name) { return `/api/jobs/${jobId}/files/${name}`; }
function showError(msg) {
  const box = $("#error");
  box.replaceChildren(el("strong", { text: "Something went wrong. " }), document.createTextNode(msg));
  box.hidden = !msg;
  if (msg) box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function clearError() { $("#error").hidden = true; }
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error("Could not reach the ReSynth server. Is it still running?");
  }
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.detail) || `The server answered ${res.status}.`);
  return body;
}
function fmt(t) { return `${t.toFixed(2)} s`; }
function busy(btn, on, label) {
  btn.disabled = on;
  if (label) btn.textContent = label;
}

// ---------- step 1: upload ----------
const drop = $("#drop");
const fileInput = $("#file");
fileInput.addEventListener("change", () => {
  $("#drop-label").textContent = fileInput.files[0] ? fileInput.files[0].name : "Choose an audio file or drop it here";
});
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.add("over")));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  if (!fileInput.files[0]) { showError("Choose an audio file first."); return; }
  const btn = $("#upload-btn");
  busy(btn, true, "Uploading…");
  const fd = new FormData($("#upload-form"));
  try {
    const state = await api("/api/jobs", { method: "POST", body: fd });
    jobId = state.job.id;
    history.replaceState(null, "", `#job=${jobId}`);
    lockUpload(state.job);
    showRegionStep(state.job);
    render(state);
    schedulePoll();
  } catch (err) {
    showError(err.message);
  } finally {
    busy(btn, false, "Upload clip");
  }
});

function lockUpload(job) {
  for (const id of ["song", "artist", "description", "gear"]) {
    const input = $("#" + id);
    input.value = job[id] || "";
    input.disabled = true;
  }
  fileInput.disabled = true;
  $("#drop-label").textContent = job.original_name || "clip";
  $("#upload-btn").hidden = true;
}

// ---------- step 2: region ----------
function showRegionStep(job) {
  $("#step-region").hidden = false;
  if (ws) return;
  const css = getComputedStyle(document.documentElement);
  ws = WaveSurfer.create({
    container: "#waveform",
    height: 110,
    waveColor: css.getPropertyValue("--wave").trim(),
    progressColor: css.getPropertyValue("--wave-progress").trim(),
    cursorColor: css.getPropertyValue("--accent").trim(),
    barWidth: 2, barGap: 1, barRadius: 1,
    url: fileUrl("upload.wav"),
  });
  const regions = ws.registerPlugin(RegionsPlugin.create());
  ws.on("decode", () => {
    const dur = ws.getDuration();
    const [lo, hi] = config.region_s;
    let start = 0, end = Math.min(2, dur);
    if (job.region) [start, end] = job.region;
    region = regions.addRegion({
      start, end, color: css.getPropertyValue("--region").trim(), drag: true, resize: true,
      minLength: lo, maxLength: hi,
    });
    updateRegionInfo();
    region.on("update-end", () => { clampRegion(); updateRegionInfo(); });
    region.on("update", updateRegionInfo);
  });
  ws.on("error", (e) => showError(`The waveform could not be loaded (${e}).`));
  $("#play-region").onclick = () => { if (region) region.play(); };
}

function clampRegion() {
  if (!region || !ws) return;
  const [lo, hi] = config.region_s;
  const dur = ws.getDuration();
  let { start, end } = region;
  if (end - start > hi) end = start + hi;
  if (end - start < lo) end = Math.min(dur, start + lo);
  if (end - start < lo) start = Math.max(0, end - lo);
  region.setOptions({ start, end });
}

function updateRegionInfo() {
  if (!region) return;
  const len = region.end - region.start;
  $("#region-info").textContent = `${fmt(region.start)} – ${fmt(region.end)}  (${len.toFixed(1)} s)`;
}

$("#region-btn").addEventListener("click", async () => {
  if (!region) return;
  clearError();
  const btn = $("#region-btn");
  busy(btn, true, "Starting…");
  try {
    const state = await api(`/api/jobs/${jobId}/region`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: region.start, end: region.end }),
    });
    stemsShownFor = null;
    $("#step-stems").hidden = true;
    render(state);
    schedulePoll();
  } catch (err) {
    showError(err.message);
  } finally {
    busy(btn, false, "Separate stems");
  }
});

// ---------- step 3: stems ----------
function renderStems(stems, job) {
  const key = JSON.stringify(job.region);
  if (stemsShownFor === key) return;
  stemsShownFor = key;
  $("#step-stems").hidden = false;
  const help = $("#stems-help");
  help.textContent = stems.status === "raw_slice_only"
    ? `Stem separation was not available (${stems.message}). You can still use the original selection.`
    : "Listen to each version of your selection and pick the one where the synth stands out most.";
  const peak = Math.max(...Object.values(stems.rms || { x: 1 }), 1e-9);
  const box = $("#stems");
  box.replaceChildren();
  const order = ["raw", ...stems.names.filter((n) => n !== "raw")];
  for (const name of order) {
    const [label, sub] = STEM_LABELS[name] || [name, ""];
    const file = name === "raw" ? "slice.wav" : `stems/${name}.wav`;
    const id = `stem-${name}`;
    const radio = el("input", { type: "radio", name: "stem", value: name, id, "aria-label": label });
    if (name === stems.suggested) radio.checked = true;
    const play = el("button", { type: "button", class: "ghost play", "aria-label": `Play ${label}`, text: "▶" });
    play.addEventListener("click", (ev) => { ev.preventDefault(); togglePlay(fileUrl(file), play); });
    const level = el("div", { class: "level meter", title: "level in your selection" },
      el("i", { style: `width:${Math.round(100 * Math.sqrt((stems.rms[name] || 0) / peak))}%` }));
    const nameBox = el("div", { class: "name" }, label, sub ? el("small", { text: sub }) : null, level);
    const row = el("label", { class: "stem", for: id }, radio, play, nameBox,
      name === stems.suggested ? el("span", { class: "tag", text: "Suggested" }) : null);
    box.append(row);
  }
  $("#stem-btn").disabled = false;
}

function togglePlay(url, btn) {
  const same = player.src.endsWith(url);
  document.querySelectorAll(".stem .play").forEach((b) => (b.textContent = "▶"));
  if (same && !player.paused) { player.pause(); return; }
  player.src = url;
  player.play().catch(() => showError("The browser could not play this audio."));
  btn.textContent = "❚❚";
  player.onended = () => (btn.textContent = "▶");
}

$("#stem-btn").addEventListener("click", async () => {
  const chosen = document.querySelector('input[name="stem"]:checked');
  if (!chosen) { showError("Pick a stem first."); return; }
  clearError();
  player.pause();
  const btn = $("#stem-btn");
  busy(btn, true, "Starting…");
  try {
    const state = await api(`/api/jobs/${jobId}/stem`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ stem: chosen.value }),
    });
    render(state);
    schedulePoll();
  } catch (err) {
    showError(err.message);
    busy(btn, false);
  } finally {
    btn.textContent = "Find the patch";
  }
});

// ---------- progress ----------
function renderStages(state) {
  const list = $("#stages");
  list.replaceChildren();
  const st = state.stages || {};
  for (const [key, label] of STAGES) {
    const s = st[key];
    const stateName = s ? s.state : "waiting";
    const nice = { running: "running", done: "done", failed: "failed", degraded: "done (fallback)",
      skipped: "skipped", waiting: "waiting" }[stateName] || stateName;
    const li = el("li", { class: "stage", "data-state": stateName, "data-stage": key },
      el("span", { class: "dot", "aria-hidden": "true" }),
      el("div", { class: "title" }, label, el("span", { text: nice })));
    if (stateName === "running") li.append(el("div", { class: "bar" }, el("i", { style: `width:${Math.round(100 * (s.progress || 0))}%` })));
    if (s && s.message) li.append(el("div", { class: "msg", text: s.message }));
    list.append(li);
  }
}

function render(state) {
  const job = state.job;
  $("#step-progress").hidden = false;
  renderStages(state);
  if (job.region || job.phase !== "uploaded") showRegionStep(job);
  if (job.phase === "isolating") { $("#region-btn").disabled = true; }
  else { $("#region-btn").disabled = job.phase === "processing" || job.phase === "done"; }
  if (state.stems && ["choose_stem", "processing", "done", "failed"].includes(job.phase)) renderStems(state.stems, job);
  if (job.phase === "processing" || job.phase === "done") {
    $("#stem-btn").disabled = true;
    document.querySelectorAll('input[name="stem"]').forEach((r) => { r.disabled = true; if (r.value === job.stem) r.checked = true; });
  }
  if (job.phase === "failed" && job.error) showError(job.error);
  if (job.phase === "done" && !resultsShown) { resultsShown = true; loadResults(); }
}

function active(state) {
  const p = state.job.phase;
  const researchRunning = state.stages.research && state.stages.research.state === "running";
  return p === "isolating" || p === "processing" || researchRunning;
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, 1500);
}

async function poll() {
  if (!jobId) return;
  try {
    const state = await api(`/api/jobs/${jobId}`);
    render(state);
    if (active(state)) schedulePoll();
  } catch (err) {
    showError(err.message);
    schedulePoll();
  }
}

// ---------- results ----------
const CONF = { confirmed: "Confirmed", "community-sourced": "Community-sourced", speculated: "Speculated" };

function safeLink(url, text) {
  if (!/^https?:\/\//i.test(url || "")) return null;
  return el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: text || "source" });
}

function section(title, ...nodes) {
  return [el("h3", { text: title }), ...nodes];
}

function gearReport(r) {
  const out = [];
  const res = r.research;
  if (!res || res.status !== "done") {
    out.push(el("p", { class: "skipped", text: `Gear research was skipped: ${(res && res.reason) || "not run"}.` }));
    return section("Gear report", ...out);
  }
  out.push(el("p", { text: res.model_statement || "" }));
  const ul = el("ul");
  for (const i of res.instruments || []) {
    ul.append(el("li", {}, el("span", { class: `conf ${i.confidence}`, text: CONF[i.confidence] || i.confidence }),
      `${i.name} (model: ${i.model}). ${i.evidence || ""} `, safeLink(i.source_url),
      i.url_verified === false ? " (link not verified)" : null));
  }
  if (res.synthesis && res.synthesis.type) {
    const s = res.synthesis;
    ul.append(el("li", {}, el("span", { class: `conf ${s.confidence}`, text: CONF[s.confidence] || "" }),
      `Synthesis: ${s.type}. ${s.details || ""}`));
  }
  for (const q of res.quotes || []) {
    ul.append(el("li", {}, `“${q.quote}” — ${q.speaker || "unknown"} `, safeLink(q.source_url)));
  }
  for (const rec of res.recreations || []) {
    ul.append(el("li", {}, el("span", { class: "conf recreation", text: "Recreation, not the original gear" }),
      `${rec.title || ""}${rec.author ? " by " + rec.author : ""}: ${rec.settings || ""} `, safeLink(rec.url)));
  }
  out.push(ul);
  return section("Gear report", ...out);
}

function paragraphs(text) {
  return (text || "").split(/\n{1,}/).filter((t) => t.trim()).map((t) => el("p", { text: t }));
}

function renderReport(r) {
  const box = $("#report");
  const nodes = [];
  nodes.push(...gearReport(r));
  const rec = [];
  for (const s of r.recipe) {
    rec.push(el("p", {}, el("strong", { text: s.title })));
    rec.push(el("ul", {}, ...s.lines.map((l) => el("li", { text: l }))));
  }
  nodes.push(...section(`Recipe (template: ${r.template})`, ...rec));
  const gear = paragraphs(r.on_your_gear);
  if (r.prose_source !== "claude") gear.push(el("p", { class: "skipped", text: `Written without Claude: ${r.prose_skipped_reason}.` }));
  nodes.push(...section("On your gear", ...gear));
  nodes.push(...section("Recommended effects", el("ul", {}, ...r.effects.map((e) =>
    el("li", {}, el("strong", { text: e.effect }), `: ${e.settings} (${e.why})`)))));
  if (r.notes && r.notes.length) nodes.push(...section("Tips", el("ul", {}, ...r.notes.map((n) => el("li", { text: n })))));
  nodes.push(...section("Caveats", el("ul", {}, ...r.caveats.map((c) => el("li", { text: c })))));
  box.replaceChildren(...nodes);
}

async function loadResults() {
  try {
    const r = await api(`/api/jobs/${jobId}/report`);
    $("#step-results").hidden = false;
    $("#error").after($("#step-results"));  // results first; the steps that led here follow below
    const label = r.similarity.label;
    const badge = $("#sim-label");
    badge.textContent = `Similarity: ${label}`;
    badge.className = `badge ${label.replace(" ", "-")}`;
    $("#sim-text").textContent = r.similarity.explanation;
    $("#ab-audio").src = fileUrl("ab.wav");
    document.querySelectorAll("[data-play]").forEach((b) => (b.onclick = () => { player.src = fileUrl(b.dataset.play); player.play(); }));
    const dl = $("#downloads");
    dl.replaceChildren(
      el("a", { class: "button primary", href: fileUrl("patch.vital"), download: "resynth_patch.vital", text: "Download patch.vital" }),
      ...(r.runner_ups || []).filter((u) => u.file).map((u, i) =>
        el("a", { class: "button ghost", href: fileUrl(u.file), download: `resynth_${u.file}`, text: `Runner-up ${i + 1}` })),
      el("a", { class: "button ghost", href: fileUrl("report.md"), download: "resynth_report.md", text: "Report (.md)" }),
      el("a", { class: "button ghost", href: fileUrl("report.html"), target: "_blank", rel: "noopener", text: "Printable report" }),
    );
    renderReport(r);
    $("#step-results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message);
  }
}

// ---------- start ----------
async function start() {
  try {
    config = await api("/api/config");
    if (!config.key_present) {
      const n = $("#notice");
      n.textContent = `Gear research is off: the ${config.key_env} setting is not set. You still get a preset, A/B audio and a rule-based report.`;
      n.hidden = false;
    }
  } catch (err) {
    showError(err.message);
  }
  const m = location.hash.match(/job=([\w-]+)/);
  if (m) {
    jobId = m[1];
    try {
      const state = await api(`/api/jobs/${jobId}`);
      lockUpload(state.job);
      render(state);
      if (active(state)) schedulePoll();
    } catch (err) {
      jobId = null;
      history.replaceState(null, "", location.pathname);
      showError(`That job could not be loaded (${err.message}).`);
    }
  }
}
start();
