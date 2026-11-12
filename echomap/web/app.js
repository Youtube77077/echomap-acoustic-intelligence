/* EchoMap UI: vanilla JS, no build step. Talks to the local token-protected API. */
"use strict";
const TOKEN = document.querySelector('meta[name="echomap-token"]').content;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtT = t => { if (t == null || isNaN(t)) return "--:--.---"; const m = Math.floor(t / 60), s = t - m * 60; return `${String(m).padStart(2,"0")}:${s.toFixed(3).padStart(6,"0")}`; };

async function api(path, opts = {}) {
  const r = await fetch("/api/" + path, {method: opts.method || "GET", headers: {"X-EchoMap-Token": TOKEN, "Content-Type": "application/json"},
    body: opts.body ? JSON.stringify(opts.body) : undefined});
  if (opts.raw) return r;
  const j = await r.json().catch(() => ({error: "bad response"}));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}

// ---------------------------------------------------------------- state
const S = {duration: 0, sr: 0, inventory: [], result: null, view: {t0: 0, t1: 10, fmin: 0, fmax: 16000}, sel: null, poll: null,
  catColors: {}, specImg: null, specMeta: null, specReq: 0};
const CAT_COLORS = ["#ffb340","#4f8cff","#ff6b6b","#2ed3b7","#c77dff","#f5e13c","#ff8fab","#7bd88f","#90caf9","#4dd0e1",
  "#00bcd4","#bcaaa4","#ffa726","#ef5350","#ab47bc","#66bb6a","#8d99ae","#9e9e9e"];
const PALETTE = [[0,[10,4,30]],[.18,[68,20,120]],[.38,[40,80,200]],[.6,[245,225,60]],[.8,[245,140,40]],[1,[215,25,28]]];
const LUT = (() => { const l = new Uint8ClampedArray(256 * 3); for (let i = 0; i < 256; i++) { const t = i / 255;
  for (let k = 0; k < PALETTE.length - 1; k++) { const [t0, c0] = PALETTE[k], [t1, c1] = PALETTE[k + 1];
    if (t >= t0 && t <= t1) { const f = (t - t0) / (t1 - t0); for (let c = 0; c < 3; c++) l[i * 3 + c] = c0[c] + f * (c1[c] - c0[c]); break; } } } return l; })();

// ---------------------------------------------------------------- tabs
$$(".tab").forEach(b => b.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".tabpane").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "geo") Globe.resize();
  if (b.dataset.tab === "analysis") { drawWave(); requestSpec(); }
});

// ---------------------------------------------------------------- import / analyse
$("#btnOpen").onclick = async () => {
  try {
    const r = await api("open", {method: "POST", body: {}});
    if (r.cancelled) return;
    const m = r.meta, a = m.primary_audio;
    S.duration = m.duration_s || 0;
    $("#fileInfo").innerHTML = `<b>${esc(m.file_name)}</b> · ${esc(m.container)} · ${esc(a.codec)} ${a.sample_rate} Hz ${a.channels} ch · ${fmtT(S.duration)} · SHA-256 <span class="mono">${r.sha256.slice(0,16)}…</span>`;
    const miss = Object.entries(r.models).filter(([k, v]) => !v.installed).map(([k, v]) => v.name);
    $("#prepPanel").classList.remove("hidden");
    $("#prepPanel").innerHTML = `<b>Ready to analyse.</b> The original file is opened read-only and its hash recorded. ${m.is_video ? "The audio track will be extracted by stream copy (no re-encoding)." : ""}
      ${miss.length ? `<div class="no">Missing models (those analyses will be skipped): ${miss.map(esc).join("; ")}. Use “Models & settings” to point EchoMap at the models folder.</div>` : '<span class="ok">All models available.</span>'}
      <div class="muted small">Embedded times: ${esc(JSON.stringify(m.embedded_times))} · device: ${esc(JSON.stringify(m.device))} · file-system times are not recording times.</div>`;
    $("#btnAnalyse").disabled = false; $("#btnContext").disabled = true;
    resetResultViews();
  } catch (e) { alert("Could not open file: " + e.message); }
};
$("#btnAnalyse").onclick = async () => {
  $("#btnAnalyse").disabled = true; $("#btnOpen").disabled = true;
  try { await api("analyse", {method: "POST", body: {}}); startPolling("analysed"); } catch (e) { alert(e.message); $("#btnOpen").disabled = false; }
};
function startPolling(target) {
  $("#progressWrap").classList.remove("hidden");
  clearInterval(S.poll);
  S.poll = setInterval(async () => {
    const s = await api("status").catch(() => null); if (!s) return;
    if (s.progress) { $("#progressBar").style.width = (s.progress.fraction * 100) + "%"; $("#progressText").textContent = `${s.progress.stage}: ${s.progress.message}`; }
    if (s.state === "error") { clearInterval(S.poll); $("#progressText").textContent = "Error: " + s.error; $("#btnOpen").disabled = false; }
    if (s.state === target || (target === "analysed" && s.state === "complete")) {
      clearInterval(S.poll); $("#btnOpen").disabled = false; await loadResult();
      if (target === "analysed") openContext(); else { $("#progressWrap").classList.add("hidden"); }
    }
  }, 600);
}
async function loadResult() {
  const r = await api("result");
  S.result = r.result; S.inventory = r.result.inventory || []; S.meta = r.meta;
  if (r.meta) $("#fileInfo").innerHTML = `<b>${esc(r.meta.file_name)}</b> · ${esc(r.meta.container)} · ${esc(r.meta.primary_audio?.codec)} ${r.meta.primary_audio?.sample_rate} Hz · SHA-256 <span class="mono">${esc((r.sha256 || "").slice(0, 16))}…</span>`;
  S.duration = r.result.analysis_copy?.duration_s || S.duration; S.sr = r.result.analysis_copy?.sample_rate || 32000;
  $("#audio").src = "/audio?t=" + encodeURIComponent(TOKEN) + "&n=" + Date.now();
  $("#btnPlay").disabled = false; $("#btnContext").disabled = false;
  $("#timeDur").textContent = fmtT(S.duration);
  S.view = {t0: 0, t1: S.duration, fmin: 0, fmax: Math.min(S.sr / 2, 16000)};
  $("#fMin").value = 0; $("#fMax").value = S.view.fmax;
  assignColors(); fillCatFilters(); renderTimeline(); renderInventory(); drawWave(); requestSpec();
  $("#invCount").textContent = S.inventory.length;
  if (r.result.geo) { Geo.render(r.result.geo); }
  if (r.result.report_html) loadReport();
}
function resetResultViews() {
  S.inventory = []; S.result = null; renderTimeline(); renderInventory(); $("#invCount").textContent = "0";
  $("#reportFrame").srcdoc = ""; ["#expHtml", "#expPdf", "#expJson"].forEach(s => $(s).disabled = true);
  Geo.render(null);
}
function assignColors() { const cats = [...new Set(S.inventory.map(d => d.category))]; cats.forEach((c, i) => S.catColors[c] = S.catColors[c] || CAT_COLORS[Object.keys(S.catColors).length % CAT_COLORS.length]); }
const catColor = c => S.catColors[c] || "#999";

// ---------------------------------------------------------------- context questionnaire
const tzSel = $("#tz"); (Intl.supportedValuesOf ? Intl.supportedValuesOf("timeZone") : []).forEach(z => tzSel.add(new Option(z, z)));
api("countries").then(cs => cs.forEach(c => $("#lCountry").add(new Option(`${c.name} (${c.iso2})`, c.iso2)))).catch(() => {});
let cityHits = [];
$("#lCity").addEventListener("input", async e => {
  const q = e.target.value; if (q.length < 2) return;
  cityHits = await api("cities?q=" + encodeURIComponent(q) + "&cc=" + ($("#lCountry").value || "")).catch(() => []);
  $("#cityList").innerHTML = cityHits.map(c => `<option value="${esc(c.name)}, ${esc(c.admin1)}, ${esc(c.country)}">`).join("");
});
function openContext() { $("#ctxDlg").showModal(); }
$("#btnContext").onclick = openContext;
$("#ctxDlg").addEventListener("close", async () => {
  if ($("#ctxDlg").returnValue !== "run") return;
  const dmode = $('input[name="dmode"]:checked').value, lmode = $('input[name="lmode"]:checked').value;
  const date = {mode: dmode};
  if (dmode === "exact") date.value = $("#dExact").value; if (dmode === "approx_date") date.value = $("#dApprox").value;
  if (dmode === "month_year") { date.month = +$("#dMonth").value; date.year = +$("#dYear").value; }
  const loc = {mode: lmode, country: $("#lCountry").value || null};
  if (lmode === "region") loc.region = $("#lRegion").value;
  if (lmode === "city") {
    const v = $("#lCity").value; const hit = cityHits.find(c => v.startsWith(c.name)) || cityHits[0];
    if (!hit) { alert("City not found in the offline gazetteer. Choose a country or region instead."); return; }
    loc.city = hit; loc.country = hit.country;
  }
  if ((lmode === "country" || lmode === "region") && !loc.country) { alert("Please select a country."); return; }
  try { await api("context", {method: "POST", body: {context: {date, timezone: $("#tz").value || null, location: loc}}}); startPolling("complete"); }
  catch (e) { alert(e.message); }
});

// ---------------------------------------------------------------- audio
const audio = $("#audio");
$("#btnPlay").onclick = () => audio.paused ? audio.play() : audio.pause();
audio.onplay = () => { $("#btnPlay").textContent = "❚❚"; tick(); };
audio.onpause = () => { $("#btnPlay").textContent = "▶"; };
function tick() {
  $("#timeNow").textContent = fmtT(audio.currentTime);
  if (S.sel && $("#chkLoop").checked && audio.currentTime >= S.sel.end) audio.currentTime = S.sel.start;
  if (S.sel && S.playToEnd && audio.currentTime >= S.playToEnd) { audio.pause(); S.playToEnd = null; }
  // keep cursor visible
  if (!audio.paused && (audio.currentTime > S.view.t1 || audio.currentTime < S.view.t0)) {
    const w = S.view.t1 - S.view.t0; S.view.t0 = Math.max(0, audio.currentTime - w * 0.1); S.view.t1 = Math.min(S.duration, S.view.t0 + w); requestSpec();
  }
  drawOverlay(); drawWave();
  if (!audio.paused) requestAnimationFrame(tick);
}
function playSegment(d) { S.sel = d; audio.currentTime = d.start; S.playToEnd = d.end; audio.play(); }

// ---------------------------------------------------------------- waveform
function drawWave() {
  const c = $("#wave"), ctx = c.getContext("2d"); c.width = c.clientWidth * devicePixelRatio; c.height = 70 * devicePixelRatio;
  ctx.clearRect(0, 0, c.width, c.height);
  const w = S.result?.signal?.waveform; if (!w) return;
  const n = w.min.length, H = c.height, mid = H / 2;
  const peak = Math.max(1e-4, ...w.max.map(Math.abs), ...w.min.map(Math.abs)), g = 0.95 / peak;  // auto-scale to peak
  ctx.fillStyle = "#3d6fd8";
  for (let x = 0; x < c.width; x++) { const i = Math.floor(x / c.width * n); ctx.fillRect(x, mid - w.max[i] * g * mid, 1, Math.max(1, (w.max[i] - w.min[i]) * g * mid)); }
  ctx.fillStyle = "rgba(24,40,63,.6)"; ctx.font = `${9 * devicePixelRatio}px Consolas`; ctx.fillText(`peak ${(20 * Math.log10(peak)).toFixed(1)} dBFS (display normalised)`, 4, 10 * devicePixelRatio);
  const x0 = S.view.t0 / S.duration * c.width, x1 = S.view.t1 / S.duration * c.width;
  ctx.fillStyle = "rgba(31,102,229,.12)"; ctx.fillRect(x0, 0, x1 - x0, H); ctx.strokeStyle = "#1f66e5"; ctx.strokeRect(x0, 0.5, x1 - x0, H - 1);
  ctx.fillStyle = "#0f3d99"; ctx.fillRect(audio.currentTime / S.duration * c.width, 0, 2, H);
}
$("#wave").onclick = e => { if (!S.duration) return; const t = e.offsetX / e.target.clientWidth * S.duration; const w = S.view.t1 - S.view.t0;
  S.view.t0 = Math.max(0, Math.min(S.duration - w, t - w / 2)); S.view.t1 = S.view.t0 + w; audio.currentTime = t; requestSpec(); drawWave(); };

// ---------------------------------------------------------------- spectrogram
const spec = $("#spec");
function legend() { const g = $("#paletteLegend"); g.style.background = `linear-gradient(90deg, ${PALETTE.map(([t, c]) => `rgb(${c}) ${t * 100}%`).join(",")})`;
  g.title = `${$("#dbMin").value} dBFS … ${$("#dbMax").value} dBFS`; }
legend();
let specTimer = null;
function requestSpec() { clearTimeout(specTimer); specTimer = setTimeout(fetchSpec, 60); }
async function fetchSpec() {
  if (!S.result) return;
  spec.width = spec.clientWidth * devicePixelRatio; spec.height = 330 * devicePixelRatio;
  const v = S.view, id = ++S.specReq;
  const q = new URLSearchParams({t0: v.t0, t1: v.t1, fmin: v.fmin, fmax: v.fmax, scale: $("#selScale").value,
    w: Math.min(2400, spec.width), h: Math.min(800, spec.height), dbmin: $("#dbMin").value, dbmax: $("#dbMax").value});
  const r = await api("spectrogram?" + q, {raw: true}); if (id !== S.specReq || !r.ok) return;
  const meta = JSON.parse(r.headers.get("X-Meta")); const buf = new Uint8Array(await r.arrayBuffer());
  const img = new ImageData(meta.width, meta.height);
  for (let i = 0; i < buf.length; i++) { const k = buf[i] * 3; img.data[i * 4] = LUT[k]; img.data[i * 4 + 1] = LUT[k + 1]; img.data[i * 4 + 2] = LUT[k + 2]; img.data[i * 4 + 3] = 255; }
  S.specImg = await createImageBitmap(img); S.specMeta = meta; S.specBuf = buf; drawOverlay();
}
function f2y(f, m, H) { let r; if (m.scale === "log") r = Math.log(f / m.fmin) / Math.log(m.fmax / m.fmin);
  else if (m.scale === "mel") { const mel = x => 2595 * Math.log10(1 + x / 700); r = (mel(f) - mel(m.fmin)) / (mel(m.fmax) - mel(m.fmin)); }
  else r = (f - m.fmin) / (m.fmax - m.fmin); return H - r * H; }
function y2f(y, m, H) { const r = 1 - y / H; if (m.scale === "log") return m.fmin * Math.pow(m.fmax / m.fmin, r);
  if (m.scale === "mel") { const mel = x => 2595 * Math.log10(1 + x / 700), imel = z => 700 * (Math.pow(10, z / 2595) - 1); return imel(mel(m.fmin) + r * (mel(m.fmax) - mel(m.fmin))); }
  return m.fmin + r * (m.fmax - m.fmin); }
function drawOverlay() {
  const ctx = spec.getContext("2d"), W = spec.width, H = spec.height, m = S.specMeta; ctx.clearRect(0, 0, W, H);
  if (!S.specImg || !m) return;
  ctx.imageSmoothingEnabled = false; ctx.drawImage(S.specImg, 0, 0, W, H);
  const t2x = t => (t - m.t0) / (m.t1 - m.t0) * W;
  // axes
  ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.font = `${10 * devicePixelRatio}px Consolas`;
  for (let i = 0; i <= 5; i++) { const y = i / 5 * H; const f = y2f(y, m, H); ctx.fillText(f >= 1000 ? (f / 1000).toFixed(1) + "k" : f.toFixed(0), 3, Math.max(10, y - 2)); }
  const span = m.t1 - m.t0, step = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60].find(s => span / s <= 10) || 60;
  for (let t = Math.ceil(m.t0 / step) * step; t <= m.t1; t += step) { const x = t2x(t); ctx.fillRect(x, H - 6, 1, 6); ctx.fillText(fmtT(t).slice(0, 8), x + 2, H - 8); }
  if ($("#chkOverlay").checked) {
    for (const d of filteredTimeline()) {
      if (d.end < m.t0 || d.start > m.t1 || d.duration > S.duration * 0.9) continue;
      const x0 = t2x(d.start), x1 = t2x(d.end); const fb = d.frequency?.band_5_95_hz;
      const y0 = fb ? f2y(Math.min(m.fmax, fb[1]), m, H) : 0, y1 = fb ? f2y(Math.max(m.fmin, fb[0] || m.fmin + 1), m, H) : H;
      ctx.strokeStyle = catColor(d.category); ctx.lineWidth = (S.sel && S.sel.id === d.id ? 3 : 1.2) * devicePixelRatio;
      ctx.strokeRect(x0, Math.max(0, y0), Math.max(2, x1 - x0), Math.max(4, Math.min(H, y1) - Math.max(0, y0)));
      ctx.fillStyle = catColor(d.category); ctx.fillText(d.id, x0 + 2, Math.max(12, y0 + 11));
    }
  }
  if (S.sel) { ctx.fillStyle = "rgba(255,255,255,.07)"; ctx.fillRect(t2x(S.sel.start), 0, t2x(S.sel.end) - t2x(S.sel.start), H); }
  ctx.fillStyle = "#fff"; ctx.fillRect(t2x(audio.currentTime), 0, 1.5 * devicePixelRatio, H);
}
spec.addEventListener("wheel", e => { e.preventDefault(); if (!S.specMeta) return; const z = e.deltaY > 0 ? 1.25 : 0.8;
  if (e.shiftKey) { const f = y2f(e.offsetY * devicePixelRatio, S.specMeta, spec.height); const lo = Math.max(0, f - (f - S.view.fmin) * z), hi = Math.min(S.sr / 2, f + (S.view.fmax - f) * z);
    if (hi - lo > 50) { S.view.fmin = lo; S.view.fmax = hi; $("#fMin").value = lo.toFixed(0); $("#fMax").value = hi.toFixed(0); } }
  else { const t = S.view.t0 + e.offsetX / spec.clientWidth * (S.view.t1 - S.view.t0); let a = t - (t - S.view.t0) * z, b = t + (S.view.t1 - t) * z;
    a = Math.max(0, a); b = Math.min(S.duration, b); if (b - a >= 0.05) { S.view.t0 = a; S.view.t1 = b; } }
  requestSpec(); drawWave(); }, {passive: false});
let drag = null;
spec.onmousedown = e => drag = {x: e.offsetX, y: e.offsetY, v: {...S.view}, moved: false};
window.addEventListener("mouseup", e => { if (drag && !drag.moved && e.target === spec && S.specMeta) { audio.currentTime = S.view.t0 + drag.x / spec.clientWidth * (S.view.t1 - S.view.t0); drawOverlay(); drawWave(); } drag = null; });
spec.onmousemove = e => {
  const m = S.specMeta; if (!m) return;
  if (drag) { const dx = e.offsetX - drag.x; if (Math.abs(dx) > 3) drag.moved = true; const dt = -dx / spec.clientWidth * (drag.v.t1 - drag.v.t0);
    const w = drag.v.t1 - drag.v.t0; S.view.t0 = Math.max(0, Math.min(S.duration - w, drag.v.t0 + dt)); S.view.t1 = S.view.t0 + w; requestSpec(); drawWave(); return; }
  const t = m.t0 + e.offsetX / spec.clientWidth * (m.t1 - m.t0), f = y2f(e.offsetY * devicePixelRatio, m, spec.height);
  const px = Math.floor(e.offsetX / spec.clientWidth * m.width), py = Math.floor(e.offsetY / spec.clientHeight * m.height);
  const db = S.specBuf ? m.db_min + S.specBuf[py * m.width + px] / 255 * (m.db_max - m.db_min) : NaN;
  const evs = S.inventory.filter(d => d.start <= t && d.end >= t && d.duration < S.duration * 0.9).slice(0, 4).map(d => `${d.id} ${d.label}`);
  const h = $("#hover"); h.classList.remove("hidden"); h.style.left = (e.offsetX + 14) + "px"; h.style.top = (e.offsetY + 10) + "px";
  h.innerHTML = `t ${fmtT(t)} · f ${f.toFixed(0)} Hz · ${isNaN(db) ? "" : db.toFixed(1) + " dBFS"}<br>Δt ${(m.time_resolution_s * 1000).toFixed(1)} ms · Δf ${m.freq_resolution_hz.toFixed(1)} Hz${evs.length ? "<br>" + evs.map(esc).join("<br>") : ""}`;
};
spec.onmouseleave = () => $("#hover").classList.add("hidden");
spec.ondblclick = e => { const m = S.specMeta; if (!m) return; const t = m.t0 + e.offsetX / spec.clientWidth * (m.t1 - m.t0);
  const d = S.inventory.filter(d => d.start <= t && d.end >= t).sort((a, b) => a.duration - b.duration)[0]; if (d) selectEvent(d, false); };
["#selScale", "#dbMin", "#dbMax"].forEach(s => $(s).addEventListener("input", () => { legend(); requestSpec(); }));
$("#fMin").onchange = $("#fMax").onchange = () => { S.view.fmin = Math.max(0, +$("#fMin").value); S.view.fmax = Math.min(S.sr / 2, +$("#fMax").value); requestSpec(); };
$("#btnZoomFit").onclick = () => { S.view = {t0: 0, t1: S.duration, fmin: 0, fmax: Math.min(S.sr / 2, 16000)}; $("#fMin").value = 0; $("#fMax").value = S.view.fmax; requestSpec(); drawWave(); };
$("#chkOverlay").onchange = drawOverlay;
window.addEventListener("resize", () => { drawWave(); requestSpec(); Globe.resize(); });

function selectEvent(d, zoom = true) {
  S.sel = d; audio.currentTime = d.start;
  if (zoom) { const pad = Math.max(0.5, d.duration * 0.3); S.view.t0 = Math.max(0, d.start - pad); S.view.t1 = Math.min(S.duration, d.end + pad);
    const fb = d.frequency?.band_5_95_hz; if (fb && $("#selScale").value === "linear") { S.view.fmin = 0; S.view.fmax = Math.min(S.sr / 2, Math.max(2000, fb[1] * 1.5)); $("#fMax").value = S.view.fmax.toFixed(0); $("#fMin").value = 0; } }
  $$(".tl-row").forEach(r => r.classList.toggle("sel", r.dataset.id === d.id)); requestSpec(); drawWave();
}

// ---------------------------------------------------------------- timeline
function fillCatFilters() { const sel = $("#tlCat"); sel.length = 1; [...new Set(S.inventory.map(d => d.category))].forEach(c => { const d = S.inventory.find(x => x.category === c); sel.add(new Option(d.category_title, c)); }); }
function filteredTimeline() {
  const q = $("#tlSearch").value.toLowerCase(), cat = $("#tlCat").value, tier = $("#tlTier").value, sort = $("#tlSort").value;
  let l = S.inventory.filter(d => (!cat || d.category === cat) && (!tier || d.tier === tier) && (!q || JSON.stringify([d.id, d.label, d.specific, d.category_title]).toLowerCase().includes(q)));
  const key = {start: d => d.start, confidence: d => -d.confidence, category: d => d.category, duration: d => -d.duration}[sort];
  return l.sort((a, b) => key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : a.start - b.start);
}
function renderTimeline() {
  const l = filteredTimeline();
  $("#timeline").innerHTML = `<div class="tl-row muted" style="cursor:default"><span>Time</span><span></span><span>Category</span><span>Detection</span><span>Conf.</span><span>Tier</span></div>` +
    (l.length ? `<div class="tl-row" style="cursor:default"><span class="mono">${fmtT(0)}</span><span></span><span class="muted">-</span><span>Recording begins</span><span></span><span></span></div>` : "") +
    l.map(d => `<div class="tl-row" data-id="${d.id}"><span class="mono">${fmtT(d.start)}–${fmtT(d.end).slice(3)}</span><span class="dot" style="background:${catColor(d.category)}"></span>
      <span>${esc(d.category_title)}</span><span>${esc(d.id)} · ${esc(d.label)}${d.specific ? ` <span class="muted">· ${esc(d.specific)}</span>` : ""}</span>
      <span><div class="confbar" title="${d.confidence.toFixed(2)}"><i style="width:${d.confidence * 100}%"></i></div></span><span class="tier ${d.tier}">${d.tier}</span></div>`).join("");
  $$("#timeline .tl-row[data-id]").forEach(r => r.onclick = () => selectEvent(S.inventory.find(d => d.id === r.dataset.id)));
  drawOverlay();
}
["#tlSearch", "#tlCat", "#tlTier", "#tlSort"].forEach(s => $(s).addEventListener("input", renderTimeline));

// ---------------------------------------------------------------- inventory
function kv(obj) { return `<div class="kv">${Object.entries(obj).filter(([k, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `<div>${esc(k)}</div><div>${typeof v === "object" ? esc(JSON.stringify(v)) : esc(v)}</div>`).join("")}</div>`; }
function renderInventory() {
  const q = $("#invSearch").value.toLowerCase(), tiers = new Set($$(".tierf:checked").map(c => c.value)), sort = $("#invSort").value;
  const key = {start: d => d.start, category: d => d.category, confidence: d => -d.confidence, duration: d => -d.duration}[sort];
  const list = S.inventory.filter(d => tiers.has(d.tier) && (!q || JSON.stringify(d).toLowerCase().includes(q))).sort((a, b) => key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : a.start - b.start);
  const summ = S.result?.inventory_summary || {};
  $("#invSummary").innerHTML = Object.entries(summ).map(([k, v]) => `<span class="chip" data-cat="${k}" style="border-color:${v.count ? catColor(k) : "var(--line)"}">${esc(v.title)} <b>${v.count}</b></span>`).join("");
  $$("#invSummary .chip").forEach(c => c.onclick = () => { const el = $(`#cat-${c.dataset.cat}`); if (el) { el.open = true; el.scrollIntoView({behavior: "smooth"}); } });
  const cats = {}; list.forEach(d => (cats[d.category] = cats[d.category] || []).push(d));
  const order = Object.keys(summ).length ? Object.keys(summ) : Object.keys(cats);
  $("#inventory").innerHTML = order.filter(k => cats[k]).map(k => `<details class="cat" id="cat-${k}" open><summary><span class="dot" style="background:${catColor(k)}"></span>${esc(summ[k]?.title || k)}
      <span class="count">${cats[k].length} detection(s) · ${["confirmed","probable","possible","unclassified"].map(t => `${cats[k].filter(d => d.tier === t).length} ${t}`).join(" · ")}</span></summary>
    ${cats[k].map(d => `<details class="det" data-id="${d.id}"><summary><span class="mono">${d.id}</span><span><b>${esc(d.label)}</b>${d.specific ? `<br><span class="muted">${esc(d.specific)}</span>` : ""}</span>
      <span class="mono">${fmtT(d.start)} → ${fmtT(d.end)}</span><span>${d.duration.toFixed(2)} s · ${d.confidence.toFixed(2)}<div class="confbar"><i style="width:${d.confidence * 100}%"></i></div></span>
      <span class="tier ${d.tier}">${d.tier}</span><span class="muted small">${esc((d.model || "").split(" (")[0])}</span><button class="round play" title="Play this segment">▶</button></summary>
      <div class="body">
        ${kv({"Detection ID": d.id, "Category": d.category_title, "Identified": d.label, "More specific": d.specific, "Common name": d.common_name, "Scientific name": d.scientific_name,
          "Taxon": d.taxon_class && `${d.taxon_class} / ${d.taxon_order}`, "Call or song": d.call_or_song, "Language": d.language_name && `${d.language_name} (${d.language_code})`,
          "Overlapping languages": d.overlapping_languages, "Start": fmtT(d.start), "End": fmtT(d.end), "Duration": d.duration.toFixed(3) + " s", "Confidence (model score)": d.confidence.toFixed(4), "Tier": d.tier, "Kind": d.kind})}
        ${kv({"Frequency": d.frequency && Object.keys(d.frequency).length ? d.frequency : "-", "Acoustic signature": d.signature, "Detection model": d.model,
          "Corroborated by": d.corroborated_by?.length ? d.corroborated_by.join("; ") : "-", "Directly measured": d.directly_measured, "Identification detail": d.identification_detail,
          "Geographic relevance": d.geographic_relevance, "Temporal relevance": d.temporal_relevance, "Limitations": d.limitations})}
        <div class="full"><b>Alternative classifications</b>: ${d.candidates?.length ? d.candidates.map(c => `${esc(c.label)}${c.score != null ? " (" + (+c.score).toFixed(2) + ")" : ""}`).join(" · ") : "none"}</div>
        <div class="full"><b>Evidence (raw model outputs and measurements)</b><pre class="ev">${esc(JSON.stringify(d.evidence, null, 1))}</pre></div>
        <div class="full"><button class="ghost show">Show in spectrogram</button></div>
      </div></details>`).join("")}</details>`).join("") || `<div class="muted">No detections${S.inventory.length ? " match the filters" : " yet"}.</div>`;
  $$("#inventory .det").forEach(el => { const d = S.inventory.find(x => x.id === el.dataset.id);
    el.querySelector(".play").onclick = ev => { ev.preventDefault(); playSegment(d); };
    el.querySelector(".show").onclick = () => { $('.tab[data-tab="analysis"]').click(); selectEvent(d); }; });
}
["#invSearch", "#invSort"].forEach(s => $(s).addEventListener("input", renderInventory)); $$(".tierf").forEach(c => c.onchange = renderInventory);
$("#invExpand").onclick = () => { const open = $("#invExpand").textContent === "Expand all"; $$("#inventory details").forEach(d => d.open = open); $("#invExpand").textContent = open ? "Collapse all" : "Expand all"; };

// ---------------------------------------------------------------- globe
const Globe = (() => {
  const cv = $("#globe"); let world = null, proj = null, mode3d = true, rot = [-10, -20, 0], scale = null, geo = null, hover = null;
  d3.json("/static/vendor/countries-50m.json").then(t => { world = topojson.feature(t, t.objects.countries); resize(); });
  function makeProj() { const W = cv.width, H = cv.height;
    proj = mode3d ? d3.geoOrthographic().rotate(rot).clipAngle(90) : d3.geoEquirectangular().rotate([rot[0], 0]);
    proj.translate(mode3d ? [W / 2, H / 2] : [W / 2 + (panX || 0), H / 2 + (panY || 0)]).scale(scale || (mode3d ? Math.min(W, H) * 0.46 : W / 6.4)); }
  let panX = 0, panY = 0;
  function resize() { cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio; makeProj(); draw(); }
  function draw() {
    if (!proj) return; const ctx = cv.getContext("2d"), path = d3.geoPath(proj, ctx); ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.beginPath(); path({type: "Sphere"}); ctx.fillStyle = "#dfe8f1"; ctx.fill();
    if ($("#lyrGrat").checked) { ctx.beginPath(); path(d3.geoGraticule10()); ctx.strokeStyle = "rgba(80,110,160,.18)"; ctx.lineWidth = 1; ctx.stroke(); }
    if (world) { ctx.beginPath(); path(world); ctx.fillStyle = "#f7f8fa"; ctx.fill();
      if ($("#lyrBorders").checked) { ctx.strokeStyle = "rgba(120,135,160,.55)"; ctx.lineWidth = 0.6 * devicePixelRatio; ctx.stroke(); } }
    if (geo && $("#lyrHeat").checked && geo.heat) { const c = geo.heat_cell_deg / 2;
      for (const [la, lo, v] of geo.heat) { const k = Math.round(70 + 185 * v) * 3; ctx.beginPath();
        path({type: "Polygon", coordinates: [[[lo - c, la - c], [lo + c, la - c], [lo + c, la + c], [lo - c, la + c], [lo - c, la - c]]]});
        ctx.fillStyle = `rgba(${LUT[k]},${LUT[k + 1]},${LUT[k + 2]},${0.25 + 0.6 * v})`; ctx.fill(); } }
    const cands = geo ? [...(geo.candidates || [])] : [];
    if (geo && $("#lyrCand").checked) {
      const lr = geo.local_refinement?.candidates || [];
      [...cands].reverse().concat(lr.map(c => ({...c, _local: true}))).forEach(c => {
        const circ = d3.geoCircle().center([c.centre.lon, c.centre.lat]).radius(c.uncertainty_radius_km / 111.195)();
        const col = c._local ? "#39c27a" : c.rank === 1 ? "#ff5a3c" : c.strength === "weaker" ? "#a0aec0" : "#ffb340";
        ctx.beginPath(); path(circ); ctx.strokeStyle = col; ctx.lineWidth = (c.rank === 1 ? 2.5 : 1.5) * devicePixelRatio; ctx.setLineDash(c.strength === "weaker" ? [6, 4] : []); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = col + "22"; ctx.fill();
        const p = proj([c.centre.lon, c.centre.lat]); if (p && visible(c.centre)) { ctx.beginPath(); ctx.arc(p[0], p[1], 4 * devicePixelRatio, 0, 7); ctx.fillStyle = col; ctx.fill();
          ctx.font = `${11 * devicePixelRatio}px Segoe UI`; ctx.fillStyle = "#18283f"; ctx.fillText(c._local ? `L${c.rank}` : `#${c.rank}`, p[0] + 6, p[1] - 6); } });
    }
    if (geo && $("#lyrEvid").checked) {
      const ctxLoc = S.result?.context?.location; if (ctxLoc?.city) marker(ctx, ctxLoc.city.lon, ctxLoc.city.lat, "#2ed3b7", "user city");
      const em = S.meta?.embedded_location; if (em) marker(ctx, em.lon, em.lat, "#c77dff", "metadata GPS (claim)");
    }
  }
  function visible(c) { if (!mode3d) return true; const r = proj.rotate(); return d3.geoDistance([c.lon, c.lat], [-r[0], -r[1]]) < Math.PI / 2; }
  function marker(ctx, lon, lat, col, label) { const p = proj([lon, lat]); if (!p || !visible({lon, lat})) return; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[0] - 6, p[1] - 14); ctx.lineTo(p[0] + 6, p[1] - 14); ctx.closePath(); ctx.fillStyle = col; ctx.fill(); ctx.fillStyle = "#18283f"; ctx.fillText(label, p[0] + 8, p[1] - 10); }
  let dragG = null;
  cv.onmousedown = e => dragG = {x: e.clientX, y: e.clientY, rot: [...rot], panX, panY};
  window.addEventListener("mouseup", () => dragG = null);
  cv.onmousemove = e => { if (dragG) { const dx = e.clientX - dragG.x, dy = e.clientY - dragG.y, k = 90 / (proj.scale() / devicePixelRatio);
      if (mode3d) { rot = [dragG.rot[0] + dx * k * 0.9, Math.max(-90, Math.min(90, dragG.rot[1] - dy * k * 0.9)), 0]; } else { panX = dragG.panX + dx * devicePixelRatio; panY = dragG.panY + dy * devicePixelRatio; }
      makeProj(); draw(); return; }
    if (!proj) return; const ll = proj.invert([e.offsetX * devicePixelRatio, e.offsetY * devicePixelRatio]);
    $("#geoHover").textContent = ll && isFinite(ll[0]) ? `lat ${ll[1].toFixed(3)}, lon ${ll[0].toFixed(3)}` : "Drag to rotate or pan · wheel to zoom"; };
  cv.addEventListener("wheel", e => { e.preventDefault(); scale = proj.scale() * (e.deltaY > 0 ? 0.85 : 1.18); makeProj(); draw(); }, {passive: false});
  $("#btn3d").onclick = () => { mode3d = true; scale = null; $("#btn3d").classList.add("active"); $("#btn2d").classList.remove("active"); makeProj(); draw(); };
  $("#btn2d").onclick = () => { mode3d = false; scale = null; panX = panY = 0; $("#btn2d").classList.add("active"); $("#btn3d").classList.remove("active"); makeProj(); draw(); };
  $("#btnGeoReset").onclick = () => { scale = null; panX = panY = 0; focus(geo?.candidates?.[0]); };
  ["#lyrHeat", "#lyrCand", "#lyrBorders", "#lyrGrat", "#lyrEvid"].forEach(s => $(s).onchange = draw);
  function focus(c) { if (!c) { makeProj(); draw(); return; } rot = [-c.centre.lon, -c.centre.lat, 0];
    const r = Math.max(c.uncertainty_radius_km, 20); scale = Math.min(cv.width, cv.height) * 0.46 * Math.min(60, Math.max(1, 2500 / r));
    if (!mode3d) { scale = cv.width / 6.4 * Math.min(60, Math.max(1, 2500 / r)); makeProj(); const p = proj([c.centre.lon, c.centre.lat]); panX = cv.width / 2 - p[0] + panX; panY = cv.height / 2 - p[1] + panY; }
    makeProj(); draw(); }
  return {resize, set(g) { geo = g; draw(); }, focus};
})();

const Geo = {
  render(g) {
    Globe.set(g);
    if (!g) { $("#geoSummary").textContent = "Geographic analysis runs after the recording-context step."; $("#geoCands").innerHTML = ""; return; }
    const ev = c => `<ul class="ev">${c.supporting.map(e => `<li class="sup">+${e.contribution_nats} ${esc(e.label)} <span class="muted">(${esc(e.group)}; ${esc(e.detail)})</span></li>`).join("")}
      ${c.contradicting.map(e => `<li class="con">${e.contribution_nats} ${esc(e.label)} <span class="muted">(${esc(e.group)}; ${esc(e.detail)})</span></li>`).join("")}</ul>`;
    $("#geoSummary").innerHTML = `<p><b>${esc(g.summary || "")}</b></p>
      ${g.status === "estimate" ? `<p class="muted small">Effective area ≈ ${(g.effective_area_km2 || 0).toLocaleString()} km² of ${(g.land_area_km2 || 0).toLocaleString()} km² land · ${esc(g.week_basis)}</p>` : ""}
      <details><summary>How this was calculated</summary><pre class="ev">${esc(g.method)}</pre><ul>${(g.independence_notes || []).map(n => `<li class="small">${esc(n)}</li>`).join("")}</ul></details>
      <details><summary>Evidence items (${(g.evidence_items || []).length})</summary><pre class="ev">${esc(JSON.stringify(g.evidence_items, null, 1))}</pre></details>`;
    const card = (c, local) => `<div class="cand ${c.rank === 1 && !local ? "r1" : ""}" data-lat="${c.centre.lat}" data-lon="${c.centre.lon}" data-r="${c.uncertainty_radius_km}">
      <h4>${local ? "Local " : ""}#${c.rank} ${esc(c.name)}</h4>
      <div class="mono small">Estimated centre ${c.centre.lat.toFixed(4)}, ${c.centre.lon.toFixed(4)}</div>
      <div>${esc(c.resolution)} · uncertainty radius <b>${c.uncertainty_radius_km} km</b> · <span class="tier ${c.strength === "stronger" ? "confirmed" : c.strength === "moderate" ? "probable" : "possible"}">${c.strength}</span></div>
      <div class="small muted">Δ to best ${c.delta_to_best_nats} nats · relative mass ${c.relative_mass} · nearest city ${esc(c.nearest_city?.name || "-")} (${c.nearest_city?.distance_km ?? "-"} km)</div>
      <div class="small">Evidence for (+) and against (−):</div>${ev(c)}</div>`;
    $("#geoCands").innerHTML = (g.candidates || []).map(c => card(c, false)).join("") +
      (g.local_refinement ? `<h4>Local refinement</h4><div class="muted small">${esc(g.local_refinement.note)}</div>` + g.local_refinement.candidates.map(c => card(c, true)).join("") : "") +
      (g.species_profiles?.length ? `<h4>Species distribution & seasonality</h4>` + g.species_profiles.map(p => `<div class="small"><b>${esc(p.common_name)}</b> <i>${esc(p.scientific_name)}</i>: ${esc(p.distribution ? "main range: " + p.distribution.main_countries.join(", ") : "no range data")}${p.seasonality_at_best_candidate ? ". " + esc(p.seasonality_at_best_candidate.pattern) : ""}</div>`).join("") : "");
    $$("#geoCands .cand").forEach(el => el.onclick = () => Globe.focus({centre: {lat: +el.dataset.lat, lon: +el.dataset.lon}, uncertainty_radius_km: +el.dataset.r}));
    if (g.candidates?.length) setTimeout(() => Globe.focus(g.candidates[0]), 50);
  }
};

// ---------------------------------------------------------------- report
async function loadReport() {
  const r = await api("report", {raw: true}); if (!r.ok) return;
  $("#reportFrame").srcdoc = await r.text(); ["#expHtml", "#expPdf", "#expJson"].forEach(s => $(s).disabled = false);
}
async function exportAs(format) { $("#expMsg").textContent = "Exporting…";
  try { const r = await api("export", {method: "POST", body: {format}}); $("#expMsg").textContent = r.cancelled ? "" : r.path ? `Saved: ${r.path}` : (r.fallback || r.error || ""); }
  catch (e) { $("#expMsg").textContent = "Export failed: " + e.message; } }
$("#expHtml").onclick = () => exportAs("html"); $("#expPdf").onclick = () => exportAs("pdf"); $("#expJson").onclick = () => exportAs("json");
window.EchoMap = {printReport() { $("#reportFrame").contentWindow.print(); }};

// ---------------------------------------------------------------- models & settings
$("#btnModels").onclick = async () => {
  const [m, s] = await Promise.all([api("models"), api("settings")]);
  $("#modelsBody").innerHTML = `<div class="small muted">Models folder: <span class="mono">${esc(m.dir)}</span></div>
    <table class="mtable"><tr><th>Model</th><th>Status</th><th>Licence</th><th>Size</th><th>RAM</th><th>Limitations</th></tr>
    ${Object.values(m.models).map(x => `<tr><td>${esc(x.name)}<div class="muted small">${esc(x.source)}</div></td><td class="${x.installed ? "ok" : "no"}">${x.installed ? "installed" : "missing: " + esc(x.missing.join(", "))}</td>
      <td>${esc(x.licence)}</td><td>${x.size_mb} MB</td><td>${esc(x.ram_mb)} MB</td><td class="small">${esc(x.limitations)}</td></tr>`).join("")}</table>`;
  $("#setGbif").checked = s.allow_online_gbif; $("#setOsm").checked = s.allow_online_osm; $("#setWx").checked = s.allow_online_weather; $("#setThr").value = s.list_threshold;
  $("#modelsDlg").showModal();
};
$("#btnModelsDir").onclick = async e => { e.preventDefault(); await api("choose_models_dir", {method: "POST", body: {}}); $("#modelsDlg").close(); $("#btnModels").click(); };
$("#modelsDlg").addEventListener("close", () => { if ($("#modelsDlg").returnValue !== "close") return;
  api("settings", {method: "POST", body: {allow_online_gbif: $("#setGbif").checked, allow_online_osm: $("#setOsm").checked, allow_online_weather: $("#setWx").checked, list_threshold: +$("#setThr").value}}); });

// ---------------------------------------------------------------- startup: resume an existing case, deep-link tabs
(async () => {
  const m = location.hash.match(/tab=(\w+)/);
  if (m) { const b = $(`.tab[data-tab="${m[1]}"]`); if (b) b.click(); }
  try {
    const s = await api("status");
    if (s.state === "analysed" || s.state === "complete") { await loadResult(); if (m) { const b = $(`.tab[data-tab="${m[1]}"]`); if (b) b.click(); } }
  } catch (e) { /* no case yet */ }
})();
