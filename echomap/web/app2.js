/* EchoMap v1.0.3 workspace extensions: clue map (2D), 3D map (MapLibre), Intelligence tab, playback tools. */
"use strict";
const CLUE_COLORS = ["#4f8cff", "#ffb340", "#2ed3b7", "#ff6b9a", "#c77dff", "#7bd88f", "#f5e13c", "#ff8a4c", "#4dd0e1",
  "#e57373", "#aed581", "#ba68c8", "#90caf9", "#ffd54f", "#80cbc4"];
const GW = 720, GH = 360;

function decodeBits(b64) {
  const bin = atob(b64), out = new Uint8Array(GW * GH);
  for (let i = 0; i < out.length; i++) out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return out;
}
const gridToLonLat = ([x, y]) => [-180 + x * 0.5, 90 - y * 0.5];
const cellOf = (lon, lat) => { const r = Math.floor((90 - lat) / 0.5), c = Math.floor((lon + 180) / 0.5);
  return r >= 0 && r < GH && c >= 0 && c < GW ? r * GW + c : -1; };

// smooth outline polygons (grid coordinates) from a 0/1 raster
function rasterContours(raster, blur = 1.3, thr = 0.35) {
  const f = Float64Array.from(raster);
  if (d3.blur2) d3.blur2({data: f, width: GW, height: GH}, blur);
  return d3.contours().size([GW, GH]).thresholds([thr])(f)[0];
}

// shared clue state for 2D and 3D views
const GROUP_NAMES = {language: "Language", electrical_grid: "Electrical hum", biogeography: "Wildlife", bells: "Bells",
  cultural: "Sirens / culture", environment: "Water / coast", transport_infrastructure: "Trams / transport",
  place_names: "Spoken place names", reference_recordings: "Reference matches", user_context: "Your context"};
const Clues = {
  geo: null, layers: [], visible: new Set(), showLow: false, groupOff: new Set(), opacity: {}, ghost: new Set(),
  set(g) {
    this.geo = g; this.layers = []; this.opacity = {}; this.ghost = new Set();
    (g?.footprints || []).forEach((f, i) => {
      const raster = decodeBits(f.bits);
      this.layers.push({...f, color: CLUE_COLORS[i % CLUE_COLORS.length], raster, contour: rasterContours(raster)});
    });
    this.visible = new Set(this.layers.filter(l => l.used).map(l => l.id));
  },
  alpha(l) { return (this.opacity[l.id] ?? 1) * (this.ghost.has(l.id) ? 0.25 : 1); },
  active() { return this.layers.filter(l => this.visible.has(l.id) && (l.used || this.showLow) && !this.groupOff.has(l.group)); },
  places() { return this.geo?.likely_places?.places || []; },
  agreement() {
    const act = this.active().filter(l => l.used), cnt = new Uint8Array(GW * GH);
    act.forEach(l => { for (let i = 0; i < cnt.length; i++) cnt[i] += l.raster[i]; });
    let mx = 0; for (let i = 0; i < cnt.length; i++) if (cnt[i] > mx) mx = cnt[i];
    return {cnt, max: mx, n: act.length};
  },
  heatAt(lon, lat) {
    const h = this.geo?.heat || []; let best = null;
    for (const [la, lo, v] of h) if (Math.abs(la - lat) <= 0.25 && Math.abs(lo - lon) <= 0.25) best = v;
    return best;
  },
  inspect(lon, lat) {
    const ci = cellOf(lon, lat); if (ci < 0) return;
    const cover = this.layers.filter(l => l.raster[ci]), miss = this.layers.filter(l => l.used && !l.raster[ci]);
    const p = this.heatAt(lon, lat);
    const near = this.places().filter(q => d3.geoDistance([lon, lat], [q.lon, q.lat]) * 6371 <= Math.max(q.radius_km * 2, 25));
    const row = l => `<li><span class="sw" style="display:inline-block;width:10px;height:10px;background:${l.color};border-radius:2px"></span>
      <b>${esc(l.label)}</b> <span class="badge">${l.used ? "used" : "low confidence, not used"}</span>
      <span class="muted">conf ${l.confidence} · ${esc(GROUP_NAMES[l.group] || l.group)} · ${esc(l.source || "")} · precision: ${esc(l.precision || "-")}</span>
      ${l.note ? `<div class="muted">${esc(l.note)}</div>` : ""}</li>`;
    $("#pointInfo").classList.remove("hidden");
    $("#pointInfo").innerHTML = `<b>Point ${lat.toFixed(3)}, ${lon.toFixed(3)}</b>
      ${near.length ? `<div class="small">Likely place here: ${near.map(q => `<b>#${q.rank} ${esc(q.name)}</b> (${(q.share * 100).toFixed(1)}%)`).join(", ")}</div>` : ""}
      <div class="small">Combined relative plausibility: ${p == null ? "outside the top area" : (p * 100).toFixed(1) + "% of the maximum"}</div>
      <div class="small"><b>${cover.length}</b> clue(s) cover this point:</div><ul class="small">${cover.map(row).join("") || "<li class='muted'>none</li>"}</ul>
      <div class="small"><b>${miss.length}</b> used clue(s) do NOT cover it (contradicting evidence):</div><ul class="small">${miss.map(row).join("") || "<li class='muted'>none</li>"}</ul>
      <div class="muted small">Country clues use exact borders; other areas are approximate (0.5° cells, about 55 km); places are circles. Inferred, not measured.</div>`;
  },
};

// ---------------------------------------------------------------- 2D clue map
const ClueMap = (() => {
  const cv = $("#clueCanvas"); let world = null, world10 = null, loading10 = false, proj = null, path = null, tf = d3.zoomIdentity, feats = [];
  let byNum = new Map();
  const index = w => { byNum = new Map(); w.features.forEach(f => { if (f.id != null) byNum.set(String(f.id).padStart(3, "0"), f); }); };
  d3.json("/static/vendor/countries-50m.json").then(t => { world = topojson.feature(t, t.objects.countries); feats = world.features; index(world); resize(); });
  function hiRes() {  // sharper borders once zoomed in (Natural Earth 1:10m), loaded on demand
    if (world10 || loading10) return; loading10 = true;
    d3.json("/static/vendor/countries-10m.json").then(t => { world10 = topojson.feature(t, t.objects.countries); feats = world10.features; index(world10); draw(); })
      .catch(() => { /* not bundled: keep 1:50m */ });
  }
  function resize() {
    if (!cv.clientWidth) return;
    cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio;
    proj = d3.geoNaturalEarth1().fitExtent([[10, 10], [cv.width - 10, cv.height - 10]], {type: "Sphere"});
    draw();
  }
  function ringPath(ctx, rings) {
    for (const ring of rings) { ring.forEach((pt, i) => { const p = proj(gridToLonLat(pt)); if (!p) return; i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); }); ctx.closePath(); }
  }
  const hex = (c, a) => c + Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");
  const fillA = () => +$("#clFill").value / 100;
  function circle(ctx, lon, lat, rkm, minPx) {   // geodesic circle, never smaller than minPx on screen
    const p = proj([lon, lat]); if (!p) return null;
    const g = d3.geoCircle().center([lon, lat]).radius(rkm / 111.195)();
    ctx.beginPath(); path(g);
    const b = path.bounds(g), px = (b[1][0] - b[0][0]) * tf.k / 2;
    if (!(px >= minPx)) { ctx.beginPath(); ctx.arc(p[0], p[1], minPx / tf.k, 0, 7); }
    return p;
  }
  function shapePath(ctx, l) {
    ctx.beginPath();
    if (l.shape?.kind === "countries" && byNum.size) {
      (l.shape.numeric || []).forEach(n => { const f = n && byNum.get(String(n).padStart(3, "0")); if (f) path(f); });
    } else ringPath(ctx, l.contour.coordinates.flat());
  }
  function draw() {
    if (!proj) return;
    const ctx = cv.getContext("2d"), k = tf.k, dpr = devicePixelRatio;
    if (k > 5) hiRes();
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.setTransform(k, 0, 0, k, tf.x * dpr, tf.y * dpr);
    path = d3.geoPath(proj, ctx);
    const w = world10 || world;
    ctx.beginPath(); path({type: "Sphere"}); ctx.fillStyle = "#0b1626"; ctx.fill();
    if (w) { ctx.beginPath(); path(w); ctx.fillStyle = "#26303f"; ctx.fill(); ctx.strokeStyle = "#5a6a84"; ctx.lineWidth = 0.7 / k; ctx.stroke(); }
    const g = Clues.geo;
    if (g && $("#clProb").checked) {
      for (const [la, lo, v] of g.heat || []) { const p = proj([lo, la]); if (!p) continue; const s = Math.max(1.5, 3.2 * proj.scale() / 400);
        const kk = Math.round(80 + 175 * v) * 3; ctx.fillStyle = `rgba(${LUT[kk]},${LUT[kk + 1]},${LUT[kk + 2]},${0.25 + 0.6 * v})`; ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s); }
    }
    const act = Clues.active(), fa = fillA();
    // area clues (countries / smooth regions)
    act.filter(l => l.shape?.kind !== "points").forEach(l => {
      const a = Clues.alpha(l); if (a <= 0) return;
      shapePath(ctx, l); ctx.fillStyle = hex(l.color, (l.used ? 0.55 : 0.3) * fa * a); ctx.fill("evenodd");
    });
    if ($("#clAgree").checked) {
      const ag = Clues.agreement();
      if (ag.max >= 2) {
        const top = new Uint8Array(GW * GH); for (let i = 0; i < top.length; i++) top[i] = ag.cnt[i] >= ag.max ? 1 : 0;
        const c = rasterContours(top, 1.0, 0.4);
        ctx.beginPath(); ringPath(ctx, c.coordinates.flat()); ctx.fillStyle = "rgba(255,90,60,0.22)"; ctx.fill("evenodd");
        ctx.strokeStyle = "#ff5a3c"; ctx.lineWidth = 2.5 / k; ctx.stroke();
        const lab = largestCentroid(c); if (lab && k < 8) halo(ctx, `Most clues agree here (${ag.max} of ${ag.n})`, lab, 13 / k, "#ffd2c8");
      }
    }
    act.filter(l => l.shape?.kind !== "points").forEach(l => { const a = Clues.alpha(l); if (a <= 0) return;
      shapePath(ctx, l); ctx.setLineDash(l.used ? [] : [6 / k, 4 / k]);
      ctx.strokeStyle = `rgba(255,255,255,${0.92 * a})`; ctx.lineWidth = (l.used ? 1.6 : 1.1) / k; ctx.stroke(); ctx.setLineDash([]); });
    // point clues (tram networks, mentioned cities, bell towers): small circles at their real size
    act.filter(l => l.shape?.kind === "points").forEach(l => { const a = Clues.alpha(l); if (a <= 0) return;
      for (const q of l.shape.points) { const p = circle(ctx, q.lon, q.lat, q.r || 3, 2.5); if (!p) continue;
        ctx.fillStyle = hex(l.color, 0.5 * a); ctx.fill(); ctx.strokeStyle = `rgba(255,255,255,${0.8 * a})`; ctx.lineWidth = 1 / k; ctx.stroke();
        if ($("#clLabels").checked && k >= 12) halo(ctx, q.name, [p[0] + 5 / k, p[1] + 3 / k], 10 / k, l.color, "left"); } });
    if ($("#clLabels").checked) {
      ctx.textAlign = "center";
      for (const f of feats) { const a = path.area(f) * k * k; if (a < 2500 * dpr * dpr) continue; const c = path.centroid(f); if (!isFinite(c[0])) continue;
        ctx.font = `${Math.min(13, 8 + Math.sqrt(a) / 80) * dpr / k}px Segoe UI`; ctx.fillStyle = "rgba(220,228,240,.55)";
        ctx.fillText((f.properties.name || "").toUpperCase(), c[0], c[1]); }
      if (k < 8) act.filter(l => l.shape?.kind !== "points").forEach(l => { const c = largestCentroid(l.contour); if (c && Clues.alpha(l) > 0.3) halo(ctx, l.label.length > 38 ? l.label.slice(0, 36) + "…" : l.label, c, 11 / k, l.color); });
    }
    if (g && $("#clCand").checked) {   // estimate circles: filled, outlined, labelled with their radius
      (g.candidates || []).forEach(c => { const p = circle(ctx, c.centre.lon, c.centre.lat, c.uncertainty_radius_km, 6); if (!p) return;
        ctx.fillStyle = c.rank === 1 ? "rgba(255,90,60,0.10)" : "rgba(255,179,64,0.07)"; ctx.fill();
        ctx.strokeStyle = c.rank === 1 ? "#ff5a3c" : "#ffb340"; ctx.lineWidth = 2 / k; ctx.setLineDash(c.strength === "weaker" ? [5 / k, 4 / k] : []); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(p[0], p[1], 3.5 / k, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
        halo(ctx, `#${c.rank} ${c.name} · ±${c.uncertainty_radius_km} km`, [p[0] + 8 / k, p[1] - 8 / k], 11 / k, "#fff", "left"); });
    }
    if (g && $("#clPlaces").checked) {   // likely places: circles sized to the place, strongest first on top
      Clues.places().slice().reverse().forEach(q => { const p = circle(ctx, q.lon, q.lat, Math.max(q.radius_km, 3), 5); if (!p) return;
        const top = q.rank === 1;
        ctx.fillStyle = top ? "rgba(255,70,50,0.35)" : `rgba(255,170,60,${0.12 + 0.5 * q.share})`; ctx.fill();
        ctx.strokeStyle = top ? "#ff4632" : "#ffb340"; ctx.lineWidth = (top ? 2.6 : 1.6) / k; ctx.stroke();
        if (q.rank <= 5 || k >= 6) halo(ctx, `${q.rank}. ${q.name} ${(q.share * 100).toFixed(0)}%`, [p[0] + 7 / k, p[1] - 6 / k], (top ? 13 : 11) / k, top ? "#ffd2c8" : "#ffe2b0", "left"); });
    }
  }
  function largestCentroid(mp) {
    let best = null, ba = 0;
    for (const poly of mp.coordinates) { const ring = poly[0].map(pt => proj(gridToLonLat(pt))).filter(Boolean); if (ring.length < 3) continue;
      const a = Math.abs(d3.polygonArea(ring)); if (a > ba) { ba = a; best = d3.polygonCentroid(ring); } }
    return best;
  }
  function halo(ctx, text, [x, y], size, color, align = "center") {
    ctx.font = `600 ${size * devicePixelRatio}px Segoe UI`; ctx.textAlign = align; ctx.lineWidth = 3 * devicePixelRatio / tf.k;
    ctx.strokeStyle = "rgba(8,12,20,.85)"; ctx.strokeText(text, x, y); ctx.fillStyle = color; ctx.fillText(text, x, y);
  }
  const zoom = d3.zoom().scaleExtent([1, 2000]).on("zoom", e => { tf = e.transform; draw(); });
  d3.select(cv).call(zoom).on("dblclick.zoom", null);
  const toLL = e => { const dpr = devicePixelRatio; return proj.invert([(e.offsetX * dpr - tf.x * dpr) / tf.k, (e.offsetY * dpr - tf.y * dpr) / tf.k]); };
  cv.addEventListener("click", e => { if (!proj) return; const ll = toLL(e); if (ll && isFinite(ll[0])) Clues.inspect(ll[0], ll[1]); });
  cv.addEventListener("mousemove", e => { if (!proj) return; const ll = toLL(e);
    if (ll && isFinite(ll[0])) $("#clueHover").textContent = `lat ${ll[1].toFixed(3)}, lon ${ll[0].toFixed(3)} · zoom ${tf.k.toFixed(1)}× · click to inspect clues`; });
  // fit the view to a set of circles {lon, lat, r}
  function fit(circles, maxK = 2000) {
    if (!proj || !circles.length) return;
    const dpr = devicePixelRatio; let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    circles.forEach(c => { const b = d3.geoPath(proj).bounds(d3.geoCircle().center([c.lon, c.lat]).radius(Math.max(c.r, 5) / 111.195)());
      x0 = Math.min(x0, b[0][0]); y0 = Math.min(y0, b[0][1]); x1 = Math.max(x1, b[1][0]); y1 = Math.max(y1, b[1][1]); });
    if (!isFinite(x0)) return;
    const kk = Math.max(1, Math.min(maxK, 0.8 * Math.min(cv.width / (x1 - x0), cv.height / (y1 - y0))));
    const cx = (x0 + x1) / 2 / dpr, cy = (y0 + y1) / 2 / dpr;
    d3.select(cv).transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(cv.clientWidth / 2 - cx * kk, cv.clientHeight / 2 - cy * kk).scale(kk));
  }
  function focus(c) { if (c) fit([{lon: c.centre.lon, lat: c.centre.lat, r: c.uncertainty_radius_km}]); }
  function focusPlace(q) { if (q) fit([{lon: q.lon, lat: q.lat, r: Math.max(q.radius_km * 3, 15)}]); }
  function best() {  // the top places (within 1 nat of the best) if any, else the best estimate circle
    const P = Clues.places().filter(q => q.delta_to_best_nats <= 1.0).slice(0, 8);
    if (P.length) return fit(P.map(q => ({lon: q.lon, lat: q.lat, r: Math.max(q.radius_km * 2, 15)})));
    const c = Clues.geo?.candidates?.[0]; if (c) focus(c);
  }
  $("#clReset").onclick = () => d3.select(cv).transition().call(zoom.transform, d3.zoomIdentity);
  $("#clZoomIn").onclick = () => d3.select(cv).transition().duration(250).call(zoom.scaleBy, 2);
  $("#clZoomOut").onclick = () => d3.select(cv).transition().duration(250).call(zoom.scaleBy, 0.5);
  $("#clBest").onclick = best;
  ["#clAgree", "#clProb", "#clCand", "#clLabels", "#clPlaces"].forEach(s => $(s).onchange = draw);
  $("#clFill").oninput = draw;
  $("#clLow").onchange = () => { Clues.showLow = $("#clLow").checked; renderClueList(); draw(); Map3D.refresh(); };
  return {resize, draw, focus, focusPlace, best};
})();

function renderFilters() {
  const groups = [...new Set(Clues.layers.filter(l => l.used || Clues.showLow).map(l => l.group))];
  $("#clFilters").innerHTML = groups.length ? `<span class="muted small">Filters:</span>` + groups.map(gname =>
    `<span class="chip ${Clues.groupOff.has(gname) ? "off" : ""}" data-g="${esc(gname)}">${esc(GROUP_NAMES[gname] || gname)}</span>`).join("") : "";
  $$("#clFilters .chip").forEach(ch => ch.onclick = () => { const gname = ch.dataset.g;
    Clues.groupOff.has(gname) ? Clues.groupOff.delete(gname) : Clues.groupOff.add(gname); renderFilters(); ClueMap.draw(); Map3D.refresh(); });
}

function renderPlaces() {
  const lp = Clues.geo?.likely_places, P = lp?.places || [];
  if (!P.length) { $("#placeList").innerHTML = "No point clues (tram networks, spoken place names or bell towers) in this recording."; return; }
  $("#placeList").innerHTML = P.map(q => `<div class="place" data-r="${q.rank}"><span class="rk">${q.rank}</span>
      <div><b>${esc(q.name)}</b>, ${esc(q.country || "")} <span class="muted">· circle ${q.radius_km} km</span>
      <div class="muted">${esc(q.supporting.map(s => s.label).slice(0, 4).join(" · "))}</div>
      <div class="bar" style="width:${Math.max(2, q.share * 100)}%"></div></div><span>${(q.share * 100).toFixed(1)}%</span></div>`).join("") +
    `<div class="muted small" style="margin-top:4px">${esc(lp.note || "")}</div>`;
  $$("#placeList .place").forEach(el => el.onclick = () => ClueMap.focusPlace(P[+el.dataset.r - 1]));
}

function renderClueList() {
  const g = Clues.geo;
  renderFilters(); renderPlaces();
  if (!g || !Clues.layers.length) { $("#clueList").innerHTML = "No geographically locatable clues in this recording."; }
  else $("#clueList").innerHTML = Clues.layers.filter(l => l.used || Clues.showLow).map(l => `<div class="clue ${l.used ? "" : "low"} ${Clues.ghost.has(l.id) ? "ghost" : ""}" data-id="${l.id}">
      <input type="checkbox" ${Clues.visible.has(l.id) ? "checked" : ""} title="Show / hide"><span class="sw" style="background:${l.color}"></span>
      <div><b class="nm" title="Click to make see-through">${esc(l.label)}</b> <span class="badge">${esc(GROUP_NAMES[l.group] || l.group)}</span>${l.used ? "" : '<span class="badge">low confidence, not used in the estimate</span>'}
      <div class="muted">confidence ${l.confidence} · area ${(l.area_km2 || 0).toLocaleString()} km² · ${esc(l.precision || "")}</div>
      <div class="muted">${esc((l.countries || []).slice(0, 5).map(c => c.name).join(", "))}${(l.countries || []).length > 5 ? "…" : ""}</div>
      <div class="muted">source: ${esc(l.source || "")}${l.record ? " · detection " + esc(l.record) : ""}</div></div>
      <div class="ctl"><button class="solo" title="Show only this clue">◉</button>
      <input type="range" min="0" max="100" value="${Math.round((Clues.opacity[l.id] ?? 1) * 100)}" title="Opacity"></div></div>`).join("");
  const redraw = () => { ClueMap.draw(); Map3D.refresh(); };
  $$("#clueList .clue").forEach(row => {
    const id = row.dataset.id;
    row.querySelector("input[type=checkbox]").onchange = e => { e.target.checked ? Clues.visible.add(id) : Clues.visible.delete(id); redraw(); };
    row.querySelector(".nm").onclick = () => { Clues.ghost.has(id) ? Clues.ghost.delete(id) : Clues.ghost.add(id); row.classList.toggle("ghost"); redraw(); };
    row.querySelector("input[type=range]").oninput = e => { Clues.opacity[id] = +e.target.value / 100; redraw(); };
    row.querySelector(".solo").onclick = () => {
      const only = Clues.visible.size === 1 && Clues.visible.has(id);
      Clues.visible = only ? new Set(Clues.layers.filter(l => l.used).map(l => l.id)) : new Set([id]);
      renderClueList(); redraw(); };
  });
  const un = g?.unlocated || [];
  const byCat = {}; un.forEach(u => (byCat[u.category] = byCat[u.category] || []).push(u));
  $("#unlocList").innerHTML = un.length ? `${un.length} detection(s) cannot be placed on a map:` + Object.entries(byCat).map(([c, us]) =>
    `<details><summary>${esc(c)} (${us.length})</summary><ul>${us.map(u => `<li>${esc(u.record)} ${esc(u.label)}${u.specific ? " · " + esc(u.specific) : ""} <span class="muted">(${esc(u.reason)})</span></li>`).join("")}</ul></details>`).join("") : "-";
}

// ---------------------------------------------------------------- 3D map (MapLibre, online tiles, opt-in)
const Map3D = (() => {
  let ml = null, map = null, ready = false, wanted = false;
  const DEM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
  const SAT = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg";
  async function ensure() {
    const st = await api("settings");
    if (!st.allow_online_maps) {
      $("#m3Note").innerHTML = `The 3D map uses online map tiles (OpenFreeMap basemap and 3D buildings, AWS Terrain Tiles elevation, optional EOX Sentinel-2 imagery). It is off by default because tile requests reveal which areas you look at. <button id="m3Enable" class="primary">Enable online 3D map</button> The offline clue map and globe always work.`;
      $("#m3Enable").onclick = async () => { await api("settings", {method: "POST", body: {allow_online_maps: true}}); ensure(); };
      return;
    }
    if (map) return;
    $("#m3Note").textContent = "Loading 3D map…";
    try {
      if (!document.querySelector("#mlcss")) { const l = document.createElement("link"); l.id = "mlcss"; l.rel = "stylesheet"; l.href = "/static/vendor/maplibre/maplibre-gl.css"; document.head.appendChild(l); }
      const mod = await import("/static/vendor/maplibre/maplibre-gl.mjs");
      ml = mod.default || mod;
      if (ml.setWorkerUrl) ml.setWorkerUrl("/static/vendor/maplibre/maplibre-gl-worker.mjs");
      map = new ml.Map({container: "map3d", style: "https://tiles.openfreemap.org/styles/liberty", center: [10, 30], zoom: 1.3,
        pitch: 0, maxPitch: 80, attributionControl: {compact: true,
          customAttribution: "Elevation: Mapzen/AWS Terrain Tiles · Imagery: Sentinel-2 cloudless by EOX (CC BY-NC-SA 4.0) · Clues: EchoMap (inferred, not measured)"}});
      map.addControl(new ml.NavigationControl({visualizePitch: true}));
      map.on("load", () => {
        try { map.setProjection({type: "globe"}); } catch (e) { /* older API */ }
        map.addSource("dem", {type: "raster-dem", tiles: [DEM], encoding: "terrarium", tileSize: 256, maxzoom: 14});
        map.addSource("sat", {type: "raster", tiles: [SAT], tileSize: 256, maxzoom: 15});
        map.addLayer({id: "sat", type: "raster", source: "sat", layout: {visibility: "none"}}, firstSymbol());
        setTerrain(); ready = true; refresh(); $("#m3Note").textContent = "Drag to pan, right-drag or Ctrl+drag to rotate and tilt, wheel to zoom. Click a point to inspect clues.";
      });
      map.on("click", e => Clues.inspect(e.lngLat.lng, e.lngLat.lat));
      map.on("error", e => { $("#m3Note").textContent = "Map tile error: " + (e?.error?.message || "unknown"); });
    } catch (e) { $("#m3Note").textContent = "3D map could not start: " + e.message + ". The offline clue map is still available."; }
  }
  function firstSymbol() { const l = map.getStyle().layers.find(l => l.type === "symbol"); return l ? l.id : undefined; }
  function setTerrain() { if (!map) return; try { map.setTerrain($("#m3Terrain").checked ? {source: "dem", exaggeration: 1.4} : null); } catch (e) {} }
  function polysFromContour(mp) {
    return mp.coordinates.map(poly => poly.map(ring => { const r = ring.map(gridToLonLat); r.push(r[0]); return r.reverse(); }));
  }
  function refresh() {
    if (!ready || !map) return;
    ["clues-fill", "clues-line", "prob", "cands-fill", "cands-line", "places-fill", "places-line", "places-label"].forEach(id => map.getLayer(id) && map.removeLayer(id));
    ["clues", "probsrc", "cands", "places"].forEach(id => map.getSource(id) && map.removeSource(id));
    const showLow = $("#m3Low").checked;
    const feats = $("#m3Clues").checked ? Clues.layers.filter(l => Clues.visible.has(l.id) && (l.used || showLow) && !Clues.groupOff.has(l.group) && l.shape?.kind !== "points").map(l => ({type: "Feature",
      properties: {color: l.color, label: l.label, used: l.used ? 1 : 0, a: Clues.alpha(l)}, geometry: {type: "MultiPolygon", coordinates: polysFromContour(l.contour)}})) : [];
    map.addSource("clues", {type: "geojson", data: {type: "FeatureCollection", features: feats}});
    map.addLayer({id: "clues-fill", type: "fill", source: "clues", paint: {"fill-color": ["get", "color"], "fill-opacity": ["*", ["get", "a"], ["case", ["==", ["get", "used"], 1], 0.25, 0.12]]}});
    map.addLayer({id: "clues-line", type: "line", source: "clues", paint: {"line-color": "#ffffff", "line-width": 1.6, "line-opacity": 0.9}});
    const g = Clues.geo;
    if (g && $("#m3Prob").checked) {
      map.addSource("probsrc", {type: "geojson", data: {type: "FeatureCollection", features: (g.heat || []).map(([la, lo, v]) => ({type: "Feature", properties: {w: v}, geometry: {type: "Point", coordinates: [lo, la]}}))}});
      map.addLayer({id: "prob", type: "heatmap", source: "probsrc", paint: {"heatmap-weight": ["get", "w"], "heatmap-radius": 18, "heatmap-opacity": 0.55}});
    }
    const cf = (g?.candidates || []).map(c => ({type: "Feature", properties: {rank: c.rank}, geometry: d3.geoCircle().center([c.centre.lon, c.centre.lat]).radius(c.uncertainty_radius_km / 111.195)()}));
    cf.forEach(f => f.geometry.coordinates = f.geometry.coordinates.map(r => r.slice().reverse()));
    map.addSource("cands", {type: "geojson", data: {type: "FeatureCollection", features: cf}});
    map.addLayer({id: "cands-line", type: "line", source: "cands", paint: {"line-color": ["case", ["==", ["get", "rank"], 1], "#ff5a3c", "#ffb340"], "line-width": 2.5}});
    const pf = Clues.places().map(q => { const c = d3.geoCircle().center([q.lon, q.lat]).radius(Math.max(q.radius_km, 3) / 111.195)();
      c.coordinates = c.coordinates.map(r => r.slice().reverse());
      return {type: "Feature", properties: {rank: q.rank, label: `${q.rank}. ${q.name} ${(q.share * 100).toFixed(0)}%`}, geometry: c}; });
    map.addSource("places", {type: "geojson", data: {type: "FeatureCollection", features: pf}});
    map.addLayer({id: "places-fill", type: "fill", source: "places", paint: {"fill-color": ["case", ["==", ["get", "rank"], 1], "#ff4632", "#ffb340"], "fill-opacity": 0.22}});
    map.addLayer({id: "places-line", type: "line", source: "places", paint: {"line-color": ["case", ["==", ["get", "rank"], 1], "#ff4632", "#ffb340"], "line-width": 2.5}});
    if (map.getLayer("building-3d")) map.setLayoutProperty("building-3d", "visibility", $("#m3Build").checked ? "visible" : "none");
    map.setLayoutProperty("sat", "visibility", $("#m3Sat").checked ? "visible" : "none");
  }
  $("#m3Terrain").onchange = setTerrain;
  $("#m3Flat").onchange = () => { if (!map) return; const flat = $("#m3Flat").checked;   // flat street map: Mercator, no tilt or terrain
    try { map.setProjection({type: flat ? "mercator" : "globe"}); } catch (e) {}
    if (flat) { map.easeTo({pitch: 0, bearing: 0}); try { map.setTerrain(null); } catch (e) {} } else setTerrain(); };
  ["#m3Build", "#m3Sat", "#m3Clues", "#m3Low", "#m3Prob"].forEach(s => $(s).onchange = refresh);
  $("#m3Globe").onclick = () => map && map.flyTo({center: [10, 30], zoom: 1.3, pitch: 0, bearing: 0});
  $("#m3Lead").onclick = () => { const q = Clues.places()[0]; if (map && q) return map.flyTo({center: [q.lon, q.lat], zoom: 11, pitch: $("#m3Flat").checked ? 0 : 50});
    const c = Clues.geo?.candidates?.[0]; if (map && c) map.flyTo({center: [c.centre.lon, c.centre.lat],
    zoom: Math.max(3, Math.min(11, 9 - Math.log2(Math.max(c.uncertainty_radius_km, 5) / 5))), pitch: 55}); };
  return {show() { wanted = true; ensure(); if (map) map.resize(); }, refresh, set() { refresh(); }};
})();

// geo sub-view switching
$$(".geov").forEach(b => b.onclick = () => {
  $$(".geov").forEach(x => x.classList.toggle("active", x === b));
  ["clue", "map3d", "globe"].forEach(v => $("#v-" + v).classList.toggle("hidden", v !== b.dataset.v));
  if (b.dataset.v === "clue") ClueMap.resize(); else if (b.dataset.v === "map3d") Map3D.show(); else Globe.resize();
});
$('.tab[data-tab="geo"]').addEventListener("click", () => setTimeout(() => ClueMap.resize(), 30));
window.addEventListener("resize", () => ClueMap.resize());

const _geoRender0 = Geo.render;
Geo.render = function (g) {
  _geoRender0(g); Clues.set(g); renderClueList(); ClueMap.draw(); Map3D.set(g);
  $("#geoStatus").textContent = g ? `${(g.footprints || []).length} locatable clue(s) · ${(g.unlocated || []).length} unlocated detection(s)` : "";
  if (g?.likely_places?.places?.length || g?.candidates?.length) setTimeout(() => ClueMap.best(), 80);
};

// ---------------------------------------------------------------- Intelligence tab
const Intel = {
  render(r) {
    const fmtRec = id => { const d = S.inventory.find(x => x.id === id); return d ? `${esc(id)} ${esc(d.label)}` : esc(id); };
    const playBtn = (t0, t1) => `<button class="round mini" data-t0="${t0}" data-t1="${t1}" title="Play">▶</button>`;
    // transcripts
    const tr = r.transcripts;
    $("#trList").innerHTML = !tr ? '<div class="muted">No transcription ran (see Capabilities).</div>' :
      `<div class="muted">${tr.counts.clear} clear · ${tr.counts.uncertain} uncertain · ${tr.counts["likely spurious"]} likely spurious. Whisper can invent text on noise or very faint audio, so rely only on "clear" lines, and check even those by listening.</div>` +
      tr.segments.map(t => `<div class="cap-row"><span>${playBtn(t.start, t.end)} ${fmtT(t.start)}–${fmtT(t.end).slice(3)}<br>
        <span class="tier ${t.quality === "clear" ? "confirmed" : t.quality === "uncertain" ? "possible" : "unclassified"}">${esc(t.quality)}</span></span>
        <span>${t.text ? `“${esc(t.text)}”` : "<i>(no text)</i>"} <span class="muted">[${esc(t.language)}${t.enhanced ? ", enhanced faint speech" : ""}; no-speech ${t.no_speech_prob}, logprob ${t.avg_logprob}, level ${t.level_dbfs} dBFS]</span></span></div>`).join("");
    // capabilities
    const cap = r.capabilities;
    $("#capList").innerHTML = cap ? `<div class="muted">Analysis mode: <b>${esc(cap.mode_title)}</b></div>` + cap.items.map(i => `<div class="cap-row"><span class="st-${i.status}">${esc(i.status.replaceAll("_", " "))}</span>
      <span><b>${esc(i.name)}</b> <span class="badge">${esc(i.class_title || i.class)}</span>${i.experimental ? '<span class="badge">experimental</span>' : ""}${i.reason ? `<div class="muted">${esc(i.reason)}</div>` : ""}</span></div>`).join("") : "-";
    // integrity
    const it = r.integrity;
    $("#intList").innerHTML = it ? `${it.summary_flags.length ? `<ul>${it.summary_flags.map(f => `<li class="con">${esc(f)}</li>`).join("")}</ul>` : '<div class="ok">No integrity flags raised.</div>'}
      <div>Bandwidth: ${it.bandwidth.effective_bandwidth_hz} Hz of ${it.bandwidth.nyquist_hz} Hz Nyquist. ${esc(it.bandwidth.interpretation)}</div>
      <div>${esc(it.clipping.interpretation)} · ${esc(it.dropouts.interpretation)} · ${esc(it.dc_offset.interpretation)}</div>
      ${(it.metadata || []).map(m => `<div class="muted">• ${esc(m.check)}: ${esc(m.finding)}. ${esc(m.interpretation)}</div>`).join("")}<div class="muted">${esc(it.note)}</div>` : "-";
    // relations
    const rel = r.relations;
    $("#relList").innerHTML = rel ? `${rel.hypotheses.length ? "<b>Sequence hypotheses</b>" + rel.hypotheses.map(h => `<div>• <b>${esc(h.pattern)}</b>: ${esc(h.explanation)} <span class="badge">${esc(h.status)}</span> <span class="muted">(${h.records.map(fmtRec).join(", ")})</span></div>`).join("") : '<div class="muted">No sequence hypotheses.</div>'}
      ${rel.repetitions.length ? "<b>Repetitions</b>" + rel.repetitions.map(p => `<div>• ${esc(p.label)}: ${p.count}×, mean interval ${p.mean_interval_s} s (${p.regular ? "regular" : "irregular"}, CV ${p.interval_cv})</div>`).join("") : ""}
      <div class="muted">${rel.edges.length} temporal relations (overlap, simultaneous onset, followed by). ${esc(rel.note)}</div>` : "-";
    // unknown sounds
    const dis = r.discovery;
    $("#unkList").innerHTML = dis ? `<div>${esc(dis.summary)}</div>` + dis.events.map(u => `<div class="cap-row"><span>${playBtn(u.start, u.end)} ${fmtT(u.start)}<br><span class="badge">${esc(u.family || "")}</span> novelty ${u.novelty}</span>
      <span><b>${esc(u.description)}</b><div class="muted">closest labels: ${u.closest_model_labels.map(l => `${esc(l.label)} ${l.score}`).join(", ")}</div>
      ${u.disagreements.length ? `<div class="con">${u.disagreements.map(esc).join("; ")}</div>` : ""}
      ${u.similar_references?.length ? `<div>similar references: ${u.similar_references.map(s => `${esc(s.title)} <span class="tierb">${esc(s.tier)}</span>`).join("; ")}</div>` : ""}
      ${u.matches_earlier_unknown ? `<div class="muted">resembles an earlier unknown (case ${esc(u.matches_earlier_unknown.case)}, sim ${u.matches_earlier_unknown.similarity})</div>` : ""}</span></div>`).join("")
      : '<div class="muted">Unknown-sound discovery runs in Experimental mode.</div>';
    // references
    const ref = r.references;
    $("#refList").innerHTML = !ref ? '<div class="muted">Reference search runs in Deep mode.</div>' : !ref.available ? esc(ref.reason) :
      `<div class="muted">Library: ${ref.library_size} openly licensed reference recordings. ${esc(ref.note)}</div>
      ${(ref.fingerprint_matches || []).map(m => `<div class="ok">• <b>${esc(m.tier)}</b>: ${esc(m.title)} (${m.aligned_hits} aligned landmarks) <span class="muted">${esc(m.license || "")}</span></div>`).join("")}
      ${ref.events.map(e => `<div>• ${fmtRec(e.record)}: ${e.matches.map(m => `${esc(m.title)} <span class="tierb">${esc(m.tier)} ${m.similarity}</span>${m.lat != null ? ` <span class="muted">(${m.lat.toFixed(2)}, ${m.lon.toFixed(2)})</span>` : ""}`).join("; ")}</div>`).join("") || '<div class="muted">No event resembles a library reference above the weak-resemblance threshold.</div>'}`;
    // bells
    const b = r.bells;
    $("#bellList").innerHTML = !b ? '<div class="muted">No bells analysed (bells are analysed in Deep mode when a bell is detected).</div>' :
      `<div>${b.analysis.valid_strikes} strike(s) with bell partial structure; ${b.analysis.ring_size_observed} bell(s) reconstructed${b.analysis.tenor_nominal_hz ? `, tenor nominal ${b.analysis.tenor_nominal_hz} Hz` : ""}. ${esc(b.analysis.reason || "")}</div>
      ${(b.analysis.bells || []).map(x => `<span class="badge">${x.note} ${x.nominal_hz} Hz ×${x.strikes}</span>`).join(" ")}
      ${b.dove_match?.candidates?.length ? `<div><b>${b.dove_match.n_compatible} compatible Dove's Guide rings</b> (top 10):</div><ul>${b.dove_match.candidates.slice(0, 10).map(c => `<li>${esc(c.place)}, ${esc(c.dedication || "")} (${esc(c.county || c.country || "")}): ${c.bells} bells, tenor ${c.tenor_hz} Hz ${esc(c.tenor_note || "")}, error ${c.rms_error_cents} cents (${esc(c.basis)})</li>`).join("")}</ul><div class="muted">${esc(b.dove_match.note)}</div>` : ""}`;
    // direction
    const sp = r.spatial;
    const dirs = S.inventory.filter(d => d.direction && d.direction.supported);
    $("#dirList").innerHTML = !sp ? "-" : !sp.supported ? `<div class="muted">${esc(sp.reason)}</div>` :
      `<div class="muted">Stereo recording (channel correlation ${sp.inter_channel_correlation}). Lateral cues only; front/back, elevation and distance cannot be determined.</div>` +
      dirs.slice(0, 60).map(d => `<div>• ${esc(d.id)} ${esc(d.label)}: ${d.direction.lateral === "undetermined" ? "undetermined" : `${esc(d.direction.lateral)}, ${d.direction.lateral_angle_deg_range?.[0]}° to ${d.direction.lateral_angle_deg_range?.[1]}°`}
        <span class="muted">(ITD ${d.direction.itd_ms} ms, ILD ${d.direction.ild_db} dB)</span></div>`).join("");
    $$("#tab-intel button.mini").forEach(bt => bt.onclick = () => playSegment({id: "seg", start: +bt.dataset.t0, end: +bt.dataset.t1}));
  }
};
$("#txtGo").onclick = async () => {
  const q = $("#txtQ").value.trim(); if (!q) return;
  try { const r = await api("search_text?q=" + encodeURIComponent(q));
    $("#txtRes").innerHTML = `<div class="muted">Mapped to AudioSet classes: ${r.mapped_classes.map(c => esc(c.name)).join(", ") || "none"}. This is label search, not a CLAP audio-text model.</div>` +
      (r.intervals.length ? r.intervals.map(i => `<div class="cap-row"><span><button class="round mini" data-t0="${i.start}" data-t1="${i.end}">▶</button> ${fmtT(i.start)}–${fmtT(i.end).slice(3)}</span>
        <span>${esc(i.best_class)} <span class="tierb">score ${i.score}</span></span></div>`).join("") : '<div class="muted">No matching interval above 0.10.</div>');
    $$("#txtRes button.mini").forEach(bt => bt.onclick = () => playSegment({id: "seg", start: +bt.dataset.t0, end: +bt.dataset.t1}));
  } catch (e) { $("#txtRes").textContent = e.message; }
};
$("#txtQ").addEventListener("keydown", e => { if (e.key === "Enter") $("#txtGo").click(); });

// ---------------------------------------------------------------- playback tools
let lastSeg = null, actx = null, srcNode = null, bp = null;
function bandRoute(seg) {
  try {
    if (!actx) { actx = new AudioContext(); srcNode = actx.createMediaElementSource(audio); bp = actx.createBiquadFilter(); bp.type = "bandpass"; }
    srcNode.disconnect(); bp.disconnect();
    const band = seg?.frequency?.band_5_95_hz;
    if ($("#chkBand").checked && band && band[1] > band[0]) {
      const f0 = Math.sqrt(Math.max(band[0], 40) * band[1]); bp.frequency.value = f0; bp.Q.value = Math.max(0.3, f0 / (band[1] - band[0]));
      srcNode.connect(bp); bp.connect(actx.destination);
    } else srcNode.connect(actx.destination);
    if (actx.state === "suspended") actx.resume();
  } catch (e) { /* WebAudio unavailable: plain playback */ }
}
playSegment = function (d) {
  lastSeg = d; S.sel = d; const c = +$("#selCtx").value;
  audio.playbackRate = +$("#selSpeed").value; bandRoute(d);
  audio.currentTime = Math.max(0, d.start - c); S.playToEnd = Math.min(S.duration || 1e9, d.end + c); audio.play();
};
$("#btnReplay").onclick = () => lastSeg && playSegment(lastSeg);
$("#btnFull").onclick = () => { S.sel = null; lastSeg = null; S.playToEnd = null; bandRoute(null); $("#btnZoomFit").click(); };
$("#selSpeed").onchange = () => audio.playbackRate = +$("#selSpeed").value;
$("#chkBand").onchange = () => bandRoute(lastSeg);

// ---------------------------------------------------------------- hooks into the existing workflow
const _loadResult0 = loadResult;
loadResult = async function () { await _loadResult0(); if (S.result) Intel.render(S.result); };
$("#btnAnalyse").onclick = async () => {
  $("#btnAnalyse").disabled = true; $("#btnOpen").disabled = true;
  try { await api("settings", {method: "POST", body: {analysis_mode: $("#selMode").value}});
    await api("analyse", {method: "POST", body: {}}); startPolling("analysed"); }
  catch (e) { alert(e.message); $("#btnOpen").disabled = false; }
};
$("#btnModels").addEventListener("click", async () => { const s = await api("settings"); $("#setMaps").checked = !!s.allow_online_maps; $("#setAsr").checked = s.transcribe_speech !== false; $("#selMode").value = s.analysis_mode || "deep"; });
$("#modelsDlg").addEventListener("close", () => { if ($("#modelsDlg").returnValue === "close") api("settings", {method: "POST", body: {allow_online_maps: $("#setMaps").checked, transcribe_speech: $("#setAsr").checked}}); });
api("settings").then(s => { if (s.analysis_mode) $("#selMode").value = s.analysis_mode; }).catch(() => {});
