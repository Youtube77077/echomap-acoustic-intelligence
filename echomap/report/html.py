"""Self-contained forensic HTML report (also the source for the PDF export)."""
from __future__ import annotations

import html
import json

from .. import config
from ..detect.taxonomy import CATEGORIES
from . import images

E = html.escape


def fmt_t(t) -> str:
    if t is None:
        return "-"
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:06.3f}"


def _kv(rows) -> str:
    return "<table class='kv'>" + "".join(f"<tr><th>{E(str(k))}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def _tbl(head, rows) -> str:
    h = "".join(f"<th>{E(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='grid'><thead><tr>{h}</tr></thead><tbody>{b or '<tr><td colspan=99><i>None</i></td></tr>'}</tbody></table>"


def _j(o) -> str:
    return f"<pre>{E(json.dumps(o, indent=1, ensure_ascii=False)[:6000])}</pre>"


CSS = """
body{font:13px/1.45 -apple-system,Segoe UI,Arial,sans-serif;color:#111;background:#fff;margin:24px auto;max-width:1100px;padding:0 16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;border-bottom:2px solid #333;padding-bottom:3px;margin-top:28px}
h3{font-size:14px;margin:16px 0 6px} .muted{color:#555} .warn{background:#fff6d6;border:1px solid #e6c64a;padding:8px 10px;margin:10px 0}
table{border-collapse:collapse;width:100%;margin:6px 0 10px} th,td{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}
table.kv th{width:240px;background:#f3f3f3} table.grid thead th{background:#e9e9e9} pre{white-space:pre-wrap;font-size:11px;background:#f6f6f6;padding:6px}
.t-confirmed{color:#0a6b2e;font-weight:600}.t-probable{color:#1d4f9c;font-weight:600}.t-possible{color:#8a5a00}.t-unclassified{color:#666}
img{max-width:100%} code{font-size:12px}
"""


def render(case) -> str:
    r = case.result
    meta = case.meta or {}
    pa = meta.get("primary_audio", {})
    inv = r.get("inventory", [])
    geo = r.get("geo", {})
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>EchoMap report: {E(meta.get('file_name', ''))}</title>"
           f"<style>{CSS}</style></head><body>"]
    out.append(f"<h1>EchoMap forensic acoustic report</h1><div class='muted'>{E(config.APP_NAME)} {E(config.APP_VERSION)}, "
               f"generated {E(case.history.steps[-1]['time_utc'] if case.history.steps else '')} UTC</div>")
    out.append("<div class='warn'><b>Reading this report:</b> model scores are <b>not calibrated probabilities</b>. "
               "Species, vehicles and languages are <b>model predictions</b> unless marked as measurements. Geographic "
               "scores are heuristic relative plausibilities. Nothing here identifies people.</div>")
    # 1-3
    out.append("<h2>1-3. File information, integrity and technical metadata</h2>")
    out.append(_kv([("File name", E(meta.get("file_name", ""))), ("SHA-256 (original)", f"<code>{E(case.sha256 or '')}</code>"),
                    ("Size", f"{meta.get('file_size_bytes', 0):,} bytes"), ("Container", E(f"{meta.get('container')} ({meta.get('container_long')})")),
                    ("Duration", f"{meta.get('duration_s') or 0:.3f} s"), ("Audio codec", E(f"{pa.get('codec')} ({pa.get('codec_long')})")),
                    ("Sample rate", f"{pa.get('sample_rate')} Hz"), ("Channels / layout", E(f"{pa.get('channels')} / {pa.get('layout')}")),
                    ("Sample format / bit depth", E(f"{pa.get('sample_format')} / {pa.get('bits_per_raw_sample') or pa.get('bits_per_sample')} bit")),
                    ("Bitrate", E(str(pa.get("bit_rate") or meta.get("bit_rate")))),
                    ("Video", "yes (audio extracted by stream copy)" if meta.get("is_video") else "no"),
                    ("Device information (metadata)", E(json.dumps(meta.get("device") or {}, ensure_ascii=False)) or "none"),
                    ("Embedded location (metadata claim)", E(json.dumps(meta.get("embedded_location"))) if meta.get("embedded_location") else "none"),
                    ("Container tags", E(json.dumps(meta.get("container_tags") or {}, ensure_ascii=False)[:800]))]))
    if r.get("extraction"):
        out.append("<h3>Audio extraction</h3>" + _kv([(k, E(str(v))) for k, v in r["extraction"].items()]))
    out.append("<h3>Analysis copy</h3>" + _kv([(k, E(str(v))) for k, v in (r.get("analysis_copy") or {}).items()]))
    # 4 overall environment
    sc = r.get("scene", {})
    out.append("<h2>4. Overall acoustic environment</h2>")
    out.append(f"<p>Indoor/outdoor: <b>{E(sc.get('indoor_outdoor', {}).get('estimate', 'n/a'))}</b>. Top scene candidates: "
               + ", ".join(f"{E(t['scene'])} ({t['score']})" for t in sc.get("top", [])) + "</p>")
    out.append(f"<p class='muted'>{E(sc.get('method', ''))} Not inferable: {E('; '.join(sc.get('not_inferable', [])))}</p>")
    # 5-12 inventory
    summ = r.get("inventory_summary", {})
    out.append(f"<h2>5. Complete detection inventory ({len(inv)} records)</h2>")
    out.append(_tbl(["Category", "Count", "Confirmed", "Probable", "Possible", "Unclassified"],
                    [[E(v["title"]), v["count"], v["tiers"]["confirmed"], v["tiers"]["probable"], v["tiers"]["possible"],
                      v["tiers"]["unclassified"]] for k, v in summ.items() if v["count"]]))
    section = {"birds_wildlife": "6. Every detected bird and animal", "vehicles": "7. Every detected vehicle",
               "engines_machinery": "8. Every detected engine and machine", "language": "9. Every detected language",
               "road_infrastructure": "10. Infrastructure sounds: road", "bells_alarms": "10. Infrastructure sounds: bells and alarms",
               "railway": "10. Infrastructure sounds: railway", "public_transport": "10. Infrastructure sounds: public transport",
               "electronic_signals": "10. Infrastructure sounds: electronic signals"}
    for key, title in CATEGORIES:
        recs = [x for x in inv if x["category"] == key]
        if not recs:
            continue
        out.append(f"<h3>{E(section.get(key, '11. ' + title))} ({len(recs)})</h3>")
        rows = []
        for x in recs:
            extra = ""
            if x.get("scientific_name"):
                extra = f"<br><i>{E(x['scientific_name'])}</i>, {E(x.get('taxon_class') or '')}; call/song: not supported"
            if x.get("identification_detail"):
                extra += f"<br><span class='muted'>{E(x['identification_detail'])}</span>"
            if x.get("directly_measured", {}).get("firing_frequency_hz"):
                extra += f"<br>Measured firing frequency ≈ {x['directly_measured']['firing_frequency_hz']} Hz"
            cands = "; ".join(f"{E(c['label'])}" + (f" {c['score']:.2f}" if c.get("score") is not None else "") for c in x.get("candidates", [])[:4])
            fq = x.get("frequency") or {}
            rows.append([x["id"], f"{E(x['label'])}<br><b>{E(x.get('specific') or '')}</b>{extra}", fmt_t(x["start"]), fmt_t(x["end"]),
                         f"{x['duration']:.2f}s", f"{x['confidence']:.2f}", f"<span class='t-{x['tier']}'>{x['tier']}</span>",
                         E(f"peak {fq.get('peak_hz', '-')} Hz, band {fq.get('band_5_95_hz', '-')}") if fq else "-",
                         E(x.get("model", "")), cands or "-",
                         E(x.get("geographic_relevance", "")), E(x.get("limitations", ""))])
        out.append(_tbl(["ID", "Detection", "Start", "End", "Dur", "Conf", "Tier", "Frequency", "Model", "Alternatives",
                         "Geographic relevance", "Limitations"], rows))
    out.append("<h2>12. Event timeline</h2>")
    out.append(_tbl(["Time", "End", "ID", "Category", "Detection", "Conf", "Tier"],
                    [[fmt_t(x["start"]), fmt_t(x["end"]), x["id"], E(x["category_title"]), E(x["label"] + (" | " + x["specific"] if x.get("specific") else "")),
                      f"{x['confidence']:.2f}", x["tier"]] for x in sorted(inv, key=lambda q: q["start"])]))
    # 13-14 spectrogram, frequency
    out.append("<h2>13-14. Spectrogram and frequency analysis</h2>")
    try:
        view = case.spectrogram(t0=0, t1=len(case.x) / case.sr, fmin=0, fmax=min(case.sr / 2, 16000), scale="linear",
                                width=1400, height=360)
        out.append(f"<img src='{images.spectrogram_png(view)}' alt='spectrogram'><div class='muted'>0-{view['fmax']:.0f} Hz linear, "
                   f"n_fft {view['n_fft']}, dB range {view['db_min']}..{view['db_max']} dBFS</div>")
    except Exception as e:  # noqa: BLE001
        out.append(f"<p>Spectrogram unavailable: {E(str(e))}</p>")
    sig = r.get("signal", {})
    out.append("<h3>Signal statistics</h3>" + _j(sig.get("summary")))
    out.append("<h3>SNR, reverberation, periodicity, mains hum</h3>" + _kv([
        ("SNR estimate", E(json.dumps(sig.get("snr")))), ("Reverberation", E(json.dumps(sig.get("reverb")))),
        ("Periodicity (onset autocorrelation)", E(json.dumps(sig.get("periodicity")))),
        ("Mains hum / ENF", E(json.dumps({k: v for k, v in (sig.get("mains") or {}).items() if k != "enf_trace"}))),
        ("Periodic patterns", E(json.dumps(sig.get("periodic_patterns"))))]))
    # 15-17 geography
    out.append("<h2>15-17. Geographic candidate areas, estimated coordinates, uncertainty radii</h2>")
    if not geo:
        out.append("<p>Geographic analysis not run.</p>")
    else:
        out.append(f"<p><b>{E(geo.get('summary', ''))}</b></p>")
        if geo.get("candidates"):
            out.append(f"<img src='{images.map_png(geo)}' alt='map'>")
            out.append(_tbl(["#", "Area", "Centre (lat, lon)", "Resolution", "Radius", "Strength", "Δ to best (nats)", "Relative mass"],
                            [[c["rank"], E(c["name"]), f"{c['centre']['lat']:.4f}, {c['centre']['lon']:.4f}", E(c["resolution"]),
                              f"{c['uncertainty_radius_km']:g} km", c["strength"], c["delta_to_best_nats"], c["relative_mass"]]
                             for c in geo["candidates"]]))
        lr = geo.get("local_refinement")
        if lr:
            out.append(f"<h3>Local refinement</h3><p class='muted'>{E(lr['note'])}</p>")
            out.append(_tbl(["#", "Area", "Centre", "Resolution", "Radius"],
                            [[c["rank"], E(c["name"]), f"{c['centre']['lat']:.4f}, {c['centre']['lon']:.4f}", E(c["resolution"]),
                              f"{c['uncertainty_radius_km']:g} km"] for c in lr["candidates"]]))
        out.append("<h3>20-21. Supporting and contradictory evidence per candidate</h3>")
        for c in geo.get("candidates", []):
            out.append(f"<p><b>#{c['rank']} {E(c['name'])}</b></p>")
            out.append(_tbl(["Direction", "Group", "Evidence", "Contribution (nats)", "Compatibility at centre", "Detail"],
                            [["supports", E(e["group"]), E(e["label"]), e["contribution_nats"], e["compatibility_at_centre"], E(e["detail"])] for e in c["supporting"]] +
                            [["contradicts", E(e["group"]), E(e["label"]), e["contribution_nats"], e["compatibility_at_centre"], E(e["detail"])] for e in c["contradicting"]]))
        out.append("<h3>Independence handling</h3><ul>" + "".join(f"<li>{E(n)}</li>" for n in geo.get("independence_notes", [])) + "</ul>")
        out.append(f"<h3>Method</h3><pre>{E(geo.get('method', ''))}</pre><p>Week basis: {E(geo.get('week_basis', ''))}</p>")
        if geo.get("species_profiles"):
            out.append("<h3>Species distribution and seasonality</h3>")
            out.append(_tbl(["Species", "Class", "Range (land fraction occ ≥ 0.25)", "Main countries", "Seasonality at best candidate", "Habitat"],
                            [[f"{E(p['common_name'])}<br><i>{E(p['scientific_name'])}</i>", E(p.get("taxon_class") or ""),
                              (p.get("distribution") or {}).get("land_fraction_occ_ge_0.25", "-"),
                              E(", ".join((p.get("distribution") or {}).get("main_countries", []))),
                              E((p.get("seasonality_at_best_candidate") or {}).get("pattern", "-")), E(p.get("habitat", ""))]
                             for p in geo["species_profiles"]]))
        if geo.get("online"):
            out.append("<h3>Online lookups (opt-in)</h3>" + _j(geo["online"]))
    # clue map + unlocated evidence
    if geo:
        cm = images.clue_map_png(geo)
        out.append("<h2>Clue map: geographic footprint of each clue</h2>")
        if cm:
            out.append(f"<img src='{cm}' alt='clue map'>")
        out.append(_tbl(["Clue", "Group", "Used", "Confidence", "Area km²", "Main countries", "Source", "Precision"],
                        [[E(f["label"]), E(f["group"]), "yes" if f["used"] else "no (low confidence)", f["confidence"],
                          f"{f['area_km2']:,}", E(", ".join(c["name"] for c in f["countries"][:5])), E(f.get("source") or ""),
                          E(f.get("precision") or "")] for f in geo.get("footprints", [])]))
        out.append("<h3>Unlocated evidence (not placeable on a map)</h3>" +
                   _tbl(["ID", "Detection", "Category", "Reason"], [[E(u["record"]), E(u["label"]), E(u["category"]), E(u["reason"])]
                                                                     for u in geo.get("unlocated", [])]))
    # intelligence sections
    if r.get("transcripts"):
        trs = r["transcripts"]
        out.append("<h2>Speech transcripts</h2><p class='muted'>Local Whisper-base transcription of every speech candidate, "
                   "including faint background speech (enhanced). Quality labels: clear / uncertain / likely spurious. "
                   "Machine transcripts can be wrong or invented; verify by listening.</p>" +
                   _tbl(["Start", "End", "Quality", "Language", "Text", "No-speech p", "Avg logprob", "Level dBFS", "Enhanced"],
                        [[fmt_t(t["start"]), fmt_t(t["end"]), E(t["quality"]), E(t["language"]), E(t["text"]), t["no_speech_prob"],
                          t["avg_logprob"], t["level_dbfs"], "yes" if t["enhanced"] else "no"] for t in trs["segments"]]))
    if r.get("integrity"):
        it = r["integrity"]
        out.append("<h2>Forensic integrity checks</h2>" + ("<ul>" + "".join(f"<li>{E(f)}</li>" for f in it["summary_flags"]) + "</ul>"
                   if it["summary_flags"] else "<p>No integrity flags raised.</p>") + _j({k: v for k, v in it.items() if k != "summary_flags"}))
    if r.get("bells"):
        out.append("<h2>Bell analysis and Dove's Guide matching</h2>" + _j(r["bells"]))
    if r.get("references"):
        out.append("<h2>Reference recording matches</h2>" + _j(r["references"]))
    if r.get("relations"):
        rel = r["relations"]
        out.append("<h2>Event relationships</h2>" + _tbl(["Pattern", "Explanation", "Records", "Status"],
                   [[E(h["pattern"]), E(h["explanation"]), E(", ".join(h["records"])), E(h["status"])] for h in rel["hypotheses"]]) +
                   _tbl(["Repeated sound", "Count", "Mean interval (s)", "Regular"],
                        [[E(p["label"]), p["count"], p["mean_interval_s"], "yes" if p["regular"] else "no"] for p in rel["repetitions"]]))
    if r.get("discovery"):
        out.append("<h2>Unknown-sound discovery</h2>" + _tbl(["Time", "Family", "Novelty", "Description", "Closest labels", "Disagreements"],
                   [[fmt_t(u["start"]), E(u.get("family", "")), u["novelty"], E(u["description"]),
                     E(", ".join(f"{l['label']} {l['score']}" for l in u["closest_model_labels"][:3])), E("; ".join(u["disagreements"]))]
                    for u in r["discovery"]["events"]]))
    if r.get("spatial"):
        out.append("<h2>Direction cues</h2>" + _j(r["spatial"]))
    if r.get("capabilities"):
        cap = r["capabilities"]
        out.append(f"<h2>Capabilities executed in this analysis (mode: {E(cap['mode_title'])})</h2>" +
                   _tbl(["Capability", "Status", "Class", "Reason"], [[E(i["name"]), E(i["status"]), E(i.get("class_title", "")),
                                                                         E(i.get("reason", ""))] for i in cap["items"]]))
    # 18 time
    t = r.get("time", {})
    out.append("<h2>18. Recording-time findings</h2>")
    out.append(_kv([("File-system times (this copy)", E(json.dumps(t.get("file_system_times")))),
                    ("Embedded metadata times (claims)", E(json.dumps(t.get("embedded_metadata_times")))),
                    ("User-provided", E(json.dumps(t.get("user_provided")))), ("Timezone", E(str(t.get("user_timezone")))),
                    ("Estimated", E(json.dumps(t.get("estimated"), ensure_ascii=False)[:3000])),
                    ("Verified external time evidence", E(json.dumps(t.get("verified_external")))), ("Summary", E(t.get("summary", "")))]))
    # 19 weather
    out.append("<h2>19. Weather-related observations</h2>")
    out.append("<h3>Direct acoustic observations</h3>" + _j(r.get("acoustic_weather")))
    if r.get("weather_external"):
        out.append("<h3>External record (Open-Meteo, opt-in)</h3>" + _j(r["weather_external"]))
    # 22-24 models, sources, limitations
    out.append("<h2>22. Model information</h2>")
    from ..models.registry import status
    st = status(case.settings.models_dir)
    out.append(_tbl(["Model", "Installed", "Source", "Licence", "Published accuracy", "Limitations"],
                    [[E(m["name"]), "yes" if m["installed"] else "no", E(m["source"]), E(m["licence"]), E(m["published_accuracy"]),
                      E(m["limitations"])] for m in st.values()]))
    out.append("<h2>23. External sources and datasets</h2><ul>"
               "<li>AudioSet ontology (Google, CC BY-SA 4.0)</li><li>BirdNET+ V3.0 and Geomodel (Cornell Lab of Ornithology / TU Chemnitz)</li>"
               "<li>Unicode CLDR territoryInfo (Unicode License v3): language population shares</li>"
               "<li>Natural Earth 1:50m boundaries (public domain) via world-atlas</li><li>GeoNames cities15000 (CC BY 4.0)</li>"
               "<li>Mains frequency per country: Wikipedia 'Mains electricity by country' (transcribed)</li></ul>")
    out.append("<h2>24. Confidence and limitations</h2><ul>"
               "<li>Tiers: <b>confirmed</b> = a deterministic measurement, or model score ≥ 0.70 corroborated by an independent detector; "
               "<b>probable</b> ≥ 0.50; <b>possible</b> ≥ listing threshold "
               f"({case.settings.list_threshold}); <b>unclassified</b> = energy without a classification.</li>"
               "<li>Scores below the listing threshold are not listed but are kept in case.json (full AudioSet frame-score matrix).</li>"
               "<li>Overlapping sounds are not separated (no source separation), so a loud source can mask quieter ones.</li>"
               "<li>Vehicle make/model, speaker identity, temperature and wind speed are not determined.</li>"
               "<li>Geographic output depends on detection quality and on range and language datasets. It is never proof of location.</li></ul>")
    # 25 history
    out.append("<h2>25. Full processing history</h2>")
    out.append(_tbl(["Time (UTC)", "Step", "Detail", "Duration", "Status", "Parameters"],
                    [[E(s["time_utc"]), E(s["step"]), E(s.get("detail", "")), f"{s.get('duration_s', '')}", E(s.get("status", "")),
                      E(json.dumps(s.get("params", {}), ensure_ascii=False)[:400])] for s in case.history.steps]))
    out.append(f"<p class='muted'>Environment: {E(json.dumps(case.history.environment))}</p></body></html>")
    return "".join(out)
