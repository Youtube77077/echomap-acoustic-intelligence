"""Plain-language summary of a case (v1.0.8): where, which language, what was said, what each sound is.

Built from the full results after the geographic step; shown first in the app and the report. It only restates
results that exist elsewhere in the case (with their confidence); it never adds new claims.
"""
from __future__ import annotations

SKIP_CATS = {"language", "buildings_acoustics", "unknown"}
TIER_WORD = {"confirmed": "confirmed", "probable": "probable", "possible": "possible", "unclassified": "unclassified"}


def _pct(x) -> str:
    return f"{round(100 * float(x or 0))}%"


def _t(s: float) -> str:
    m, sec = divmod(float(s), 60)
    return f"{int(m):02d}:{sec:04.1f}"


def describe(r: dict) -> str:
    """One plain sentence for an inventory record."""
    label = r.get("label", "")
    spec = r.get("specific") or ""
    ev = r.get("evidence") or {}
    parts = [label]
    vt = ev.get("vehicle_type")
    if vt and vt.get("top"):
        alt = ", ".join(f"{k} {_pct(v)}" for k, v in list(vt.get("probs", {}).items())[1:3])
        if vt.get("confident", vt["p"] >= 0.4):
            parts.append(f"vehicle type: most likely {vt['top']} ({_pct(vt['p'])}; also {alt})")
        else:
            parts.append(f"vehicle type uncertain (best guess {vt['top']} {_pct(vt['p'])}, {alt})")
    if r.get("common_name"):
        parts.append(f"species: {r['common_name']}" + (f" ({r['scientific_name']})" if r.get("scientific_name") else ""))
    elif spec and spec not in label:
        parts.append(spec)
    if r.get("season_check"):
        parts.append(r["season_check"])
    cands = [c["label"] for c in (r.get("candidates") or [])[:2] if c.get("label") and c["label"] != label]
    if cands and not vt and not r.get("common_name"):
        parts.append("or: " + ", ".join(cands))
    return " — ".join(parts)


def build(result: dict) -> dict:
    inv = result.get("inventory") or []
    geo = result.get("geo") or {}
    out: dict = {}
    fps = {f.get("record"): f for f in geo.get("footprints") or [] if f.get("record")}
    unloc = {u["record"]: u["reason"] for u in geo.get("unlocated") or []}

    def where_found(rid):
        """Where in the world this kind of sound/species/language occurs (its map clue), or why it has none."""
        f = fps.get(rid)
        if f:
            names = [c["name"] for c in f.get("countries", [])[:6]]
            return {"footprint": f["id"], "area_km2": f.get("area_km2"), "countries": names,
                    "text": "Found in: " + ", ".join(names) + (" and more" if len(f.get("countries", [])) > 6 else "")}
        if rid in unloc:
            return {"footprint": None, "text": unloc[rid]}
        return None
    # --- where
    cands = geo.get("candidates") or []
    places = (geo.get("likely_places") or {}).get("places") or []
    used = [it for it in geo.get("evidence_items") or [] if it.get("used") and it.get("group") != "population"]
    where = {"status": geo.get("status") or "not_run", "clues": [it.get("label") for it in used][:8]}
    if cands:
        c = cands[0]
        where["headline"] = f"Most likely area: {c['name']} (within about {c['uncertainty_radius_km']:g} km)"
        where["alternatives"] = [f"{x['name']} (±{x['uncertainty_radius_km']:g} km)" for x in cands[1:4]]
    elif geo.get("status") == "no_estimate":
        weak = geo.get("weak_candidates") or []
        where["headline"] = "No reliable location estimate"
        where["detail"] = geo.get("summary", "")
        if weak:
            where["alternatives"] = [f"weak hint: {x['name']}" for x in weak[:3]]
    else:
        where["headline"] = "Location step not run"
    places = [p for p in places if float(p.get("share") or 0) >= 0.05]   # a 1% "place" is noise, not a finding
    if places:
        where["places"] = [f"{p['name']} ({_pct(p.get('share'))})" for p in places[:5]]
    tl = result.get("transit_lines")
    if tl and tl.get("lines"):
        b = tl["lines"][0]
        where["transit"] = {"line": b["line"], "operator": b["operator"], "city": b["city"], "country": b["country"],
                            "stops_heard": b["stops_heard"], "lat": b["lat"], "lon": b["lon"],
                            "others": [f"{x['line']} ({x['operator']}, {x['city']})" for x in tl["lines"][1:5]],
                            "text": f"{b['line']} — {b['operator']}, {b['city']} ({b['country']}); stop(s) heard: "
                                    f"{', '.join(b['stops_heard'])} at {b['lat']:.4f}, {b['lon']:.4f}"}
    out["where"] = where
    # --- languages
    langs = {}
    for r in inv:
        if r.get("language_code"):
            k = r["language_code"]
            if k not in langs or r["confidence"] > langs[k]["confidence"]:
                langs[k] = {"name": r.get("language_name") or k, "code": k, "confidence": r["confidence"],
                            "source": r.get("model", ""), "start": r["start"], "end": r["end"],
                            "where": where_found(r.get("id"))}
    out["languages"] = sorted(langs.values(), key=lambda d: -d["confidence"])
    # --- speech
    tr = result.get("transcripts") or {}
    segs = [s for s in tr.get("segments") or [] if s.get("quality") in ("clear", "uncertain")]
    out["speech"] = {"segments": len(segs), "clear": sum(1 for s in segs if s["quality"] == "clear"),
                     "lines": [{"start": s["start"], "end": s["end"], "text": s["text"], "quality": s["quality"],
                                "language": s.get("language")} for s in segs[:10]],
                     "not_run_reason": None if tr else "Speech transcription did not run (see Capabilities)."}
    skipped = (tr.get("stats") or {}).get("skipped_for_time") or []
    if skipped:
        out["speech"]["note"] = f"{len(skipped)} weaker speech segment(s) were not transcribed (time budget)."
    # --- sounds (plain descriptions, strongest first, one line per distinct label)
    seen, sounds = set(), []
    # sound sources the models recognised first; DSP measurements (wind gusts, periodic ticks) after them
    for r in sorted(inv, key=lambda r: (r.get("kind") in ("measurement", "derived"), -float(r.get("confidence") or 0))):
        if r.get("category") in SKIP_CATS or r.get("tier") == "unclassified":
            continue
        key = (r.get("label"), (r.get("evidence") or {}).get("vehicle_type", {}).get("top"), r.get("common_name"))
        if key in seen:
            continue
        seen.add(key)
        n_same = sum(1 for x in inv if x.get("label") == r.get("label"))
        sounds.append({"id": r.get("id"), "image": r.get("image"), "where": where_found(r.get("id")), "category": r.get("category_title") or r.get("category"),
                       "text": describe(r), "start": r["start"], "end": r["end"], "confidence": r["confidence"],
                       "tier": r.get("tier"), "times": n_same,
                       "when": f"{_t(r['start'])}–{_t(r['end'])}" + (f" (+{n_same - 1} more)" if n_same > 1 else "")})
        if len(sounds) >= 14:
            break
    out["sounds"] = sounds
    # --- signal facts
    sig = result.get("signal") or {}
    facts = []
    m = sig.get("mains") or {}
    if m.get("detected"):
        facts.append(f"Mains hum at {m.get('nominal_hz')} Hz (electrical grid of the place where it was recorded or "
                     f"of the playback device)")
    rv = (sig.get("reverb") or {})
    if rv.get("rt60_s"):
        facts.append(f"Reverberation RT60 ≈ {rv['rt60_s']} s")
    sc = (result.get("scene") or {}).get("top") or []
    if sc:
        facts.append("Scene: " + ", ".join(f"{s['scene']} ({_pct(s['score'])})" for s in sc[:3]))
    out["facts"] = facts
    return out
