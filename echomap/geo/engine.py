"""Probabilistic geographic inference.

Every evidence item yields a per-location log-likelihood ratio (relative to its land average):
    l_i(x) = log( c * L_i(x) + (1 - c) * mean(L_i) ) - log(mean(L_i))
where c is the detection's model confidence and L_i(x) is how compatible location x is with the
observation (e.g. species occurrence from the range model, the language's share of the population).
A wrong detection (c → 0) therefore contributes nothing.

Items are grouped into *independence groups*. Within a group, items are assumed to be positively
correlated (several birds share a habitat, repeated speech comes from the same people), so the group
score is sum(l_i) / sqrt(n) instead of a plain sum, then clipped to ±GROUP_CLIP nats so no single
group can dominate. Floors on L_i keep any single observation from eliminating a region (captive or
escaped animals, tourists, imported equipment).

The output is a *heuristic relative plausibility*, NOT a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import numpy as np

from . import places
from .gazetteer import cities_in_region, nearest_city
from .grid import countries, grid, haversine_km, local_grid

GROUP_CLIP = 6.0
GROUP_DISCOUNT_EXP = 0.25      # n clues in one group count as n**(1-exp) independent clues (0.5 = sqrt(n) discount)
OUT_OF_RANGE_FLOOR = 0.02      # captive / escaped / vagrant / transported animals
LANGUAGE_FLOOR = 0.005         # tourists, migrants, broadcast media
RESOLUTIONS_KM = [1, 5, 10, 25, 50, 100]
UNCORROBORATED_MIN_CONF = 0.6  # species without simultaneous AudioSet animal activity need this confidence
LANG_MIN_CONF = 0.5            # weaker language guesses (bells or trams read as speech) are shown but not used
LANG_ALIASES = {"nn": "no", "jw": "jv"}   # Whisper codes -> CLDR codes
SPECIES_MIN_CONF = 0.4         # species predictions below this are listed but not used for geography
MIN_INFORMATION_RATIO = 1.5
# Radius calibration: on the iNaturalist evaluation (docs/eval/geo.json, 43 estimates, 2026-11-12) the true location fell
# inside the best candidate's raw radius only 43% of the time. Scaling by the 68th percentile of error/radius (1.74)
# makes the best circle a ~68% region on that data. Small sample: re-estimated whenever the evaluation is re-run.
RADIUS_CALIBRATION = 1.75    # evidence must shrink the plausible land area at least 2x to report candidates


@dataclass
class Points:
    lat: np.ndarray
    lon: np.ndarray
    iso: np.ndarray
    coastal: np.ndarray
    mains: np.ndarray
    area_km2: np.ndarray

    @classmethod
    def global_land(cls):
        g = grid()
        return cls(g.lat, g.lon, g.iso, g.coastal, g.mains, g.area_km2)

    @classmethod
    def local(cls, lat, lon, radius_km, step_km):
        la, lo = local_grid(lat, lon, radius_km, step_km)
        g = grid()
        # attribute each local point to the nearest global land cell (country, coast, mains)
        r = g.res
        key = {(round(float(a) / r), round(float(b) / r)): i for i, (a, b) in enumerate(zip(g.lat - r / 2, g.lon - r / 2))}
        idx = []
        for a, b in zip(la, lo):
            k = (int(np.floor(a / r)), int(np.floor(b / r)))
            idx.append(key.get(k, -1))
        idx = np.array(idx)
        ok = idx >= 0
        la, lo, idx = la[ok], lo[ok], idx[ok]
        return cls(la, lo, g.iso[idx], g.coastal[idx], g.mains[idx], np.full(len(la), step_km ** 2, np.float32))

    @classmethod
    def at(cls, lat, lon, area_km2: float = 1.0):
        """Points at given coordinates (country, coast and mains from the global cell containing each point)."""
        g = grid()
        r = g.res
        key = {(round(float(a) / r), round(float(b) / r)): i for i, (a, b) in enumerate(zip(g.lat - r / 2, g.lon - r / 2))}
        la, lo = np.asarray(lat, np.float32), np.asarray(lon, np.float32)
        idx = np.array([key.get((int(np.floor(a / r)), int(np.floor(b / r))), -1) for a, b in zip(la, lo)])
        ok = idx >= 0
        from .topo import country_at
        iso = np.array([country_at(float(a), float(b)) or g.iso[i] for a, b, i in zip(la[ok], lo[ok], idx[ok])], dtype=object)
        c = countries()
        mains = np.array([str(c.get(i, {}).get("mains_hz", 50)) if i != "JP" else g.mains[j] for i, j in zip(iso, idx[ok])], dtype=object)
        p = cls(la[ok], lo[ok], iso, g.coastal[idx[ok]], mains, np.full(int(ok.sum()), area_km2, np.float32))
        return p, ok

    def language_share(self, code: str) -> np.ndarray:
        c = countries()
        lut = {k: v.get("languages", {}).get(code, 0.0) for k, v in c.items()}
        return np.array([lut.get(i, 0.0) for i in self.iso], np.float32)


def build_footprints(pts: "Points", items: list[dict]) -> list[dict]:
    """Encode each clue's geographic footprint as a bitset over the global 0.5° raster (720 x 360)."""
    import base64
    rows = np.clip(np.round((89.75 - pts.lat) / 0.5).astype(int), 0, 359)
    cols = np.clip(np.round((pts.lon + 179.75) / 0.5).astype(int), 0, 719)
    flat = rows * 720 + cols
    out = []
    for i, it in enumerate(items):
        fp = np.asarray(it.get("fp"), bool)
        if fp.shape != pts.lat.shape or not fp.any():
            continue
        bits = np.zeros(720 * 360, np.uint8)
        bits[flat[fp]] = 1
        area = float(pts.area_km2[fp].sum())
        ctry: dict[str, float] = {}
        for iso, a in zip(pts.iso[fp], pts.area_km2[fp]):
            ctry[iso] = ctry.get(iso, 0.0) + float(a)
        top = sorted(ctry.items(), key=lambda kv: -kv[1])[:10]
        if it.get("shape") == "countries":
            isos = sorted(set(pts.iso[fp].tolist()) - {""})
            shape = {"kind": "countries", "iso": isos, "numeric": [countries().get(c, {}).get("numeric") for c in isos]}
        elif it.get("points"):
            shape = {"kind": "points", "points": it["points"]}
        else:
            shape = {"kind": "raster"}
        out.append({"id": f"F{i + 1}", "label": it["label"], "group": it["group"], "record": it["record"], "shape": shape,
                    "used": bool(it.get("used", True)), "confidence": round(float(it["c"]), 3),
                    "source": it.get("source"), "precision": it.get("precision"), "note": it.get("note", ""),
                    "area_km2": round(area), "countries": [{"iso2": k, "name": countries().get(k, {}).get("name", k),
                                                             "area_km2": round(v)} for k, v in top],
                    "bits": base64.b64encode(np.packbits(bits).tobytes()).decode()})
    return out


def unlocated_reason(r: dict) -> str:
    if r.get("scientific_name"):
        return "Species not in the range-model label set"
    if r["category"] in ("vehicles", "engines_machinery", "industrial", "human_nonspeech", "music", "weather", "ambience"):
        return "Occurs worldwide: not location-specific"
    if r["tier"] == "unclassified":
        return "Unidentified sound: no geographic meaning without identification"
    return "No validated link between this sound and specific places"


def birdnet_week(month: int, day: int | None = None) -> int:
    return (month - 1) * 4 + (min(4, (day - 1) // 7 + 1) if day else 2)


def context_weeks(ctx: dict) -> tuple[list[int] | None, str]:
    d = (ctx or {}).get("date") or {}
    mode = d.get("mode", "unknown")
    try:
        if mode in ("exact", "approx_date") and d.get("value"):
            y, m, dd = [int(v) for v in d["value"][:10].split("-")]
            w = birdnet_week(m, dd)
            ws = [w] if mode == "exact" else sorted({max(1, w - 2), w, min(48, w + 2)})
            return ws, f"week(s) {ws} from user-provided date"
        if mode == "month_year" and d.get("month"):
            m = int(d["month"])
            return [birdnet_week(m, 1), birdnet_week(m, 8), birdnet_week(m, 15), birdnet_week(m, 22)], f"month {m} from user"
    except (ValueError, TypeError):
        pass
    return None, "date unknown, so species ranges use the maximum over the whole year"


MASKING_CATS = {"bells_alarms", "music", "vehicles", "engines_machinery", "railway", "public_transport", "industrial"}
MASKED_MIN_CONF = 0.85


def _species_usable(r: dict, records: list[dict]) -> bool:
    """Species detections during loud man-made sounds (bells, sirens, engines, trams, music) are often fake: church
    bells became American birds and tram squeal became whales in the soundscape benchmark. Such a detection without
    an AudioSet animal sound at the same time counts only when the model is very sure."""
    if r["confidence"] < SPECIES_MIN_CONF:
        return False
    masked = any(o.get("category") in MASKING_CATS and o.get("start", 0) < r.get("end", 0) and o.get("end", 0) > r.get("start", 0)
                 and o.get("confidence", 0) >= 0.2 for o in records)
    if not masked:
        return bool(r.get("corroborated_by")) or r["confidence"] >= UNCORROBORATED_MIN_CONF
    # under bells / engines / music: the iNatSounds head reads the same embedding and a weak generic "Bird" tag is
    # often the bell itself, so only a clear AudioSet animal sound (>= 0.5) or a very sure model counts
    by_id = {o.get("id"): o for o in records}
    strong = any((by_id.get(str(c).split(" ")[0]) or {}).get("confidence", 0) >= 0.5
                 for c in r.get("corroborated_by") or [] if "(AudioSet)" in str(c))
    return strong or r["confidence"] >= MASKED_MIN_CONF


def _language_backed(r: dict, code: str) -> bool:
    """A language counts for location only with a clear transcript in it, or very confident ID on clear speech
    (bells and engines were read as Korean / Nynorsk speakers in the soundscape benchmark)."""
    if "transcript" in str(r.get("model", "")).lower() or "speech-to-text" in str(r.get("model", "")).lower():
        return True
    if any(t.get("quality") == "clear" and LANG_ALIASES.get(t.get("language"), t.get("language")) == code
           for t in r.get("transcripts") or []):
        return True
    sp = float((r.get("evidence") or {}).get("speech_score_max") or 0)
    return r["confidence"] >= 0.8 and sp >= 0.5


@lru_cache(maxsize=1)
def species_ranges() -> dict:
    """GBIF country / month statistics per species (tools/build_species_ranges.py); empty when not bundled."""
    import gzip
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "species_ranges.json.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as g:
        return json.load(g).get("species", {})


@lru_cache(maxsize=1)
def xc_ranges() -> dict:
    """Xeno-canto recording sites (1-degree cells) and months per species (tools/xc_harvest.py); empty if absent."""
    import gzip
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "xc_ranges.json.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as g:
        return json.load(g).get("species", {})


@lru_cache(maxsize=1)
def _population_cells() -> dict:
    """Summed city population per 0.5° cell, spread to the 8 neighbouring cells (suburbs, smaller towns)."""
    from .gazetteer import _cities
    rows, lat, lon = _cities()
    pop = np.array([r[6] or 0 for r in rows], np.float64)
    ki = np.floor(lat / 0.5).astype(np.int64)
    kj = np.floor(lon / 0.5).astype(np.int64)
    cells: dict = {}
    for i, j, p in zip(ki, kj, pop):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = (int(i + di), int(j + dj))
                cells[k] = cells.get(k, 0.0) + p * (1.0 if di == dj == 0 else 0.25)
    return cells


def population_likelihood(pts: "Points") -> np.ndarray:
    cells = _population_cells()
    v = np.array([cells.get((int(np.floor(a / 0.5)), int(np.floor(b / 0.5))), 0.0) for a, b in zip(pts.lat, pts.lon)])
    ref = 500_000.0   # a cell with a city of ~500k (or several towns) counts as fully populated
    return np.clip(np.maximum(v / ref, 0.03), 0, 1).astype(np.float32)


def _llr(L: np.ndarray, c: float, weights: np.ndarray, mean: float | None = None) -> np.ndarray:
    if mean is None:
        mean = float((L * weights).sum() / weights.sum())
    mean = max(mean, 1e-9)
    return np.log(c * L + (1 - c) * mean) - np.log(mean)


class GeoEngine:
    def __init__(self, models_dir: str, history=None):
        self.models_dir = models_dir
        self.history = history
        self._occ_cache: dict = {}

    def occurrence(self, pts: "Points", weeks, idxs: list[int]) -> np.ndarray:
        """Geomodel occurrence (max over the given weeks) for species indices at points, cached per point set."""
        from ..models import birdnet as bnm
        key = (len(pts.lat), float(pts.lat[0]) if len(pts.lat) else 0.0, tuple(weeks or [-1]))
        cache = self._occ_cache.setdefault(key, {})
        todo = [i for i in idxs if i not in cache]
        if todo:
            # evaluate on a 1-degree lattice for the global grid (4x fewer model calls), then map back;
            # species ranges are smooth at this scale
            if len(pts.lat) > 20000:
                step = 2.0  # degrees; species ranges are smooth at this scale and the geomodel output layer is large
                la_i = np.floor(pts.lat / step).astype(np.int32)
                lo_i = np.floor(pts.lon / step).astype(np.int32)
                kl = la_i * 1000 + lo_i + 500
                uk, inv_idx = np.unique(kl, return_inverse=True)
                lat_i = np.floor_divide(uk, 1000)
                qlat = (lat_i.astype(np.float32) + 0.5) * step
                qlon = ((uk - lat_i * 1000 - 500).astype(np.float32) + 0.5) * step
            else:
                qlat, qlon, inv_idx = pts.lat, pts.lon, None
            occ = None
            for wk in (weeks or [-1]):
                o = bnm.geo_occurrence(self.models_dir, qlat, qlon, wk, todo)
                occ = o if occ is None else np.maximum(occ, o)
            if inv_idx is not None:
                occ = occ[inv_idx]
            for j, i in enumerate(todo):
                cache[i] = occ[:, j]
        return np.stack([cache[i] for i in idxs], axis=1)

    # ------------------------------------------------------------------ evidence builders
    def evidence_items(self, pts: Points, records: list[dict], mains: dict, ctx: dict, weeks,
                       extra: dict | None = None, include_all: bool = False, means: dict | None = None) -> list[dict]:
        items = []
        w = pts.area_km2.astype(np.float64)
        # --- biogeography (BirdNET species + geomodel)
        # Species gating: only predictions >= SPECIES_MIN_CONF are used as geographic evidence. Uncorroborated
        # predictions (no overlapping AudioSet animal activity) get half weight.
        sp = [r for r in records if r.get("scientific_name") and (include_all or r["confidence"] >= SPECIES_MIN_CONF)]
        best: dict[str, dict] = {}
        for r in sp:
            k = r["scientific_name"].lower()
            if k not in best or r["confidence"] > best[k]["confidence"]:
                best[k] = r
        if best:
            try:
                from ..models import birdnet as bnm
                gidx = bnm.geo_index(self.models_dir)
                keys = [k for k in best if k in gidx]
                if keys:
                    idxs = [gidx[k] for k in keys]
                    # warm the cache for every detected species at once (profiles reuse it)
                    all_sp = sorted({gidx[r["scientific_name"].lower()] for r in records
                                     if r.get("scientific_name") and r["scientific_name"].lower() in gidx})
                    self.occurrence(pts, weeks, all_sp)
                    if weeks:
                        self.occurrence(pts, None, all_sp) if len(pts.lat) > 20000 else None
                    occ = self.occurrence(pts, weeks, idxs)
                    for j, k in enumerate(keys):
                        r = best[k]
                        L = np.maximum(occ[:, j], OUT_OF_RANGE_FLOOR)
                        frac = float((occ[:, j] >= 0.25).astype(float) @ w / w.sum())
                        corr = bool(r.get("corroborated_by"))
                        c_eff = r["confidence"] if corr else 0.5 * r["confidence"]
                        # v1.0.9: bells, trams and engines produce false "species"; without an AudioSet animal sound at
                        # the same time a species is only used when the model is very sure
                        use = _species_usable(r, records)
                        items.append({"group": "biogeography", "label": f"{r['common_name']} ({r['scientific_name']})",
                                      "record": r["id"], "c": c_eff, "L": L, "fp": occ[:, j] >= 0.25,
                                      "used": use, "source": "BirdNET+ Geomodel V3.0.4",
                                      "precision": "2° range-model lattice (~220 km)",
                                      "note": f"range model: occurrence ≥ 0.25 over {frac*100:.1f}% of land area",
                                      "range_fraction": frac})
                for k in best:
                    if k not in gidx:
                        best[k]["geographic_relevance"] = "Species not in the geomodel label set, so no range evidence."
            except FileNotFoundError:
                gidx = {}
            # v1.0.9: GBIF country ranges for species the geomodel does not cover (iNatSounds insects, frogs, mammals)
            rng = species_ranges()
            xcr = xc_ranges()
            for k, r in best.items():
                if k in gidx:
                    continue
                xc = xcr.get(k)
                if xc and sum(c[2] for c in xc["cells"]) >= 5:   # Xeno-canto recording sites: 1-degree cells +- 1 cell
                    occ = {(a + da, b + db) for a, b, _ in xc["cells"] for da in (-1, 0, 1) for db in (-1, 0, 1)}
                    keyc = list(zip(np.floor(pts.lat).astype(int).tolist(), np.floor(pts.lon).astype(int).tolist()))
                    L = np.array([1.0 if kc in occ else OUT_OF_RANGE_FLOOR for kc in keyc], np.float32)
                    corr = bool(r.get("corroborated_by"))
                    use = _species_usable(r, records)
                    frac = float((L >= 0.5).astype(float) @ w / w.sum())
                    items.append({"group": "biogeography", "label": f"{r.get('common_name') or xc.get('en') or k} "
                                  f"({r['scientific_name']})", "record": r["id"], "c": r["confidence"] if corr else
                                  0.5 * r["confidence"], "L": L, "fp": L >= 0.5, "used": use,
                                  "source": "Xeno-canto recording sites", "precision": "1-degree cells (~110 km)",
                                  "note": f"recorded at {len(xc['cells'])} 1-degree cells ({frac * 100:.1f}% of land area)",
                                  "range_fraction": frac})
                    r["geographic_relevance"] = "Range from Xeno-canto recording sites."
                    continue
                if k not in rng:
                    continue
                g = rng[k]
                tot = max(1, sum(g["c"].values()))
                present = {cc for cc, n in g["c"].items() if n >= max(3, 0.002 * tot)}
                if not present:
                    continue
                L = np.where(np.isin(pts.iso, list(present)), 1.0, OUT_OF_RANGE_FLOOR).astype(np.float32)
                corr = bool(r.get("corroborated_by"))
                use = _species_usable(r, records)
                frac = float((L >= 0.5).astype(float) @ w / w.sum())
                items.append({"group": "biogeography", "label": f"{r.get('common_name') or k} ({r['scientific_name']})",
                              "record": r["id"], "c": r["confidence"] if corr else 0.5 * r["confidence"], "L": L,
                              "fp": L >= 0.5, "shape": "countries", "used": use,
                              "source": "GBIF occurrence statistics (countries)", "precision": "country level",
                              "note": f"recorded in {len(present)} countries ({frac * 100:.1f}% of land area), "
                                      f"{g['n']:,} GBIF occurrences", "range_fraction": frac})
                r["geographic_relevance"] = "Range from GBIF occurrence statistics (country level)."
        # --- season check for every species with a known date (GBIF monthly occurrence counts)
        month = (ctx or {}).get("date", {}).get("month")
        dv = (ctx or {}).get("date", {}).get("value")
        if not month and dv and len(str(dv)) >= 7:
            try:
                month = int(str(dv)[5:7])
            except ValueError:
                month = None
        if month:
            rng = species_ranges()
            for it in items:
                if it["group"] != "biogeography" or not it.get("record"):
                    continue
                rec = next((r for r in records if r["id"] == it["record"]), None)
                sk = (rec or {}).get("scientific_name", "").lower() if rec else ""
                g = rng.get(sk)
                xg = xc_ranges().get(sk)
                if xg and sum(xg["m"]) >= 30 and (not g or sum(g["m"]) < sum(xg["m"])):
                    g = {"m": xg["m"]}
                if g and sum(g["m"]) >= 30:
                    share = g["m"][int(month) - 1] / sum(g["m"])
                    rec["season_check"] = (f"{share * 100:.1f}% of GBIF records are from this month"
                                           + (" — unusual at this time of year" if share < 0.01 else ""))
                    if share < 0.01:
                        it["c"] *= 0.5
        # --- language
        langs: dict[str, dict] = {}
        for r in records:
            if r.get("language_code"):
                if r["language_code"] not in langs or r["confidence"] > langs[r["language_code"]]["confidence"]:
                    langs[r["language_code"]] = r
        for code, r in langs.items():
            code = LANG_ALIASES.get(code, code)
            share = pts.language_share(code)
            if share.max() <= 0:
                r["geographic_relevance"] = "Language not in the CLDR territory data, so no geographic evidence."
                continue
            L = np.maximum(share, LANGUAGE_FLOOR)
            items.append({"group": "language", "label": f"Language {r['language_name']} ({code})", "record": r["id"],
                          "c": r["confidence"], "L": L, "fp": share >= 0.15, "shape": "countries", "source": "Unicode CLDR territoryInfo",
                          "used": r["confidence"] >= LANG_MIN_CONF and _language_backed(r, code),
                          "precision": "country-level (language shares are national)",
                          "note": f"CLDR population share; spoken by ≥ 50% in {int((share >= 0.5).sum())} land cells' countries"})
        # --- electrical grid
        if mains.get("detected"):
            hz = str(mains["nominal_hz"])
            match = (pts.mains == hz).astype(np.float32)
            mixed = np.char.find(pts.mains.astype(str), "/") >= 0
            L = np.where(mixed, 0.5, np.where(match > 0, 0.97, 0.03)).astype(np.float32)
            rec = next((r for r in records if r["label"].startswith("Mains hum")), None)
            c = 0.85 if rec and rec["tier"] == "confirmed" else 0.5
            items.append({"group": "electrical_grid", "label": f"Mains hum {hz} Hz", "record": rec["id"] if rec else None,
                          "c": c, "L": L, "fp": L >= 0.5, "shape": "countries", "source": "Mains frequency per country (Wikipedia, transcribed)",
                          "precision": "country / grid level", "note": "Nominal grid frequency per country (Japan split east/west)"})
        # --- cultural infrastructure
        for r in records:
            names = set(r.get("evidence", {}).get("ontology_parents_detected", [])) | {r["label"]}
            if "Change ringing (campanology)" in names and r["confidence"] >= 0.2:
                L = np.where(pts.iso == "GB", 1.0, 0.08).astype(np.float32)
                items.append({"group": "bells", "label": "Change ringing (English-style bells)", "record": r["id"],
                              "c": r["confidence"], "L": L, "fp": L >= 0.5, "shape": "countries", "source": "Dove's Guide tower distribution",
                              "precision": "country level", "note": "Change-ringing towers concentrated in England"})
            pat = (r.get("evidence", {}).get("siren_pattern") or {}).get("pattern", "")
            # only sirens the sound model itself heard (DSP-only "siren-like" sweeps came from church bells)
            real_siren = (not str(r["label"]).startswith("Siren-like") and r["confidence"] >= 0.3 and
                          any("siren" in n.lower() for n in names))
            if pat and real_siren and ("wail" in pat or "yelp" in pat) and "two-tone" not in pat:
                am = np.array([countries().get(i, {}).get("continent") in ("NA", "SA", "OC") for i in pts.iso])
                items.append({"group": "cultural", "label": "Wail / yelp emergency siren", "record": r["id"],
                              "c": min(0.4, r["confidence"]), "L": np.where(am, 1.0, 0.5).astype(np.float32), "fp": am,
                              "shape": "countries", "source": "Siren-pattern regional practice (heuristic)",
                              "precision": "continental",
                              "note": "Weak: wail/yelp sirens are standard in North America and Australia, common elsewhere"})
            if "two-tone" in pat and real_siren:
                eu = np.array([countries().get(i, {}).get("continent") == "EU" for i in pts.iso])
                items.append({"group": "cultural", "label": "Two-tone emergency siren", "record": r["id"],
                              "c": min(0.6, r["confidence"]), "L": np.where(eu, 1.0, 0.45).astype(np.float32), "fp": eu, "shape": "countries",
                              "source": "Siren-pattern regional practice (heuristic)", "precision": "continental",
                              "note": "Weak: hi-lo sirens are common in Europe"})
        # --- physical environment
        for r in records:
            if r["label"] in ("Waves, surf", "Ocean") and r["confidence"] >= 0.2:
                L = np.where(pts.coastal, 1.0, 0.2).astype(np.float32)
                items.append({"group": "environment", "label": f"{r['label']} (coastal water)", "record": r["id"],
                              "c": min(0.8, r["confidence"]), "L": L, "fp": pts.coastal.copy(),
                              "source": "Natural Earth coastline (0.5° coastal cells)", "precision": "coastal strip ~55 km",
                              "note": "Coastal cells favoured. Large lakes, rivers and recordings also produce wave-like sounds."})
                break
        # --- point-location clues (v1.0.3): tram networks and spoken place names
        extra = extra or {}
        th = extra.get("tram")
        if th:
            nets = places.tram_networks()["networks"]
            if nets:
                L = places.tram_likelihood(pts, nets)
                items.append({"group": "transport_infrastructure", "label": "Tram / streetcar (hypothesis)", "record": th["record"],
                              "c": th["c"], "L": L, "fp": L >= 0.3,
                              "points": [{"name": n["name"], "cc": n["cc"], "lat": n["lat"], "lon": n["lon"], "r": n["extent_km"]} for n in nets],
                              "source": "OpenStreetMap railway=tram track (ODbL), " + str(len(nets)) + " networks",
                              "precision": "city level: kernel around each tram network",
                              "note": th["basis"] + "; tram cities favoured, larger networks slightly more"})
        tl = extra.get("transit_lines")
        if tl and tl.get("pins"):   # announced stops matched to public-transport lines: stop-level pins
            L = np.zeros(len(pts.lat), np.float64)
            for q in tl["pins"]:
                d = haversine_km(q["lat"], q["lon"], pts.lat, pts.lon)
                L = np.maximum(L, np.exp(-0.5 * (d / places.kernel_sigma(1.5, pts)) ** 2))
            best = tl["lines"][0] if tl.get("lines") else {}
            items.append({"group": "place_names", "label": f"Announced stop(s): {', '.join(h['stop'] for h in tl['heard'][:3])}",
                          "record": None, "c": float(tl["confidence"]), "L": np.maximum(L, 0.03).astype(np.float32),
                          "fp": L >= 0.3, "points": [{"name": q["name"], "cc": q["cc"], "lat": q["lat"], "lon": q["lon"], "r": 0.5}
                                                     for q in tl["pins"]],
                          "source": "Speech transcript + public-transport timetables (GTFS, Mobility Database)",
                          "precision": "stop level (pins at the announced stops)",
                          "note": f"{tl['basis']}; best line {best.get('line', '?')} ({best.get('operator', '')}, {best.get('city', '')}); "
                                  f"{tl['n_matching_lines']} line(s) match"})
        for m in extra.get("place_mentions") or []:
            L, cities = places.mention_likelihood(pts, m)
            items.append({"group": "place_names", "label": f"Spoken place name: {m['name']}", "record": m.get("record"),
                          "c": places.mention_confidence(m), "L": L, "fp": L >= 0.3,
                          "points": [{"name": c["name"], "cc": c["cc"], "lat": c["lat"], "lon": c["lon"], "r": c["radius_km"]} for c in cities],
                          "source": "Speech transcript + GeoNames cities (≥ 15,000 inhabitants)",
                          "precision": "city level; every city with this name is kept, weighted by size and language",
                          "note": (f"Heard '{m['segments'][0]['text'][:80]}'" + (f" after the cue '{m['cue']}'" if m["cue"] else "") +
                                   "; mentioning a place does not prove being there")})
        # --- where people are (v1.0.8). Speech, place names and transit sounds are made where people live, so these
        # clues get a population prior: without it a large, thinly populated country with a high language share
        # (Suriname for Dutch) outranks the densely populated core area (the Netherlands / Flanders).
        human = {"language", "place_names", "transport_infrastructure", "reference_recordings"}
        if any(it["group"] in human for it in items):
            items.append({"group": "population", "label": "People live here (speech / transport clues)", "record": None,
                          "c": 0.6, "L": population_likelihood(pts), "used": True,
                          "source": "GeoNames cities ≥ 15,000 inhabitants, summed per 0.5° cell (smoothed)",
                          "precision": "0.5° cells (~55 km)",
                          "note": "Human-made sounds are most likely where people live; thinly populated areas are "
                                  "down-weighted, never excluded"})
        # --- user context (prior, not evidence)
        loc = (ctx or {}).get("location") or {}
        mode = loc.get("mode", "unknown")
        if mode == "country" and loc.get("country"):
            items.append({"group": "user_context", "label": f"User: country {loc['country']}", "record": None, "c": 0.999,
                          "L": np.where(pts.iso == loc["country"], 1.0, 0.002).astype(np.float32), "note": "User-provided",
                          "shape": "countries"})
        elif mode == "region" and loc.get("country"):
            cs = cities_in_region(loc["country"], loc.get("region", ""))
            if cs:
                d = np.min([haversine_km(c["lat"], c["lon"], pts.lat, pts.lon) for c in cs[:60]], axis=0)
                L = np.maximum(np.exp(-0.5 * (d / 60.0) ** 2), np.where(pts.iso == loc["country"], 0.05, 0.001))
            else:
                L = np.where(pts.iso == loc["country"], 1.0, 0.002)
            items.append({"group": "user_context", "label": f"User: region {loc.get('region')} ({loc['country']})", "record": None,
                          "c": 0.999, "L": L.astype(np.float32), "note": "User-provided (region approximated by its cities)"})
        elif mode == "city" and loc.get("city"):
            c = loc["city"]
            d = haversine_km(c["lat"], c["lon"], pts.lat, pts.lon)
            sig = float(loc.get("city_radius_km", 12.0))
            L = np.maximum(np.exp(-0.5 * (d / sig) ** 2), 1e-4)
            items.append({"group": "user_context", "label": f"User: city {c.get('name')}", "record": None, "c": 0.999,
                          "L": L.astype(np.float32), "note": f"User-provided, Gaussian σ = {sig:g} km"})
        # --- bell towers (Dove's Guide match) and geotagged reference recordings
        bm = extra.get("bell_match") or {}
        cands = [c for c in (bm.get("all_compatible") or []) if c.get("lat") is not None]
        if cands:
            top_s = max(c["score"] for c in cands)
            L = np.zeros(len(pts.lat), np.float32)
            for c in cands[:200]:
                if c["score"] >= 0.3 * top_s:
                    d = haversine_km(c["lat"], c["lon"], pts.lat, pts.lon)
                    L = np.maximum(L, c["score"] / top_s * np.exp(-0.5 * (d / 25.0) ** 2))
            L = np.maximum(L, 0.02)
            items.append({"group": "bells", "label": f"Bell-ring match ({bm.get('observed_bells')} bells, "
                                                     f"{len(cands)} compatible Dove rings)",
                          "record": extra.get("bell_record"), "c": float(extra.get("bell_confidence", 0.4)), "L": L,
                          "fp": L >= 0.3, "source": "Dove's Guide (CC BY-SA 4.0) + EchoMap bell-partial analysis",
                          "points": [{"name": c.get("place") or c.get("dedication") or "tower", "cc": c.get("iso") or "",
                                      "lat": c["lat"], "lon": c["lon"], "r": 2.0} for c in cands[:200] if c["score"] >= 0.3 * top_s],
                          "precision": "tower-level points with 25 km kernels; many towers share bell notes",
                          "note": "Rings whose bell nominals match the recording within ~25 cents"})
        for rm in (extra.get("reference_geo") or []):
            d = haversine_km(rm["lat"], rm["lon"], pts.lat, pts.lon)
            city = rm.get("city")
            if rm.get("exact"):     # the same announcement / jingle recording: strong, city-level
                sig, c, fl = places.kernel_sigma(12.0, pts), 0.7, 0.05
            elif city:              # resembles a city transit recording: weak, city-level
                sig, c, fl = places.kernel_sigma(15.0, pts), float(min(0.35, 0.35 * rm["similarity"])), 0.1
            else:
                sig, c, fl = 150.0, float(min(0.35, 0.35 * rm["similarity"])), 0.1
            L = np.maximum(np.exp(-0.5 * (d / sig) ** 2), fl).astype(np.float32)
            where = f" ({city})" if city else ""
            if rm.get("exact"):
                note = f"{rm['tier']}: the same recorded content (e.g. an announcement jingle) as a recording from {city or 'this place'}"
            else:
                note = (f"{rm['tier']} ({rm['similarity']}) to a recording made in {city or 'a geotagged place'}; "
                        "resemblance is not location proof")
            items.append({"group": "reference_recordings", "label": f"Reference match{where}: {rm['title'][:50]}",
                          "record": rm.get("record"), "c": c, "L": L, "fp": L >= 0.5,
                          "points": [{"name": city or rm["title"][:30], "cc": rm.get("country") or "", "lat": rm["lat"],
                                      "lon": rm["lon"], "r": 12.0 if city else 50.0}],
                          "source": f"Reference library ({rm.get('license')})",
                          "precision": "city-level kernel" if city else "150 km kernel around the reference", "note": note})
        for it in items:
            if "fp" not in it:
                it["fp"] = it["L"] >= 0.3 * float(it["L"].max())
            it.setdefault("used", True)
            it.setdefault("source", it["group"])
            it.setdefault("precision", "")
            it["llr"] = _llr(it["L"].astype(np.float64), float(np.clip(it["c"], 0, 0.999)), w,
                             (means or {}).get(it["label"]))
        return items

    @staticmethod
    def fuse(items: list[dict]) -> tuple[np.ndarray, dict]:
        groups: dict[str, list[dict]] = {}
        for it in items:
            groups.setdefault(it["group"], []).append(it)
        total = None
        gscore = {}
        for g, its in groups.items():
            s = np.sum([it["llr"] for it in its], axis=0)
            if g != "user_context":
                s = s / (len(its) ** GROUP_DISCOUNT_EXP)
                s = np.clip(s, -GROUP_CLIP, GROUP_CLIP)
            gscore[g] = s
            total = s if total is None else total + s
        return total, gscore

    # ------------------------------------------------------------------ candidates
    @staticmethod
    def _resolution(radius_km: float, iso_set: set) -> tuple[str, float]:
        for r in RESOLUTIONS_KM:
            if radius_km <= r:
                return f"Within {r} km", float(r)
        if radius_km <= 500:
            return "Regional", float(np.ceil(radius_km / 50) * 50)
        if len(iso_set) == 1:
            return "Country-level", float(np.ceil(radius_km / 100) * 100)
        return "Multi-country / continental", float(np.ceil(radius_km / 100) * 100)

    def candidates(self, pts: Points, total: np.ndarray, gscore: dict, items: list[dict], max_c: int = 5,
                   min_cell_km: float = 55.0) -> list[dict]:
        p = np.exp(total - total.max()) * pts.area_km2
        p = p / p.sum()
        order = np.argsort(-total)
        assigned = np.zeros(len(total), bool)
        cands = []
        top = total.max()
        for seed in order:
            if len(cands) >= max_c or total[seed] < top - 3.0:
                break
            if assigned[seed]:
                continue
            d = haversine_km(pts.lat[seed], pts.lon[seed], pts.lat, pts.lon)
            member = (~assigned) & (total >= total[seed] - 1.0) & (d <= 1500)
            # keep the connected neighbourhood only: grow by distance rings
            ring = max(min_cell_km * 1.6, 5.0)
            midx = np.where(member)[0]
            mlat, mlon = pts.lat[midx], pts.lon[midx]
            conn_m = midx == seed
            frontier = np.where(conn_m)[0]
            while len(frontier):
                nd = np.full(len(midx), np.inf)
                for f in frontier[:400]:
                    nd = np.minimum(nd, haversine_km(mlat[f], mlon[f], mlat, mlon))
                new = ~conn_m & (nd <= ring)
                frontier = np.where(new)[0]
                conn_m |= new
            idx = midx[conn_m]
            assigned[idx] = True
            wts = p[idx]
            mass = float(wts.sum())
            # weighted centroid on the sphere
            la, lo = np.radians(pts.lat[idx]), np.radians(pts.lon[idx])
            xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
            v = (xyz * wts[:, None]).sum(0)
            v /= np.linalg.norm(v) + 1e-12
            clat, clon = float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))
            dist = haversine_km(clat, clon, pts.lat[idx], pts.lon[idx])
            o = np.argsort(dist)
            cw = np.cumsum(wts[o]) / wts.sum()
            r68 = float(dist[o][min(len(o) - 1, np.searchsorted(cw, 0.68))])
            radius = max(r68 + min_cell_km / 2, min_cell_km / 2)
            if min_cell_km >= 20:  # global grid only; local refinement is driven by the user's own prior
                radius *= RADIUS_CALIBRATION
            iso_set = set(pts.iso[idx]) - {""}
            res_label, res_km = self._resolution(radius, iso_set)
            ci = int(np.argmin(haversine_km(clat, clon, pts.lat, pts.lon)))
            support, contra = [], []
            for it in items:
                val = float(it["llr"][ci])
                entry = {"group": it["group"], "label": it["label"], "record": it["record"],
                         "contribution_nats": round(val, 2), "detail": it["note"],
                         "compatibility_at_centre": round(float(it["L"][ci]), 3)}
                if it["group"] == "biogeography" and it["L"][ci] <= OUT_OF_RANGE_FLOOR + 1e-6:
                    entry["detail"] += ("; outside modelled range here, which could mean captive, escaped, vagrant "
                                        "or misidentified, so it is floored rather than eliminating the area")
                (support if val > 0.05 else contra if val < -0.05 else support).append(entry)
            countries_in = sorted(iso_set, key=lambda i: -float(wts[pts.iso[idx] == i].sum()))
            nc = nearest_city(clat, clon)
            strength = ("stronger" if (total[seed] >= top - 0.5 and mass >= 0.25) else
                        "moderate" if total[seed] >= top - 1.5 else "weaker")
            cands.append({
                "rank": len(cands) + 1, "centre": {"lat": round(clat, 4), "lon": round(clon, 4)},
                "uncertainty_radius_km": round(res_km, 1), "resolution": res_label,
                "radius_68_km": round(radius, 1), "relative_mass": round(mass, 4),
                "score_nats": round(float(total[seed]), 2), "delta_to_best_nats": round(float(top - total[seed]), 2),
                "strength": strength, "countries": [{"iso2": i, "name": countries().get(i, {}).get("name", i)} for i in countries_in[:6]],
                "nearest_city": nc,
                "name": f"{nc['name']} area, {nc['country_name']}" if nc and radius <= 120 else
                        ", ".join(countries().get(i, {}).get("name", i) for i in countries_in[:3]) + (" region" if radius > 120 else ""),
                "supporting": sorted([s for s in support if s["contribution_nats"] > 0.05], key=lambda e: -e["contribution_nats"]),
                "contradicting": sorted(contra, key=lambda e: e["contribution_nats"]),
                "neutral": [s for s in support if abs(s["contribution_nats"]) <= 0.05],
                "cells": int(len(idx)),
            })
        cands.sort(key=lambda c: -c["relative_mass"])
        best_mass = cands[0]["relative_mass"] if cands else 1.0
        for i, c in enumerate(cands, 1):
            c["rank"] = i
            c["strength"] = ("stronger" if c["relative_mass"] >= 0.25 and c["relative_mass"] >= 0.5 * best_mass else
                             "moderate" if c["relative_mass"] >= 0.05 else "weaker")
        return cands

    # ------------------------------------------------------------------ likely places (point clues)
    def likely_places(self, items: list[dict], records, mains, ctx, weeks, extra, max_n: int = 10) -> dict | None:
        """Rank concrete places (tram networks, mentioned cities, bell towers) by the full evidence evaluated AT each place."""
        P: list[dict] = []
        for it in items:   # merge points that are the same place (e.g. a tram network and the spoken name of its city)
            for q in it.get("points") or []:
                same = next((e for e in P if haversine_km(e["lat"], e["lon"], q["lat"], q["lon"]) <= max(e["r"], q.get("r", 2.0), 3.0)), None)
                if same is None:
                    P.append({**q, "r": q.get("r", 2.0), "kinds": {it["label"]}})
                else:
                    same["kinds"].add(it["label"])
                    same["r"] = max(same["r"], q.get("r", 2.0))
        if not P:
            return None
        pts, ok = Points.at([q["lat"] for q in P], [q["lon"] for q in P])
        P = [q for q, o in zip(P, ok) if o]
        if not P:
            return None
        # global means, so a place's score is on the same scale as the world map (not relative to the pool)
        gw = self._gpts.area_km2.astype(np.float64)
        means = {it["label"]: max(float((it["L"] * gw).sum() / gw.sum()), 1e-9) for it in items}
        its = self.evidence_items(pts, records, mains, ctx, weeks, extra, means=means)
        total, _ = self.fuse(its)
        order = np.argsort(-total)
        rel = np.exp(total - total.max())
        share = rel / rel.sum()
        n_close = int((total >= total.max() - 1.0).sum())
        out = []
        for i in order[:max_n]:
            q = P[i]
            sup = sorted(((float(it["llr"][i]), it["label"]) for it in its if it["llr"][i] > 0.05), reverse=True)
            nc = nearest_city(q["lat"], q["lon"])
            out.append({"rank": len(out) + 1, "name": q["name"], "country": countries().get(q.get("cc") or "", {}).get("name", q.get("cc", "")),
                        "lat": round(float(q["lat"]), 4), "lon": round(float(q["lon"]), 4), "radius_km": round(float(max(q.get("r", 2.0), 2.0)), 1),
                        "share": round(float(share[i]), 4), "score_nats": round(float(total[i]), 2),
                        "delta_to_best_nats": round(float(total.max() - total[i]), 2), "kinds": sorted(q["kinds"]),
                        "supporting": [{"label": l, "nats": round(v, 2)} for v, l in sup[:6]], "nearest_city": nc})
        return {"places": out, "pool": len(P), "within_1_nat": n_close,
                "note": ("Each place is scored with ALL evidence evaluated at that exact point. 'share' is relative among the "
                         f"{len(P)} places considered (tram networks, mentioned cities, bell towers), not a calibrated probability."
                         + (f" {n_close} places are within 1 nat of the best, so the ranking among them is weak." if n_close > 3 else ""))}

    # ------------------------------------------------------------------ main
    def run(self, records: list[dict], mains: dict, ctx: dict, extra: dict | None = None) -> dict:
        res = self._run(records, mains, ctx, extra)
        res["footprints"], res["unlocated"] = self._footprints, self._unlocated
        res["grid"] = {"res": 0.5, "lat_top": 90.0, "lon_left": -180.0, "w": 720, "h": 360,
                       "encoding": "bitset over row-major 720x360 cells (row 0 = 90..89.5 N), base64"}
        return res

    def _run(self, records: list[dict], mains: dict, ctx: dict, extra: dict | None = None) -> dict:
        weeks, week_note = context_weeks(ctx)
        loc = (ctx or {}).get("location") or {}
        pts = Points.global_land()
        self._gpts = pts
        all_items = self.evidence_items(pts, records, mains, ctx, weeks, extra, include_all=True)
        self._footprints = build_footprints(pts, all_items)
        used_ids = {it["record"] for it in all_items if it["record"]}
        self._unlocated = [{"record": r["id"], "label": r["label"], "category": r["category_title"],
                            "specific": r.get("specific"), "confidence": r["confidence"], "tier": r["tier"],
                            "reason": unlocated_reason(r)} for r in records if r["id"] not in used_ids]
        items = [it for it in all_items if it["used"]]
        acoustic = [it for it in items if it["group"] != "user_context"]
        base = {"week_basis": week_note, "method": __doc__.strip(), "groups": sorted({it["group"] for it in items}),
                "evidence_items": [{k: v for k, v in it.items() if k not in ("L", "llr", "fp")} for it in items],
                "independence_notes": [
                    "Bird and wildlife species are one group (shared habitat/region): √n discount, clipped to ±6 nats.",
                    "Multiple languages are one group (same scene / speakers): √n discount, clipped.",
                    "GBIF occurrence counts are NOT added, because the BirdNET geomodel is built on similar occurrence data.",
                    "Mains frequency, language, biogeography, cultural infrastructure and physical environment are treated as independent groups.",
                    "User-provided context is a prior, not acoustic evidence."]}
        if not acoustic and loc.get("mode", "unknown") == "unknown":
            return {**base, "status": "no_estimate", "summary": "No reliable geographic estimate: the recording contains no "
                    "geographically informative detections (species, language, mains frequency or distinctive infrastructure).",
                    "candidates": [], "heat": []}
        total, gscore = self.fuse(items)
        # information content: effective area vs land area
        p = np.exp(total - total.max()) * pts.area_km2
        p /= p.sum()
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        eff_area = float(np.exp(ent) * np.median(pts.area_km2))
        land_area = float(pts.area_km2.sum())
        cands = self.candidates(pts, total, gscore, items)
        info_ratio = land_area / max(eff_area, 1.0)
        lp = self.likely_places(items, records, mains, ctx, weeks, extra)
        if lp:
            base["likely_places"] = lp
        n_groups = len({it["group"] for it in acoustic if it["group"] != "population"})
        # one kind of clue alone (e.g. only a 60 Hz hum) needs the stricter 2x narrowing to count as an estimate
        need = MIN_INFORMATION_RATIO if n_groups >= 2 else max(MIN_INFORMATION_RATIO, 2.0)
        if info_ratio < need and loc.get("mode", "unknown") == "unknown":
            keep = np.argsort(-total)[:1500]
            return {**base, "status": "no_estimate", "effective_area_km2": round(eff_area), "land_area_km2": round(land_area),
                    "information_ratio": round(info_ratio, 2), "candidates": [], "weak_candidates": cands[:3],
                    "heat": [[round(float(pts.lat[i]), 2), round(float(pts.lon[i]), 2),
                              round(float(np.exp(total[i] - total.max())), 4)] for i in keep],
                    "heat_cell_deg": grid().res,
                    "summary": (f"No reliable geographic estimate: the evidence only narrows the plausible land area by "
                                f"{info_ratio:.2f}x (threshold {MIN_INFORMATION_RATIO}x). Weak candidates are kept in the "
                                "case data for transparency but are not presented as estimates.")}
        result = {**base, "status": "estimate", "effective_area_km2": round(eff_area), "land_area_km2": round(land_area),
                  "information_ratio": round(land_area / max(eff_area, 1.0), 2), "candidates": cands}
        # local refinement when a city/region prior or a compact best candidate exists
        if cands and (loc.get("mode") in ("city", "region") or cands[0]["uncertainty_radius_km"] <= 100):
            c0 = cands[0]["centre"]
            rad = 150.0 if loc.get("mode") != "city" else 60.0
            lp = Points.local(c0["lat"], c0["lon"], rad, step_km=2.0 if rad <= 60 else 4.0)
            if len(lp.lat) > 10:
                litems = self.evidence_items(lp, records, mains, ctx, weeks, extra)
                ltotal, lg = self.fuse(litems)
                lc = self.candidates(lp, ltotal, lg, litems, max_c=3, min_cell_km=2.0 if rad <= 60 else 4.0)
                if lc:
                    result["local_refinement"] = {"centre": c0, "radius_km": rad, "candidates": lc,
                                                  "note": ("Re-evaluated on a fine grid around the best candidate. Precision "
                                                           "is limited by the evidence. Range and language layers are smooth "
                                                           "at this scale, so the radius mainly reflects the user prior.")}
        # heat layer (top cells) for the globe
        keep = np.argsort(-total)[:2500]
        keep = keep[total[keep] >= total.max() - 6]
        rel = np.exp(total[keep] - total.max())
        result["heat"] = [[round(float(pts.lat[i]), 2), round(float(pts.lon[i]), 2), round(float(v), 4)] for i, v in zip(keep, rel)]
        result["heat_cell_deg"] = grid().res
        if cands:
            b = cands[0]
            result["summary"] = (f"Best candidate: {b['name']} (centre {b['centre']['lat']:.3f}, {b['centre']['lon']:.3f}; "
                                 f"{b['resolution'].lower()}, uncertainty radius {b['uncertainty_radius_km']:g} km). "
                                 f"Heuristic relative scores, not calibrated probabilities.")
        # annotate records with their geographic relevance
        for it in items:
            if it["record"]:
                for r in records:
                    if r["id"] == it["record"]:
                        r["geographic_relevance"] = f"Used as {it['group']} evidence: {it['note']}."
        return result

    # ------------------------------------------------------------------ species detail for the report
    def species_profiles(self, records: list[dict], cand: dict | None) -> list[dict]:
        sp = {}
        for r in records:
            if r.get("scientific_name"):
                sp.setdefault(r["scientific_name"], r)
        if not sp:
            return []
        from ..models import birdnet as bnm
        gidx = bnm.geo_index(self.models_dir)
        out = []
        pts = Points.global_land()
        for sci, r in sp.items():
            k = sci.lower()
            info = {"scientific_name": sci, "common_name": r["common_name"], "taxon_class": r.get("taxon_class")}
            if k in gidx:
                occ = self.occurrence(pts, None, [gidx[k]])[:, 0]
                m = occ >= 0.25
                ctry = {}
                for i in pts.iso[m]:
                    ctry[i] = ctry.get(i, 0) + 1
                topc = sorted(ctry.items(), key=lambda t: -t[1])[:8]
                info["distribution"] = {"land_fraction_occ_ge_0.25": round(float(m.mean()), 4),
                                        "main_countries": [countries().get(i, {}).get("name", i) for i, _ in topc]}
                if cand:
                    prof = bnm.geo_weekly_profile(self.models_dir, cand["centre"]["lat"], cand["centre"]["lon"], [gidx[k]])[:, 0]
                    months = [float(prof[i * 4:(i + 1) * 4].max()) for i in range(12)]
                    present = [i + 1 for i, v in enumerate(months) if v >= 0.25]
                    if len(present) == 12:
                        pattern = "expected year-round at the best candidate"
                    elif not present:
                        pattern = "not expected at the best candidate in any month (range model)"
                    else:
                        pattern = "seasonal at the best candidate: expected in months " + ", ".join(map(str, present))
                    info["seasonality_at_best_candidate"] = {"monthly_max_occurrence": [round(v, 3) for v in months],
                                                             "pattern": pattern}
            else:
                info["distribution"] = None
            info["habitat"] = "Not available in the offline datasets used (optional online GBIF lookup)."
            out.append(info)
        return out
