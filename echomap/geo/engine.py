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

import numpy as np

from .gazetteer import cities_in_region, nearest_city
from .grid import countries, grid, haversine_km, local_grid

GROUP_CLIP = 6.0
OUT_OF_RANGE_FLOOR = 0.02      # captive / escaped / vagrant / transported animals
LANGUAGE_FLOOR = 0.005         # tourists, migrants, broadcast media
RESOLUTIONS_KM = [1, 5, 10, 25, 50, 100]
SPECIES_MIN_CONF = 0.4         # species predictions below this are listed but not used for geography
MIN_INFORMATION_RATIO = 2.0
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
        out.append({"id": f"F{i + 1}", "label": it["label"], "group": it["group"], "record": it["record"],
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


def _llr(L: np.ndarray, c: float, weights: np.ndarray) -> np.ndarray:
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
                       extra: dict | None = None, include_all: bool = False) -> list[dict]:
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
                        c_eff = r["confidence"] if r.get("corroborated_by") else 0.5 * r["confidence"]
                        items.append({"group": "biogeography", "label": f"{r['common_name']} ({r['scientific_name']})",
                                      "record": r["id"], "c": c_eff, "L": L, "fp": occ[:, j] >= 0.25,
                                      "used": r["confidence"] >= SPECIES_MIN_CONF, "source": "BirdNET+ Geomodel V3.0.4",
                                      "precision": "2° range-model lattice (~220 km)",
                                      "note": f"range model: occurrence ≥ 0.25 over {frac*100:.1f}% of land area",
                                      "range_fraction": frac})
                for k in best:
                    if k not in gidx:
                        best[k]["geographic_relevance"] = "Species not in the geomodel label set, so no range evidence."
            except FileNotFoundError:
                pass
        # --- language
        langs: dict[str, dict] = {}
        for r in records:
            if r.get("language_code"):
                if r["language_code"] not in langs or r["confidence"] > langs[r["language_code"]]["confidence"]:
                    langs[r["language_code"]] = r
        for code, r in langs.items():
            share = pts.language_share(code)
            if share.max() <= 0:
                r["geographic_relevance"] = "Language not in the CLDR territory data, so no geographic evidence."
                continue
            L = np.maximum(share, LANGUAGE_FLOOR)
            items.append({"group": "language", "label": f"Language {r['language_name']} ({code})", "record": r["id"],
                          "c": r["confidence"], "L": L, "fp": share >= 0.15, "source": "Unicode CLDR territoryInfo",
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
                          "c": c, "L": L, "fp": L >= 0.5, "source": "Mains frequency per country (Wikipedia, transcribed)",
                          "precision": "country / grid level", "note": "Nominal grid frequency per country (Japan split east/west)"})
        # --- cultural infrastructure
        for r in records:
            names = set(r.get("evidence", {}).get("ontology_parents_detected", [])) | {r["label"]}
            if "Change ringing (campanology)" in names and r["confidence"] >= 0.2:
                L = np.where(pts.iso == "GB", 1.0, 0.08).astype(np.float32)
                items.append({"group": "bells", "label": "Change ringing (English-style bells)", "record": r["id"],
                              "c": r["confidence"], "L": L, "fp": L >= 0.5, "source": "Dove's Guide tower distribution",
                              "precision": "country level", "note": "Change-ringing towers concentrated in England"})
            pat = (r.get("evidence", {}).get("siren_pattern") or {}).get("pattern", "")
            if "two-tone" in pat:
                eu = np.array([countries().get(i, {}).get("continent") == "EU" for i in pts.iso])
                items.append({"group": "cultural", "label": "Two-tone emergency siren", "record": r["id"],
                              "c": min(0.6, r["confidence"]), "L": np.where(eu, 1.0, 0.45).astype(np.float32), "fp": eu,
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
        # --- user context (prior, not evidence)
        loc = (ctx or {}).get("location") or {}
        mode = loc.get("mode", "unknown")
        if mode == "country" and loc.get("country"):
            items.append({"group": "user_context", "label": f"User: country {loc['country']}", "record": None, "c": 0.999,
                          "L": np.where(pts.iso == loc["country"], 1.0, 0.002).astype(np.float32), "note": "User-provided"})
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
        extra = extra or {}
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
                          "precision": "tower-level points with 25 km kernels; many towers share bell notes",
                          "note": "Rings whose bell nominals match the recording within ~25 cents"})
        for rm in (extra.get("reference_geo") or []):
            d = haversine_km(rm["lat"], rm["lon"], pts.lat, pts.lon)
            L = np.maximum(np.exp(-0.5 * (d / 150.0) ** 2), 0.1).astype(np.float32)
            items.append({"group": "reference_recordings", "label": f"Reference match: {rm['title'][:60]}",
                          "record": rm.get("record"), "c": float(min(0.35, 0.35 * rm["similarity"])), "L": L, "fp": L >= 0.5,
                          "source": f"Reference library ({rm.get('license')})", "precision": "150 km kernel around the reference",
                          "note": f"{rm['tier']} ({rm['similarity']}) to a geotagged recording; resemblance is not location proof"})
        for it in items:
            if "fp" not in it:
                it["fp"] = it["L"] >= 0.3 * float(it["L"].max())
            it.setdefault("used", True)
            it.setdefault("source", it["group"])
            it.setdefault("precision", "")
            it["llr"] = _llr(it["L"].astype(np.float64), float(np.clip(it["c"], 0, 0.999)), w)
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
                s = s / np.sqrt(len(its))
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
        if info_ratio < MIN_INFORMATION_RATIO and loc.get("mode", "unknown") == "unknown":
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
