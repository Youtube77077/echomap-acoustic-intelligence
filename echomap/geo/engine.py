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

    # ------------------------------------------------------------------ evidence builders
    def evidence_items(self, pts: Points, records: list[dict], mains: dict, ctx: dict, weeks) -> list[dict]:
        items = []
        w = pts.area_km2.astype(np.float64)
        # --- biogeography (BirdNET species + geomodel)
        sp = [r for r in records if r.get("scientific_name")]
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
                    occ = None
                    for wk in (weeks or [-1]):
                        o = bnm.geo_occurrence(self.models_dir, pts.lat, pts.lon, wk, idxs)
                        occ = o if occ is None else np.maximum(occ, o)
                    for j, k in enumerate(keys):
                        r = best[k]
                        L = np.maximum(occ[:, j], OUT_OF_RANGE_FLOOR)
                        frac = float((occ[:, j] >= 0.25).astype(float) @ w / w.sum())
                        items.append({"group": "biogeography", "label": f"{r['common_name']} ({r['scientific_name']})",
                                      "record": r["id"], "c": r["confidence"], "L": L,
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
                          "c": r["confidence"], "L": L,
                          "note": f"CLDR population share; spoken by ≥ 50% in {int((share >= 0.5).sum())} land cells' countries"})
        # --- electrical grid
        if mains.get("detected"):
            hz = str(mains["nominal_hz"])
            match = (pts.mains == hz).astype(np.float32)
            mixed = np.char.find(pts.mains.astype(str), "/") >= 0
            L = np.where(mixed, 0.5, np.where(match > 0, 0.97, 0.03)).astype(np.float32)
            rec = next((r for r in records if r["label"].startswith("Mains hum")), None)
            c = 0.95 if rec and rec["tier"] == "confirmed" else 0.7
            items.append({"group": "electrical_grid", "label": f"Mains hum {hz} Hz", "record": rec["id"] if rec else None,
                          "c": c, "L": L, "note": "Nominal grid frequency per country (Japan split east/west)"})
        # --- cultural infrastructure
        for r in records:
            names = set(r.get("evidence", {}).get("ontology_parents_detected", [])) | {r["label"]}
            if "Change ringing (campanology)" in names and r["confidence"] >= 0.2:
                L = np.where(pts.iso == "GB", 1.0, 0.08).astype(np.float32)
                items.append({"group": "cultural", "label": "Change ringing (English-style bells)", "record": r["id"],
                              "c": r["confidence"], "L": L, "note": "Change-ringing towers concentrated in England"})
            pat = r.get("evidence", {}).get("siren_pattern", {}).get("pattern", "")
            if "two-tone" in pat:
                eu = np.array([countries().get(i, {}).get("continent") == "EU" for i in pts.iso])
                items.append({"group": "cultural", "label": "Two-tone emergency siren", "record": r["id"],
                              "c": min(0.6, r["confidence"]), "L": np.where(eu, 1.0, 0.45).astype(np.float32),
                              "note": "Weak: hi-lo sirens are common in Europe"})
        # --- physical environment
        for r in records:
            if r["label"] in ("Waves, surf", "Ocean") and r["confidence"] >= 0.2:
                L = np.where(pts.coastal, 1.0, 0.2).astype(np.float32)
                items.append({"group": "environment", "label": f"{r['label']} (coastal water)", "record": r["id"],
                              "c": min(0.8, r["confidence"]), "L": L,
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
        for it in items:
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
        return cands

    # ------------------------------------------------------------------ main
    def run(self, records: list[dict], mains: dict, ctx: dict) -> dict:
        weeks, week_note = context_weeks(ctx)
        loc = (ctx or {}).get("location") or {}
        pts = Points.global_land()
        items = self.evidence_items(pts, records, mains, ctx, weeks)
        acoustic = [it for it in items if it["group"] != "user_context"]
        base = {"week_basis": week_note, "method": __doc__.strip(), "groups": sorted({it["group"] for it in items}),
                "evidence_items": [{k: v for k, v in it.items() if k not in ("L", "llr")} for it in items],
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
        result = {**base, "status": "estimate", "effective_area_km2": round(eff_area), "land_area_km2": round(land_area),
                  "information_ratio": round(land_area / max(eff_area, 1.0), 2), "candidates": cands}
        # local refinement when a city/region prior or a compact best candidate exists
        if cands and (loc.get("mode") in ("city", "region") or cands[0]["uncertainty_radius_km"] <= 100):
            c0 = cands[0]["centre"]
            rad = 150.0 if loc.get("mode") != "city" else 60.0
            lp = Points.local(c0["lat"], c0["lon"], rad, step_km=2.0 if rad <= 60 else 4.0)
            if len(lp.lat) > 10:
                litems = self.evidence_items(lp, records, mains, ctx, weeks)
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
                occ = bnm.geo_occurrence(self.models_dir, pts.lat, pts.lon, -1, [gidx[k]])[:, 0]
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
