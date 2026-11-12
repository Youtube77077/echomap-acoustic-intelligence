"""Detection inventory: one record per individual acoustic event, with model-output association,
de-duplication, and transparent confidence tiers.

Tiers
- confirmed   : deterministic measurement above a stated threshold, OR model score ≥ 0.70 with
                corroboration from an *independent* detector (another model or a measurement)
- probable    : model score ≥ 0.50 without corroboration (or ≥ 0.35 with corroboration)
- possible    : model score ≥ list threshold (default 0.15)
- unclassified: energy above the noise floor not explained by any detection
Scores are model outputs, NOT calibrated probabilities.
"""
from __future__ import annotations

import numpy as np

from .taxonomy import CATEGORY_NAMES, GENERIC, class_table


def tier_for(score: float, corroborated: bool, measurement: bool = False) -> str:
    if measurement:
        return "confirmed"
    if score >= 0.70 and corroborated:
        return "confirmed"
    if score >= 0.50 or (corroborated and score >= 0.35):
        return "probable"
    return "possible"


def runs(mask: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    """[start, end) index runs of True, bridging gaps of up to `gap - 1` False frames."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if out and i - out[-1][1] < gap:
                out[-1] = (out[-1][0], j)
            else:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class Inventory:
    def __init__(self, duration: float, list_threshold: float = 0.15):
        self.duration = duration
        self.thr = list_threshold
        self.records: list[dict] = []
        self._n = 0

    def add(self, **kw) -> dict:
        self._n += 1
        rec = {
            "id": f"D{self._n:04d}",
            "category": kw.pop("category"),
            "label": kw.pop("label"),
            "specific": kw.pop("specific", None),
            "candidates": kw.pop("candidates", []),
            "start": round(float(kw.pop("start")), 3),
            "end": round(float(kw.pop("end")), 3),
            "confidence": round(float(kw.pop("confidence")), 4),
            "tier": kw.pop("tier"),
            "frequency": kw.pop("frequency", {}),
            "signature": kw.pop("signature", ""),
            "model": kw.pop("model", ""),
            "evidence": kw.pop("evidence", {}),
            "corroborated_by": kw.pop("corroborated_by", []),
            "geographic_relevance": kw.pop("geographic_relevance", "Not geographically distinctive on its own."),
            "temporal_relevance": kw.pop("temporal_relevance", "None."),
            "limitations": kw.pop("limitations", ""),
            "kind": kw.pop("kind", "model"),
        }
        rec["duration"] = round(rec["end"] - rec["start"], 3)
        rec["category_title"] = CATEGORY_NAMES.get(rec["category"], rec["category"])
        rec.update(kw)
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------ AudioSet events
    def sed_events(self, sed: dict) -> list[dict]:
        """Raw per-class events from frame-pooled SED scores (before association)."""
        table = class_table()
        fs = sed["frame_scores"]
        hop = sed["hop_s"]
        raw = []
        for c in range(fs.shape[1]):
            col = fs[:, c]
            if col.max() < self.thr:
                continue
            for a, b in runs(col >= self.thr, gap=2):
                t0, t1 = a * hop, min(self.duration, b * hop)
                win = (sed["starts"] < t1) & (sed["ends"] > t0)
                raw.append({"cls": c, "name": table[c]["name"], "category": table[c]["category"],
                            "depth": table[c]["depth"], "alt_group": table[c]["alt_group"],
                            "ancestors": table[c]["ancestors"], "start": t0, "end": max(t1, t0 + hop),
                            "score": float(col[a:b].max()),
                            "max_window_score": float(sed["scores"][win, c].max()) if win.any() else float(col[a:b].max()),
                            "mean_score": float(col[a:b].mean())})
        return raw

    @staticmethod
    def associate(raw: list[dict]) -> list[list[dict]]:
        """Cluster events that describe the same physical sound: overlapping in time AND related by
        ontology (ancestor/descendant) or by an alternative group (e.g. Car vs Bus)."""
        n = len(raw)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                a, b = raw[i], raw[j]
                ov = overlap(a["start"], a["end"], b["start"], b["end"])
                if ov <= 0 or ov < 0.5 * min(a["end"] - a["start"], b["end"] - b["start"]):
                    continue
                related = (a["name"] in b["ancestors"] or b["name"] in a["ancestors"] or
                           (a["alt_group"] is not None and a["alt_group"] == b["alt_group"]))
                if related:
                    parent[find(i)] = find(j)
        groups: dict[int, list[dict]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(raw[i])
        return list(groups.values())

    def add_sed_groups(self, groups: list[list[dict]], profile_fn) -> list[dict]:
        out = []
        for g in groups:
            g = sorted(g, key=lambda e: (-(e["name"] not in GENERIC), -e["depth"] * 0.05 - e["score"]))
            specific = [e for e in g if e["name"] not in GENERIC]
            primary = max(specific or g, key=lambda e: e["score"] + 0.05 * e["depth"])
            t0 = min(e["start"] for e in g)
            t1 = max(e["end"] for e in g)
            cands = sorted(({"label": e["name"], "score": round(e["score"], 4)} for e in g if e is not primary),
                           key=lambda d: -d["score"])
            supporting = [e for e in g if e["name"] in primary["ancestors"]]
            rec = self.add(
                category=primary["category"], label=primary["name"], specific=None, candidates=cands,
                start=t0, end=t1, confidence=primary["score"], tier=tier_for(primary["score"], False),
                frequency=profile_fn(t0, t1), model="EfficientAT mn10_as (AudioSet)",
                signature="", kind="model",
                evidence={"model_outputs": [{"model": "EfficientAT mn10_as", "label": e["name"],
                                             "frame_pooled_max": round(e["score"], 4),
                                             "max_window_score": round(e["max_window_score"], 4),
                                             "mean": round(e["mean_score"], 4),
                                             "start": round(e["start"], 2), "end": round(e["end"], 2)} for e in g],
                          "ontology_parents_detected": [e["name"] for e in supporting]},
                limitations="AudioSet tagger trained on weak 10 s labels. Boundaries are about 0.5 s (window pooling). "
                            "Overlapping sounds of the same family may be merged into one record.",
            )
            rec["_audioset_names"] = [e["name"] for e in g]
            out.append(rec)
        return out

    # ------------------------------------------------------------------ helpers
    def find_overlapping(self, t0, t1, pred, min_frac=0.3):
        res = []
        for r in self.records:
            ov = overlap(t0, t1, r["start"], r["end"])
            if ov > 0 and ov >= min_frac * min(t1 - t0, max(r["duration"], 1e-3)) and pred(r):
                res.append(r)
        return res

    def remove(self, recs):
        ids = {r["id"] for r in recs}
        self.records = [r for r in self.records if r["id"] not in ids]

    def finalize(self) -> list[dict]:
        for r in self.records:
            r.pop("_audioset_names", None)
            r["duration"] = round(r["end"] - r["start"], 3)
        self.records.sort(key=lambda r: (r["start"], r["category"]))
        for i, r in enumerate(self.records, 1):
            r["id"] = f"D{i:04d}"
        return self.records

    def summary(self) -> dict:
        cats = {k: {"title": v, "count": 0, "tiers": {"confirmed": 0, "probable": 0, "possible": 0, "unclassified": 0}}
                for k, v in CATEGORY_NAMES.items()}
        for r in self.records:
            c = cats.setdefault(r["category"], {"title": r["category"], "count": 0,
                                                "tiers": {"confirmed": 0, "probable": 0, "possible": 0, "unclassified": 0}})
            c["count"] += 1
            c["tiers"][r["tier"]] = c["tiers"].get(r["tier"], 0) + 1
        return cats
