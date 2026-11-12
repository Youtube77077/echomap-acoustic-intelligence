"""Unknown-sound discovery.

Finds events that the classifiers do not explain well, describes them acoustically, checks model
disagreement, groups similar unknowns (within the recording and against EchoMap's local library of earlier
unknowns), and never forces them into the nearest known class.

Signals used:
- Novelty: low maximum AudioSet score and low BirdNET score where energy is clearly present.
- Ambiguity: the top AudioSet labels come from different ontology branches with similar scores.
- Model disagreement: AudioSet and BirdNET disagree about whether an animal is present.
- Embedding similarity: 960-d EfficientAT window embeddings, compared by cosine similarity. These cover 5 s
  context windows, so short events share their embedding with surrounding sound (a stated limitation).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import app_home
from .taxonomy import class_table


def _segment_embedding(sed: dict, t0: float, t1: float) -> np.ndarray | None:
    st, en, E = sed["starts"], sed["ends"], sed.get("embeddings")
    if E is None or len(E) == 0:
        return None
    c = (t0 + t1) / 2
    cover = (st <= c) & (en >= c)
    if not cover.any():
        cover = np.abs((st + en) / 2 - c) == np.abs((st + en) / 2 - c).min()
    v = E[cover].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def describe(prof: dict, flat: float | None, periodic: dict | None, dur: float) -> str:
    parts = []
    if flat is not None:
        parts.append("tonal" if flat < 0.15 else "noise-like" if flat > 0.45 else "mixed tonal/noise")
    if prof:
        parts.append(f"energy peak ~{prof.get('peak_hz', 0):.0f} Hz (5-95% band {prof.get('band_5_95_hz')})")
    if periodic:
        parts.append(f"repeating every {periodic['period_s']} s")
    parts.append(f"{dur:.2f} s long")
    return ", ".join(parts)


def _flatness(x, sr, t0, t1):
    seg = x[int(t0 * sr):int(t1 * sr)]
    if len(seg) < 512:
        return None
    p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 + 1e-14
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def discover(records: list[dict], sed: dict, bn: dict | None, x: np.ndarray, sr: int, patterns: list[dict],
             profile_fn, labels: list[str], reference_search=None, case_id: str = "") -> dict:
    table = class_table()
    branch = {c["name"]: (c["ancestors"][-1] if c["ancestors"] else c["name"]) for c in table}
    fs, hop = sed["frame_scores"], sed["hop_s"]
    out = []
    cands = [r for r in records if r["tier"] == "unclassified" or (r["kind"] == "model" and r["confidence"] < 0.3)]
    for r in cands:
        t0, t1 = r["start"], r["end"]
        a, b = int(t0 / hop), max(int(t0 / hop) + 1, int(np.ceil(t1 / hop)))
        seg = fs[a:b].max(axis=0)
        order = np.argsort(-seg)
        top = [(labels[i], float(seg[i])) for i in order[:5]]
        novelty = 1.0 - top[0][1]
        bn_max, bn_top = 0.0, None
        if bn is not None:
            m = (bn["starts"] < t1) & (bn["ends"] > t0)
            if m.any():
                col = bn["scores"][m].max(axis=0)
                k = int(col.argmax())
                bn_max, bn_top = float(col[k]), k
        disagreements = []
        animal_as = any("Animal" in (table[i]["ancestors"] + [table[i]["name"]]) and seg[i] >= 0.2 for i in order[:5])
        if bn_max >= 0.3 and not animal_as:
            disagreements.append("BirdNET suggests an animal vocalization; AudioSet does not")
        if animal_as and bn is not None and bn_max < 0.1:
            disagreements.append("AudioSet suggests an animal; BirdNET finds no species")
        if len(top) > 1 and top[0][1] - top[1][1] < 0.08 and branch.get(top[0][0]) != branch.get(top[1][0]) and top[0][1] > 0.05:
            disagreements.append(f"Ambiguous between '{top[0][0]}' and '{top[1][0]}' (different sound families)")
        per = next((p for p in patterns if p["start"] <= t1 and p["end"] >= t0), None)
        prof = profile_fn(t0, t1)
        emb = _segment_embedding(sed, t0, t1)
        refs = reference_search(emb, k=3) if (reference_search and emb is not None) else []
        out.append({"record": r["id"], "start": t0, "end": t1, "novelty": round(novelty, 3),
                    "description": describe(prof, _flatness(x, sr, t0, t1), per, t1 - t0),
                    "closest_model_labels": [{"label": l, "score": round(s, 3)} for l, s in top],
                    "birdnet_max": round(bn_max, 3), "disagreements": disagreements, "similar_references": refs,
                    "_emb": emb})
    # group similar unknowns within the recording (cosine >= 0.9 on context embeddings)
    fam, groups = 0, {}
    for u in out:
        if u["_emb"] is None:
            continue
        for g, rep in groups.items():
            if float(u["_emb"] @ rep) >= 0.9:
                u["family"] = g
                break
        else:
            fam += 1
            g = f"U{fam}"
            groups[g] = u["_emb"]
            u["family"] = g
    # compare with EchoMap's local library of unknowns from earlier cases (stored locally only)
    lib_path = app_home() / "unknown_library.jsonl"
    library = []
    if lib_path.exists():
        for line in lib_path.read_text(encoding="utf-8").splitlines()[-2000:]:
            try:
                library.append(json.loads(line))
            except ValueError:
                pass
    if library:
        L = np.array([e["emb"] for e in library], np.float32)
        for u in out:
            if u["_emb"] is None:
                continue
            sims = L @ u["_emb"]
            j = int(np.argmax(sims))
            if sims[j] >= 0.9:
                u["matches_earlier_unknown"] = {"case": library[j]["case"], "time": library[j]["t"],
                                                "similarity": round(float(sims[j]), 3), "description": library[j]["desc"]}
    try:
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lib_path, "a", encoding="utf-8") as f:
            for u in out:
                if u["_emb"] is not None and u["novelty"] >= 0.7:
                    f.write(json.dumps({"case": case_id, "t": u["start"], "desc": u["description"],
                                        "emb": [round(float(v), 4) for v in u["_emb"]], "added": time.time()}) + "\n")
    except OSError:
        pass
    for u in out:
        u.pop("_emb", None)
    for r in records:
        hit = next((u for u in out if u["record"] == r["id"]), None)
        if hit:
            r["discovery"] = {k: v for k, v in hit.items() if k != "record"}
    return {"events": out, "families": len(groups),
            "method": __doc__.strip(),
            "summary": f"{len(out)} poorly explained event(s) in {len(groups)} acoustic family(ies)."}
