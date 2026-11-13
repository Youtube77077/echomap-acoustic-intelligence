"""Animal-species head trained on iNatSounds 2024 (v1.0.9): 5,569 species of birds, insects, amphibians, mammals and
reptiles, from EfficientAT embeddings (every mode) or BirdNET+ embeddings (when BirdNET runs and that head was shipped).

Trained by tools/train_inat_head.py on GitHub (inat-head workflow); accuracy on the iNatSounds validation split is in
docs/eval/inat_head.json and is shown with every result. No recordings are stored: the file holds a PCA projection and
a softmax regression. iNatSounds terms: non-commercial research and education (approved by the user for personal use).
A species is only reported when the head's probability reaches the threshold at which validation precision was >= 70%.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "inat_head.npz"


@lru_cache(maxsize=1)
def head() -> dict | None:
    if not DATA.exists():
        return None
    d = np.load(DATA, allow_pickle=False)
    h = {"classes": [str(c) for c in d["classes"]], "common": [str(c) for c in d["common"]],
         "groups": [str(g) for g in d["groups"]], "shipped": [str(s) for s in d["shipped"]], "heads": {}}
    for k in h["shipped"]:
        h["heads"][k] = {"mu": d[f"{k}_mu"].astype(np.float32), "P": d[f"{k}_P"].astype(np.float32),
                         "W": d[f"{k}_W"].astype(np.float32), "b": d[f"{k}_b"].astype(np.float32),
                         "thr": float(d[f"{k}_thr"])}
    h["metrics"] = {}
    if "metrics_json" in d.files:   # validation metrics travel with the head (docs/ is not in the exe)
        h["metrics"] = json.loads(str(d["metrics_json"]))
    return h


def _l2(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def classify(eat_emb: np.ndarray | None, bn_emb: np.ndarray | None = None, k: int = 3) -> dict | None:
    h = head()
    if h is None or eat_emb is None:
        return None
    kind = "eat"
    x = _l2(eat_emb)
    if bn_emb is not None and np.any(bn_emb):
        for cand in ("both", "bn"):
            if cand in h["heads"]:
                kind = cand
                x = np.concatenate([_l2(eat_emb), _l2(bn_emb)]) if cand == "both" else _l2(bn_emb)
                break
    m = h["heads"][kind]
    z = (x - m["mu"]) @ m["P"] @ m["W"] + m["b"]
    p = np.exp(z - z.max())
    p /= p.sum()
    top = np.argsort(-p)[:k]
    return {"kind": kind, "threshold": m["thr"], "metrics": h["metrics"].get(kind, {}),
            "top": [{"sci": h["classes"][i], "common": h["common"][i], "group": h["groups"][i], "p": round(float(p[i]), 4)}
                    for i in top]}


def _seg(starts, ends, E, t0, t1):
    if E is None or not len(E):
        return None
    c = (t0 + t1) / 2
    cover = (starts <= c) & (ends >= c)
    if not cover.any():
        cover = np.abs((starts + ends) / 2 - c) == np.abs((starts + ends) / 2 - c).min()
    return E[cover].mean(axis=0)


def add_records(inv, sed: dict, bn: dict | None, profile_fn) -> int:
    """Classify every animal-sound interval; add a species record (or corroborate BirdNET's) when confident."""
    from ..detect.inventory import tier_for
    from ..detect.wildlife import CLASS_TITLE, CLASS_TO_AUDIOSET
    if head() is None:
        return 0
    animal = set().union(*CLASS_TO_AUDIOSET.values())
    spans = [r for r in inv.records if r["category"] == "birds_wildlife" or set(r.get("_audioset_names", [])) & animal]
    # background animals: every stretch of SED windows with some animal activity that no record covers yet
    from ..detect.taxonomy import class_table
    aidx = [c["idx"] for c in class_table() if c["name"] in animal]
    if aidx and len(sed.get("scores", [])):
        act = sed["scores"][:, aidx].max(axis=1) >= 0.08
        i = 0
        while i < len(act):
            if act[i]:
                j = i
                while j + 1 < len(act) and act[j + 1]:
                    j += 1
                t0, t1 = float(sed["starts"][i]), float(sed["ends"][j])
                if not any(r["start"] < t1 and r["end"] > t0 for r in spans):
                    spans.append({"start": t0, "end": t1, "label": "Animal activity (background)", "category": "",
                                  "id": None})
                i = j + 1
            else:
                i += 1
    bn_E = bn.get("embeddings") if bn else None
    added = 0
    for r in list(spans):
        res = classify(_seg(sed["starts"], sed["ends"], sed.get("embeddings"), r["start"], r["end"]),
                       _seg(bn["starts"], bn["ends"], bn_E, r["start"], r["end"]) if bn_E is not None else None)
        if not res:
            continue
        if r.get("id"):
            r.setdefault("evidence", {})["inat_head"] = res
        best = res["top"][0]
        if best["p"] < res["threshold"]:
            continue
        same = [x for x in inv.records if (x.get("scientific_name") or "").lower() == best["sci"].lower()
                and x["start"] < r["end"] and x["end"] > r["start"]]
        if same:   # BirdNET already found this species here: independent agreement
            for x in same:
                x.setdefault("corroborated_by", []).append(f"iNatSounds species head ({best['p']:.0%})")
                x["tier"] = tier_for(x["confidence"], True)
            continue
        acc = res["metrics"].get("top1")
        inv.add(category="birds_wildlife", label=f"{CLASS_TITLE.get(best['group'], 'Animal')} vocalization",
                specific=f"Possible species: {best['common'] or best['sci']} ({best['sci']})",
                common_name=best["common"] or best["sci"], scientific_name=best["sci"], taxon_class=best["group"],
                candidates=[{"label": f"{t['common'] or t['sci']} ({t['sci']})", "score": t["p"]} for t in res["top"][1:]],
                start=r["start"], end=r["end"], confidence=best["p"], tier=tier_for(best["p"], True),
                frequency=profile_fn(r["start"], r["end"], 100.0, 14000.0), model="iNatSounds species head (EchoMap)",
                corroborated_by=[f"{r['id']} {r['label']}"] if r.get("id") else [],
                evidence={"inat_head": res},
                limitations=(f"Model prediction from a head trained on iNatSounds (5,569 species). Validation top-1 "
                             f"accuracy {acc:.0%} over all species; " if acc else "Model prediction; ") +
                            "shown only above the threshold where validation precision was at least 70%. Similar "
                            "species, recordings played back, captive animals and mimics can all produce detections.",
                signature="Animal sound classified by the iNatSounds head")
        added += 1
    return added
