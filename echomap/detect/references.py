"""Acoustic signature database and reverse sound search.

Two independent matchers:
1. Landmark fingerprints (spectral-peak pairs hashed with their time offset, Wang 2003 style). Many
   aligned hash hits with one consistent offset means *the same recording* (near-exact match).
2. Embedding similarity (EfficientAT 960-d, cosine). This is acoustic similarity, not identity.

Match tiers (shown with every result):
- near-exact fingerprint match : ≥ 20 aligned landmark hits with one consistent offset
- strong acoustic similarity   : cosine ≥ 0.92
- category similarity          : cosine ≥ 0.80
- weak resemblance             : cosine ≥ 0.65
A similarity never proves two sounds came from the same physical source.
The reference library (references.json.gz) is built in CI from openly licensed recordings, and each entry
keeps its source, licence, attribution and coordinates.
"""
from __future__ import annotations

import base64
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..dsp.spectral import stft_power

DATA = Path(__file__).resolve().parents[1] / "data" / "references.json.gz"
FP_SR = 8000


def fingerprint(x8k: np.ndarray, fan: int = 5) -> list[tuple[int, int]]:
    """Return (hash, frame) landmarks from an 8 kHz signal."""
    if len(x8k) < 2048:
        return []
    p = stft_power(x8k, 512, 128)
    db = 10 * np.log10(p + 1e-12)
    T = db.shape[1]
    peaks = []
    for t in range(T):
        col = db[:, t]
        thr = np.percentile(col, 95)
        idx = np.where((col > thr) & (col >= np.roll(col, 1)) & (col >= np.roll(col, -1)))[0]
        idx = idx[np.argsort(-col[idx])][:3]
        peaks += [(t, int(f)) for f in idx]
    out = []
    for i, (t1, f1) in enumerate(peaks):
        paired = 0
        for t2, f2 in peaks[i + 1:i + 1 + fan * 6]:
            dt = t2 - t1
            if 1 <= dt <= 63:
                out.append((((f1 & 0x1FF) << 15) | ((f2 & 0x1FF) << 6) | dt, t1))
                paired += 1
                if paired >= fan:
                    break
    return out


@lru_cache(maxsize=1)
def library() -> dict:
    if not DATA.exists():
        return {"refs": [], "E": np.zeros((0, 960), np.float32), "fp": {}, "meta": {}}
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        d = json.load(f)
    refs = d["refs"]
    E = np.array([np.frombuffer(base64.b64decode(r.pop("embedding")), np.float16).astype(np.float32) for r in refs]) \
        if refs else np.zeros((0, 960), np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    fp: dict[int, list[tuple[int, int]]] = {}
    for i, r in enumerate(refs):
        for h, t in r.pop("fingerprint", []):
            fp.setdefault(h, []).append((i, t))
    return {"refs": refs, "E": E, "fp": fp, "meta": d.get("meta", {})}


def tier(sim: float) -> str | None:
    if sim >= 0.92:
        return "strong acoustic similarity"
    if sim >= 0.80:
        return "category similarity"
    if sim >= 0.65:
        return "weak resemblance"
    return None


def search_embedding(emb: np.ndarray | None, k: int = 5) -> list[dict]:
    lib = library()
    if emb is None or not len(lib["refs"]):
        return []
    sims = lib["E"] @ emb
    out = []
    for i in np.argsort(-sims)[:k]:
        t = tier(float(sims[i]))
        if t:
            r = lib["refs"][i]
            out.append({"ref_id": r["id"], "title": r["title"], "category": r["category"], "similarity": round(float(sims[i]), 3),
                        "tier": t, "source": r.get("source_url"), "license": r.get("license"), "artist": r.get("artist"),
                        "lat": r.get("lat"), "lon": r.get("lon"), "why": f"cosine similarity of AudioSet embeddings = {sims[i]:.3f}"})
    return out


def search_fingerprint(x8k: np.ndarray) -> list[dict]:
    lib = library()
    if not lib["fp"]:
        return []
    votes: dict[tuple[int, int], int] = {}
    for h, t in fingerprint(x8k):
        for ri, rt in lib["fp"].get(h, []):
            key = (ri, rt - t)
            votes[key] = votes.get(key, 0) + 1
    best: dict[int, tuple[int, int]] = {}
    for (ri, off), n in votes.items():
        if n > best.get(ri, (0, 0))[0]:
            best[ri] = (n, off)
    out = []
    for ri, (n, off) in sorted(best.items(), key=lambda kv: -kv[1][0])[:5]:
        if n >= 20:
            r = lib["refs"][ri]
            out.append({"ref_id": r["id"], "title": r["title"], "category": r["category"], "aligned_hits": n,
                        "offset_s": round(off * 128 / FP_SR, 2), "tier": "near-exact fingerprint match",
                        "source": r.get("source_url"), "license": r.get("license"), "lat": r.get("lat"), "lon": r.get("lon"),
                        "why": f"{n} spectral-landmark pairs aligned at one time offset (same recording content)"})
    return out


def search_events(records: list[dict], sed: dict, x8k: np.ndarray) -> dict:
    from .discovery import _segment_embedding
    lib = library()
    if not lib["refs"]:
        return {"available": False, "reason": "Reference library not bundled."}
    per_event = []
    for r in records:
        if r["duration"] > 0.95 * (sed["ends"].max() if len(sed["ends"]) else 1e9) and r["kind"] == "measurement":
            continue
        emb = _segment_embedding(sed, r["start"], r["end"])
        hits = search_embedding(emb, k=3)
        if hits:
            r["reference_matches"] = hits
            per_event.append({"record": r["id"], "label": r["label"], "matches": hits})
    fp = search_fingerprint(x8k)
    return {"available": True, "library_size": len(lib["refs"]), "library_meta": lib["meta"], "events": per_event,
            "fingerprint_matches": fp, "tiers": ["near-exact fingerprint match", "strong acoustic similarity",
                                                 "category similarity", "weak resemblance"],
            "note": "Similarity is evidence of resemblance only. It never proves the same physical source or the same place."}
