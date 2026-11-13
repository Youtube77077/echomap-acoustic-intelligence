"""Acoustic signature database and reverse sound search.

Two independent matchers:
1. Landmark fingerprints (spectral-peak pairs hashed with their time offset, Wang 2003 style). Many
   aligned hash hits with one consistent offset means *the same recording* (near-exact match).
2. Embedding similarity (EfficientAT 960-d, cosine). This is acoustic similarity, not identity.

Match tiers (shown with every result):
- near-exact fingerprint match : ≥ 40 aligned landmark hits with one consistent offset, ≥ 10x the background
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
FP_MAX_POSTINGS = 40     # a hash seen more often than this in the library is a stop-hash
FP_MIN_HITS = 40         # aligned landmark pairs for a near-exact match ...
FP_BG_FACTOR = 10.0      # ... and at least this many times the reference's mean votes per offset


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


TRANSIT = DATA.parent / "transit_refs.json.gz"


@lru_cache(maxsize=1)
def library() -> dict:
    """General reference library plus the city transit library (v1.0.3), loaded once."""
    refs, meta = [], {}
    for path in (DATA, TRANSIT):
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                d = json.load(f)
            refs += d["refs"]
            meta[path.name] = d.get("meta", {})
    # evaluation only: leave test recordings out of the library so they cannot match themselves
    import os
    excl = {u.strip() for u in os.environ.get("ECHOMAP_EXCLUDE_REFS", "").split("\n") if u.strip()}
    if excl:
        refs = [r for r in refs if r.get("source_url") not in excl and r.get("title") not in excl]
    if not refs:
        return {"refs": [], "E": np.zeros((0, 960), np.float32), "fp": {}, "meta": {}}
    E = np.array([np.frombuffer(base64.b64decode(r.pop("embedding")), np.float16).astype(np.float32) for r in refs])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    hs, ri, ts = [], [], []
    for i, r in enumerate(refs):
        for h, t in r.pop("fingerprint", []):
            hs.append(h)
            ri.append(i)
            ts.append(t)
    H = np.array(hs, np.int64)
    R = np.array(ri, np.int32)
    T = np.array(ts, np.int32)
    # Stop-hashes (v1.0.8): landmark hashes that occur very often in the library (broadband noise, hum) carry almost no
    # identity information but made the search take minutes. Like other landmark matchers, they are dropped.
    if len(H):
        uh, cnt = np.unique(H, return_counts=True)
        common = uh[cnt > FP_MAX_POSTINGS]
        keep = ~np.isin(H, common)
        H, R, T = H[keep], R[keep], T[keep]
    o = np.argsort(H, kind="stable")
    fp = {"H": H[o], "R": R[o], "T": T[o]}
    return {"refs": refs, "E": E, "fp": fp, "meta": meta}


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
                        "lat": r.get("lat"), "lon": r.get("lon"), "city": r.get("city"), "country": r.get("country"),
                        "location_basis": r.get("location_basis"), "why": f"cosine similarity of AudioSet embeddings = {sims[i]:.3f}"})
    return out


def search_fingerprint(x8k: np.ndarray) -> list[dict]:
    """Vectorised landmark voting (v1.0.8): sorted posting arrays + numpy counting, processed in chunks."""
    lib = library()
    fp = lib["fp"]
    if not len(fp.get("H", [])):
        return []
    q = fingerprint(x8k)
    if not q:
        return []
    qh = np.array([h for h, _ in q], np.int64)
    qt = np.array([t for _, t in q], np.int64)
    H, R, T = fp["H"], fp["R"].astype(np.int64), fp["T"].astype(np.int64)
    best: dict[int, tuple[int, int]] = {}
    acc = []
    for a in range(0, len(qh), 20000):
        h, t = qh[a:a + 20000], qt[a:a + 20000]
        lo = np.searchsorted(H, h, "left")
        hi = np.searchsorted(H, h, "right")
        n = hi - lo
        m = n > 0
        if not m.any():
            continue
        lo, n, t = lo[m], n[m], t[m]
        idx = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(n.sum())   # every posting of every query hash
        off = T[idx] - np.repeat(t, n)
        acc.append(R[idx] * (1 << 24) + (off + (1 << 23)))
    if not acc:
        return []
    keys, counts = np.unique(np.concatenate(acc), return_counts=True)
    refs_of = keys >> 24
    # background per reference: mean votes per occupied offset. Random landmark collisions pile up on long
    # recordings, so a real match must stand far above this background, not just pass a fixed count.
    ur, first, n_offs = np.unique(refs_of, return_index=True, return_counts=True)
    tot = np.add.reduceat(counts, first)
    bg = dict(zip(ur.tolist(), (tot / n_offs).tolist()))
    for k, c in zip(keys[counts >= 5], counts[counts >= 5]):
        ri, off = int(k >> 24), int((k & ((1 << 24) - 1)) - (1 << 23))
        if c > best.get(ri, (0, 0, 0))[0]:
            best[ri] = (int(c), off, bg[ri])
    best = {ri: (n, off) for ri, (n, off, b) in best.items() if n >= FP_MIN_HITS and n >= FP_BG_FACTOR * b}
    out = []
    for ri, (n, off) in sorted(best.items(), key=lambda kv: -kv[1][0])[:5]:
        if n >= FP_MIN_HITS:
            r = lib["refs"][ri]
            out.append({"ref_id": r["id"], "title": r["title"], "category": r["category"], "aligned_hits": n,
                        "offset_s": round(off * 128 / FP_SR, 2), "tier": "near-exact fingerprint match",
                        "source": r.get("source_url"), "license": r.get("license"), "lat": r.get("lat"), "lon": r.get("lon"),
                        "city": r.get("city"), "country": r.get("country"),
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
