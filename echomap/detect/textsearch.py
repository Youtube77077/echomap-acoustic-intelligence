"""Text search over the recording using AudioSet label semantics.

This is NOT a CLAP-style audio-text model. The query is matched to AudioSet class names and descriptions
by word overlap with simple synonym expansion, and the recording's frame scores for those classes are
searched. It works for sounds covered by the AudioSet ontology. A CLAP-based search would generalise
further, which is why the result says which classes the query was mapped to.
"""
from __future__ import annotations

import re

import numpy as np

from .taxonomy import class_table

SYN = {"church": ["bell"], "chime": ["bell", "chime"], "diesel": ["engine", "train", "truck"], "tram": ["train", "rail"],
       "metro": ["subway"], "underground": ["subway"], "crossing": ["beep", "alarm", "bell"], "pedestrian": ["beep"],
       "motor": ["engine"], "car": ["car", "vehicle"], "bird": ["bird"], "birds": ["bird"], "forest": ["bird", "rustling"],
       "plane": ["aircraft"], "airplane": ["aircraft"], "overhead": ["aircraft", "helicopter"], "electric": ["hum", "buzz"],
       "people": ["speech", "crowd"], "talking": ["speech", "conversation"], "water": ["water", "stream"], "sea": ["waves", "ocean"]}
STOP = {"a", "an", "the", "in", "on", "of", "over", "at", "with", "several", "some", "distance", "distant", "passing", "is",
        "and", "sound", "noise", "ringing", "calling", "flying", "accelerating", "railway", "tracks"}


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", s.lower()) if t not in STOP]


def search(query: str, sed: dict, labels: list[str], top_classes: int = 6, thr: float = 0.1) -> dict:
    q = _tokens(query)
    q += [w for t in q for w in SYN.get(t, [])]
    if not q:
        return {"query": query, "mapped_classes": [], "intervals": []}
    table = class_table()
    scores = []
    for c in table:
        name = set(_tokens(c["name"]))
        desc = set(_tokens(c.get("description", "")))
        s = sum(2.0 for t in q if t in name) + sum(0.3 for t in q if t in desc)
        if s > 0:
            scores.append((s, c["idx"], c["name"]))
    scores.sort(reverse=True)
    mapped = scores[:top_classes]
    if not mapped:
        return {"query": query, "mapped_classes": [], "intervals": [],
                "note": "No AudioSet class matches the query words."}
    fs, hop = sed["frame_scores"], sed["hop_s"]
    w = np.array([s for s, _, _ in mapped], np.float32)
    w /= w.sum()
    comb = (fs[:, [i for _, i, _ in mapped]] * w).sum(axis=1)
    best = fs[:, [i for _, i, _ in mapped]].max(axis=1)
    ints = []
    on = best >= thr
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            k = i + int(np.argmax(best[i:j]))
            cls = mapped[int(np.argmax(fs[k, [ii for _, ii, _ in mapped]]))][2]
            ints.append({"start": round(i * hop, 2), "end": round(j * hop, 2), "score": round(float(best[i:j].max()), 3),
                         "weighted_score": round(float(comb[i:j].max()), 3), "best_class": cls})
            i = j
        else:
            i += 1
    ints.sort(key=lambda d: -d["score"])
    return {"query": query, "mapped_classes": [{"name": n, "match_weight": round(float(s), 2)} for s, _, n in mapped],
            "intervals": ints[:20], "method": __doc__.strip()}
