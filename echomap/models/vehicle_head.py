"""Vehicle-type head (v1.0.3): car / bus / truck / tram / motorcycle / bicycle from EfficientAT embeddings.

A multinomial logistic regression trained on MELAUDIS (CC BY 4.0, 20 roadside sites in Melbourne) by
tools/train_vehicle_head.py, evaluated with whole recording sites held out (docs/eval/vehicle_head.json).
It only runs on events the sound model already marked as vehicle or rail sounds. All training audio is from one
city, so its accuracy elsewhere (other tram and bus models) is expected to be lower than the reported figure.
Vehicle make or model is not attempted: no open labelled data supports it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "vehicle_head.json"
VEHICLE_LABELS = {"Vehicle", "Motor vehicle (road)", "Car", "Car passing by", "Bus", "Truck", "Motorcycle", "Rail transport",
                  "Train", "Railroad car, train wagon", "Train wheels squealing", "Subway, metro, underground", "Bicycle",
                  "Traffic noise, roadway noise", "Engine", "Light engine (high frequency)", "Medium engine (mid frequency)",
                  "Heavy engine (low frequency)", "Idling", "Accelerating, revving, vroom", "Tire squeal", "Skidding"}


@lru_cache(maxsize=1)
def head() -> dict | None:
    if not DATA.exists():
        return None
    h = json.loads(DATA.read_text(encoding="utf-8"))
    h["W"], h["b"], h["mu"] = np.array(h["W"], np.float32), np.array(h["b"], np.float32), np.array(h["mu"], np.float32)
    return h


def classify(emb: np.ndarray) -> dict | None:
    h = head()
    if h is None or emb is None:
        return None
    z = (emb - h["mu"]) @ h["W"] + h["b"]
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)
    return {"top": h["classes"][order[0]], "p": round(float(p[order[0]]), 3),
            "probs": {h["classes"][i]: round(float(p[i]), 3) for i in order},
            "model": "vehicle-type head (MELAUDIS, Melbourne)", "held_out_accuracy": h.get("metrics", {}).get("balanced_accuracy")}


def annotate(records: list[dict], sed: dict) -> int:
    """Attach evidence['vehicle_type'] to vehicle/rail events. Returns how many were classified."""
    from ..detect.discovery import _segment_embedding
    if head() is None:
        return 0
    n = 0
    for r in records:
        names = {r["label"]} | set(r.get("evidence", {}).get("ontology_parents_detected", []))
        if names & VEHICLE_LABELS:
            res = classify(_segment_embedding(sed, r["start"], r["end"]))
            if res:
                r.setdefault("evidence", {})["vehicle_type"] = res
                n += 1
    return n


def tram_precision() -> float | None:
    h = head()
    return ((h or {}).get("metrics", {}).get("per_class", {}).get("tram") or {}).get("precision")
