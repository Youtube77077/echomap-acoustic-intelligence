"""BirdNET+ V3 species detection and the BirdNET+ geomodel (species occurrence by place and week)."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .runtime import session

SR = 32000
CHUNK_S = 3.0
RESCALED: list[int] = []  # chunks re-run at reduced gain because of non-finite outputs (logged by the pipeline)


@lru_cache(maxsize=2)
def labels(models_dir: str) -> list[dict]:
    with open(Path(models_dir) / "birdnet_v3_labels.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return [{"idx": int(r["idx"]), "sci": r["sci_name"], "common": r["com_name"], "class": r["class"],
             "order": r["order"]} for r in rows]


def run(x32k: np.ndarray, models_dir: str | Path, hop_s: float = 1.5, progress=None) -> dict:
    sess = session(Path(models_dir) / "birdnet_v3_fp16_pruned.onnx")
    n = int(CHUNK_S * SR)
    hop = int(hop_s * SR)
    starts = list(range(0, max(1, len(x32k) - n + 1), hop))
    if not starts or starts[-1] + n < len(x32k):
        starts.append(max(0, len(x32k) - n))
    preds = []
    emb = []
    for b in range(0, len(starts), 16):
        batch = []
        for s in starts[b:b + 16]:
            c = x32k[s:s + n]
            batch.append(np.pad(c, (0, n - len(c))))
        arr = np.stack(batch).astype(np.float32)
        out = sess.run(None, {"input": arr})
        pr = out[0]
        bad = ~np.isfinite(pr).all(axis=1)
        if bad.any():  # FP16 overflow on very loud input: retry those chunks at reduced gain
            sub = arr[bad] / max(1e-6, float(np.abs(arr[bad]).max())) * 0.25
            pr[bad] = sess.run(None, {"input": sub})[0]
            RESCALED.append(int(bad.sum()))
        preds.append(np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0))
        if len(out) > 1:
            emb.append(out[1])
        if progress:
            progress(min(1.0, (b + 16) / len(starts)))
    st = np.array(starts) / SR
    return {"starts": st, "ends": np.minimum(st + CHUNK_S, len(x32k) / SR), "scores": np.concatenate(preds).astype(np.float32)}


@lru_cache(maxsize=2)
def geo_labels(models_dir: str) -> list[dict]:
    out = []
    for line in (Path(models_dir) / "geomodel_v3.0.4_labels.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 3:
            out.append({"code": f[0], "sci": f[1], "common": f[2]})
    return out


def geo_index(models_dir: str) -> dict[str, int]:
    return {r["sci"].lower(): i for i, r in enumerate(geo_labels(models_dir))}


def geo_occurrence(models_dir: str | Path, lats: np.ndarray, lons: np.ndarray, week: int | float,
                   species_idx: list[int], batch: int = 8192) -> np.ndarray:
    """Occurrence scores for the given geomodel indices at points. week: 1-48, or -1 for the yearly max."""
    sess = session(Path(models_dir) / "geomodel_v3.0.4_fp16.onnx")
    lats = np.asarray(lats, np.float32)
    lons = np.asarray(lons, np.float32)
    weeks = [week] if week and week > 0 else [4 * m - 2 for m in range(1, 13)]  # mid-month weeks
    res = np.zeros((len(lats), len(species_idx)), np.float32)
    for w in weeks:
        for b in range(0, len(lats), batch):
            inp = np.stack([lats[b:b + batch], lons[b:b + batch], np.full(len(lats[b:b + batch]), w, np.float32)], axis=1)
            out = sess.run(None, {"input": inp})[0][:, species_idx]
            res[b:b + batch] = np.maximum(res[b:b + batch], out)
    return res


def geo_weekly_profile(models_dir: str | Path, lat: float, lon: float, species_idx: list[int]) -> np.ndarray:
    """(48, n_species) occurrence by week at one location."""
    sess = session(Path(models_dir) / "geomodel_v3.0.4_fp16.onnx")
    inp = np.array([[lat, lon, w] for w in range(1, 49)], np.float32)
    return sess.run(None, {"input": inp})[0][:, species_idx]
