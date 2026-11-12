"""Train EchoMap's vehicle-type head (car / bus / truck / tram / motorcycle / bicycle) on MELAUDIS (CI only).

Usage: python tools/train_vehicle_head.py <models_dir> <work_dir>
Data:  MELAUDIS_Vehicles (Figshare 27115870, CC BY 4.0), recorded at 20 roadside collection points in Melbourne.
Output: echomap/data/vehicle_head.json  (multinomial logistic regression on EfficientAT 960-d embeddings, ~25 KB)
        docs/eval/vehicle_head.json      (held-out accuracy, per-class precision/recall, confusion matrix)

Evaluation is grouped by collection point: whole recording sites are held out, so the score reflects new
places, not new clips from a site the model already heard. All recordings are from one city (Melbourne), so
accuracy elsewhere, with other tram and bus models, is expected to be lower; the app says so.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echomap.ingest.media import decode_mono  # noqa: E402
from echomap.models import sed  # noqa: E402

URL = "https://ndownloader.figshare.com/files/49436053"
CLASSES = ["car", "bus", "truck", "tram", "motorcycle", "bicycle"]
PATTERNS = [("motorcycle", r"motor ?cycle|motorbike|\bmc\b|\bmoto"), ("bicycle", r"bicycle|\bbike|cycl"),
            ("tram", r"tram|streetcar"), ("bus", r"\bbus"), ("truck", r"truck|lorry|\bhgv|heavy"), ("car", r"\bcar|sedan|\bsuv")]
CAP = int(os.environ.get("VEH_CAP", "900"))


def label_of(path: str) -> str | None:
    p = path.lower().replace("_", " ").replace("-", " ")
    for c, rx in PATTERNS:
        if re.search(rx, p):
            return c
    return None


def site_of(rel: Path) -> str:
    return rel.parts[1] if len(rel.parts) > 2 else rel.parts[0]


def main():
    models, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    rar = work / "vehicles.rar"
    if not rar.exists():
        with requests.get(URL, stream=True, timeout=600, headers={"User-Agent": "EchoMap-trainer/1.0"}) as r:
            r.raise_for_status()
            with open(rar, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
    print("downloaded", round(rar.stat().st_size / 1e9, 2), "GB")
    out = work / "x"
    subprocess.run(["7z", "x", "-y", f"-o{out}", str(rar)], check=True, stdout=subprocess.DEVNULL)
    rar.unlink()
    wavs = [p for p in out.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")]
    print("audio files:", len(wavs))
    dirs = Counter(str(p.relative_to(out).parent) for p in wavs)
    print("folders (sample):", list(dirs.items())[:25])
    print("file names (sample):", [p.name for p in wavs[:15]])
    items = [(p, label_of(str(p.relative_to(out))), site_of(p.relative_to(out))) for p in wavs]
    print("label counts:", Counter(l for _, l, _ in items))
    items = [it for it in items if it[1]]
    if len({l for _, l, _ in items}) < 3:
        raise SystemExit("could not infer vehicle classes from paths; inspect the printed structure")
    rng = np.random.default_rng(0)
    by = defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    chosen = []
    for c, its in by.items():
        idx = rng.permutation(len(its))[:CAP]
        chosen += [its[i] for i in idx]
    X, y, g = [], [], []
    t0 = time.time()
    for k, (p, lab, site) in enumerate(chosen):
        try:
            x, sr, _ = decode_mono(p, 32000)
        except Exception as e:  # noqa: BLE001
            print("skip", p.name, e)
            continue
        if len(x) < sr * 0.5:
            continue
        r = sed.run(x[: sr * 20], models)
        e = r["embeddings"].mean(axis=0)
        X.append(e / (np.linalg.norm(e) + 1e-9))
        y.append(CLASSES.index(lab))
        g.append(site)
        if k % 200 == 0:
            print(f"{k}/{len(chosen)} embedded, {int(time.time() - t0)} s")
    X, y, g = np.array(X, np.float32), np.array(y), np.array(g)
    sites = sorted(set(g))
    test_sites = set(rng.permutation(sites)[: max(3, len(sites) // 5)])
    te = np.array([s in test_sites for s in g])
    mu = X[~te].mean(0)
    present = sorted(set(y.tolist()))

    def train(Xt, yt, epochs=400, lr=0.5, l2=1e-3):
        C = len(CLASSES)
        W = np.zeros((Xt.shape[1], C), np.float64)
        b = np.zeros(C)
        cw = np.array([len(yt) / (C * max(1, (yt == c).sum())) for c in range(C)])  # class balance
        Y = np.eye(C)[yt]
        for _ in range(epochs):
            Z = (Xt - mu) @ W + b
            Z -= Z.max(1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(1, keepdims=True)
            G = (P - Y) * cw[yt][:, None] / len(yt)
            W -= lr * ((Xt - mu).T @ G + l2 * W)
            b -= lr * G.sum(0)
        return W, b

    W, b = train(X[~te], y[~te])
    pred = np.argmax((X[te] - mu) @ W + b, 1)
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p_ in zip(y[te], pred):
        conf[t, p_] += 1
    per = {}
    for c in present:
        tp = conf[c, c]
        per[CLASSES[c]] = {"precision": round(float(tp / max(1, conf[:, c].sum())), 3),
                           "recall": round(float(tp / max(1, conf[c].sum())), 3), "test_n": int(conf[c].sum())}
    acc = float((pred == y[te]).mean())
    metrics = {"held_out_sites": sorted(test_sites), "test_n": int(te.sum()), "train_n": int((~te).sum()),
               "accuracy": round(acc, 3), "balanced_accuracy": round(float(np.mean([per[k]["recall"] for k in per])), 3),
               "per_class": per, "confusion": {"classes": CLASSES, "matrix": conf.tolist()},
               "note": "Grouped by recording site (whole sites held out). Melbourne only: expect lower accuracy elsewhere."}
    print(json.dumps(metrics, indent=1))
    W, b = train(X, y)  # final model on all sites
    head = {"classes": CLASSES, "W": np.round(W, 5).tolist(), "b": np.round(b, 5).tolist(), "mu": np.round(mu, 6).tolist(),
            "embedding": "EfficientAT mn10_as 960-d, mean over 5 s windows, L2-normalised",
            "data": "MELAUDIS_Vehicles, Figshare 27115870, CC BY 4.0 (Melbourne, 20 roadside sites)", "metrics": metrics,
            "built": time.strftime("%Y-%m-%d")}
    (ROOT / "echomap" / "data" / "vehicle_head.json").write_text(json.dumps(head, separators=(",", ":")), encoding="utf-8")
    (ROOT / "docs" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "eval" / "vehicle_head.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
