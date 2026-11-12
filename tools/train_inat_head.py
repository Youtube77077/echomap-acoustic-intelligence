"""Train EchoMap's animal-species head on iNatSounds embeddings (CI only). Usage: python tools/train_inat_head.py <emb_dir>

Input : <emb_dir>/train_*.npz, val_*.npz and train_0.categories.json from tools/inat_embed.py
Output: echomap/data/inat_head.npz   (PCA projection + softmax regression, float16)
        docs/eval/inat_head.json     (held-out accuracy on the iNatSounds validation split, per taxon group,
                                      precision at confidence thresholds, which feature set won)

Three feature sets are compared: EfficientAT (960-d, available in every analysis mode), BirdNET+ (1280-d, only when
BirdNET runs) and both concatenated. The EfficientAT head is always shipped (works in Quick); the BirdNET-based head is
shipped too when it beats it by >= 3 points top-1. Evaluation is per recording (logits averaged over its windows) on
the official validation split, which contains different recordings from the training split.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIM = int(__import__("os").environ.get("INAT_DIM", "320"))


def load(pattern):
    parts = [np.load(f) for f in sorted(glob.glob(pattern))]
    if not parts:
        raise SystemExit(f"no files for {pattern}")
    return {k: np.concatenate([p[k] for p in parts]) for k in ("eat", "bn", "y", "audio_id", "license")}


def l2(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def feats(d, kind):
    if kind == "eat":
        return l2(d["eat"])
    if kind == "bn":
        return l2(d["bn"])
    return np.hstack([l2(d["eat"]), l2(d["bn"])])


def fit_pca(X, dim, rng):
    sub = X[rng.permutation(len(X))[:60000]]
    mu = sub.mean(0)
    _, _, vt = np.linalg.svd(sub - mu, full_matrices=False)
    P = vt[:dim].T.astype(np.float32)
    Z = (sub - mu) @ P
    sd = Z.std(0) + 1e-6
    return mu.astype(np.float32), P / sd   # whitened projection


def train_softmax(Z, y, n_cls, epochs=30, bs=4096, lr=3e-3, wd=1e-5, rng=None):
    n, d = Z.shape
    W = np.zeros((d, n_cls), np.float32)
    b = np.zeros(n_cls, np.float32)
    cnt = np.bincount(y, minlength=n_cls).astype(np.float32)
    cw = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0).astype(np.float32)
    cw /= (cw[y].mean())
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        loss = 0.0
        cur_lr = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            X, yy = Z[idx], y[idx]
            logits = X @ W + b
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            w = cw[yy][:, None]
            loss += float(-(np.log(p[np.arange(len(yy)), yy] + 1e-9) * w[:, 0]).sum())
            p[np.arange(len(yy)), yy] -= 1
            g = p * w / len(yy)
            gW = X.T @ g + wd * W
            gb = g.sum(0)
            t += 1
            for P_, G, M, V in ((W, gW, mW, vW), (b, gb, mb, vb)):
                M *= 0.9
                M += 0.1 * G
                V *= 0.999
                V += 0.001 * G * G
                P_ -= cur_lr * (M / (1 - 0.9 ** t)) / (np.sqrt(V / (1 - 0.999 ** t)) + 1e-8)
        print(f"  epoch {ep + 1}/{epochs} loss {loss / n:.3f}", flush=True)
    return W, b


def evaluate(Zv, yv, aid, W, b, groups, names):
    logits = Zv @ W + b
    rec = defaultdict(list)
    for i, a in enumerate(aid):
        rec[a].append(i)
    L = np.stack([logits[ix].mean(0) for ix in rec.values()])
    Y = np.array([yv[ix[0]] for ix in rec.values()])
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(1, keepdims=True)
    top5 = np.argsort(-P, axis=1)[:, :5]
    top1 = top5[:, 0]
    pmax = P.max(1)
    res = {"recordings": int(len(Y)), "top1": round(float((top1 == Y).mean()), 4),
           "top5": round(float((top5 == Y[:, None]).any(1).mean()), 4), "by_group": {}, "precision_at": {}}
    for g in sorted(set(groups[y] for y in Y)):
        m = np.array([groups[y] == g for y in Y])
        res["by_group"][g] = {"recordings": int(m.sum()), "top1": round(float((top1[m] == Y[m]).mean()), 4),
                              "top5": round(float((top5[m] == Y[m][:, None]).any(1).mean()), 4)}
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = pmax >= thr
        res["precision_at"][str(thr)] = {"coverage": round(float(m.mean()), 4),
                                         "precision": round(float((top1[m] == Y[m]).mean()), 4) if m.any() else None}
    conf = Counter((names[y], names[t]) for y, t in zip(Y, top1) if y != t)
    res["top_confusions"] = [f"{a} -> {b}: {n}" for (a, b), n in conf.most_common(15)]
    return res


def main():
    ed = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    tr, va = load(str(ed / "train_*.npz")), load(str(ed / "val_*.npz"))
    meta = json.loads((ed / "train_0.categories.json").read_text(encoding="utf-8"))
    cats = sorted(meta["categories"], key=lambda c: c["id"])
    cid = {c["id"]: i for i, c in enumerate(cats)}
    names = [c.get("name", str(c["id"])) for c in cats]
    common = [c.get("common_name") or "" for c in cats]
    groups = [c.get("supercategory") or c.get("class") or c.get("iconic_taxon_name") or "" for c in cats]
    ytr = np.array([cid[v] for v in tr["y"]], np.int32)
    keep_v = np.array([v in cid for v in va["y"]])
    yva = np.array([cid[v] for v in va["y"][keep_v]], np.int32)
    print(f"train windows {len(ytr)} ({len(set(tr['audio_id']))} recordings, {len(set(ytr))} species); "
          f"val windows {len(yva)}; categories {len(cats)}; sample category {cats[0]}", flush=True)
    has_bn = tr["bn"].shape[1] > 1
    kinds = ["eat"] + (["bn", "both"] if has_bn else [])
    report = {"dataset": "iNatSounds 2024 (visipedia/inat_sounds), non-commercial research/education terms",
              "training_recordings": int(len(set(tr["audio_id"]))), "training_windows": int(len(ytr)),
              "species": len(cats), "licence_ids_train": dict(Counter(map(int, tr["license"])).most_common()),
              "licenses": meta.get("licenses", []), "feature_sets": {}}
    heads = {}
    for kind in kinds:
        t0 = time.time()
        Xtr = feats(tr, kind)
        mu, P = fit_pca(Xtr, DIM, rng)
        Ztr = (Xtr - mu) @ P
        del Xtr
        W, b = train_softmax(Ztr, ytr, len(cats), rng=rng)
        del Ztr
        Zva = (feats({k: v[keep_v] for k, v in va.items()}, kind) - mu) @ P
        ev = evaluate(Zva, yva, va["audio_id"][keep_v], W, b, groups, names)
        ev["train_seconds"] = round(time.time() - t0)
        report["feature_sets"][kind] = ev
        heads[kind] = (mu, P, W, b)
        print(kind, json.dumps({k: ev[k] for k in ("top1", "top5", "by_group")}), flush=True)
    best_bn = max((k for k in kinds if k != "eat"), key=lambda k: report["feature_sets"][k]["top1"], default=None)
    ship = ["eat"]
    if best_bn and report["feature_sets"][best_bn]["top1"] >= report["feature_sets"]["eat"]["top1"] + 0.03:
        ship.append(best_bn)
    report["shipped"] = ship
    out = {"classes": np.array(names), "common": np.array(common), "groups": np.array(groups),
           "shipped": np.array(ship), "dim": np.array(DIM)}
    for k in ship:
        mu, P, W, b = heads[k]
        out.update({f"{k}_mu": mu.astype(np.float16), f"{k}_P": P.astype(np.float16), f"{k}_W": W.astype(np.float16),
                    f"{k}_b": b.astype(np.float32)})
        pa = report["feature_sets"][k]["precision_at"]
        out[f"{k}_thr"] = np.array(min((float(t) for t, v in pa.items() if v["precision"] and v["precision"] >= 0.7),
                                       default=0.8))
    np.savez_compressed(ROOT / "echomap" / "data" / "inat_head.npz", **out)
    (ROOT / "docs" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "eval" / "inat_head.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SHIPPED", ship, "size MB", round((ROOT / "echomap" / "data" / "inat_head.npz").stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
