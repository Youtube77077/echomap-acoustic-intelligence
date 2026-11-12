"""AudioSet sound event detection with EfficientAT over sliding windows.

Window length: 5 s (500 mel frames). Testing found the exported mn10 graph gives saturated,
meaningless outputs for inputs of 300 frames or fewer (3 s), while ≥ 320 frames behaves normally
(see docs/TEST_RESULTS.md). 5 s is also the clip length of ESC-50, a common downstream use.

Time localisation: each 0.5 s frame receives the 25th percentile of the scores of all windows that
cover it. A short event inside overlapping windows therefore stays high only near its true position,
which gives about 0.5 s resolution without frame-level labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .frontends import EAT_HOP, efficientat_mel
from .runtime import session

WIN_S = 5.0
SR = 32000


def load_labels(models_dir: str | Path) -> list[dict]:
    with open(Path(models_dir) / "audioset_labels.csv", encoding="utf-8") as fh:
        return [{"idx": int(r["index"]), "mid": r["mid"], "name": r["display_name"]} for r in csv.DictReader(fh)]


def run(x32k: np.ndarray, models_dir: str | Path, hop_s: float = 0.5, progress=None) -> dict:
    md = Path(models_dir)
    melbank = np.load(md / "efficientat_mn10_as_melbank.npy")
    sess = session(md / "efficientat_mn10_as.onnx")
    dur = len(x32k) / SR
    x = x32k if dur >= WIN_S else np.pad(x32k, (0, int(WIN_S * SR) - len(x32k)))  # zero-pad short recordings
    mel = efficientat_mel(x, melbank)  # (128, T), 100 frames/s
    fps = SR / EAT_HOP
    win = int(WIN_S * fps)
    hop = max(1, int(round(hop_s * fps)))
    T = mel.shape[1]
    starts = list(range(0, max(1, T - win + 1), hop))
    if starts[-1] + win < T:
        starts.append(T - win)
    scores = []
    embs = []
    batch = 16
    for b in range(0, len(starts), batch):
        chunk = np.stack([mel[:, s:s + win] for s in starts[b:b + batch]])[:, None].astype(np.float32)
        out, emb = sess.run(["probs", "embedding"], {"mel": chunk})
        scores.append(out.reshape(-1, out.shape[-1]))  # exported graph keeps a leading singleton dim
        embs.append(emb.reshape(-1, emb.shape[-1]).astype(np.float16))
        if progress:
            progress(min(1.0, (b + batch) / len(starts)))
    sc = np.concatenate(scores).astype(np.float32)
    st = np.array(starts) / fps
    en = st + WIN_S
    # frame-level pooling on a hop_s grid
    n_frames = max(1, int(np.ceil(dur / hop_s)))
    ft = np.arange(n_frames) * hop_s
    frame_scores = np.zeros((n_frames, sc.shape[1]), np.float32)
    for i, t in enumerate(ft):
        c = t + hop_s / 2
        cover = (st <= c) & (en >= c)
        if not cover.any():
            cover = np.abs((st + en) / 2 - c) == np.abs((st + en) / 2 - c).min()
        frame_scores[i] = np.percentile(sc[cover], 25, axis=0)
    return {"starts": st, "ends": np.minimum(en, max(dur, WIN_S)), "scores": sc, "hop_s": hop_s, "win_s": WIN_S,
            "embeddings": np.concatenate(embs).astype(np.float32),
            "frame_t": ft, "frame_scores": frame_scores, "padded_to_5s": dur < WIN_S}
