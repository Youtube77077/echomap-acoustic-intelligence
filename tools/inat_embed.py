"""Embed iNatSounds 2024 recordings for EchoMap's animal-species head (CI only; recordings are never stored or committed).

Usage: python tools/inat_embed.py <models_dir> <split: train|val> <shard> <n_shards> <out.npz> [max_windows_per_file]

The split archive is streamed from the public iNatSounds bucket (curl | pigz | tar stream). Every n-th recording
(index % n_shards == shard) is decoded in memory. Its loudest non-overlapping 5 s windows (up to max_windows, first 60 s)
get two embeddings: EfficientAT mn10 (960-d, 5 s window) and BirdNET+ V3 (1280-d, the central 3 s). Only embeddings,
species ids and licence ids are written. iNatSounds terms: non-commercial research/education, recordings not
redistributed (https://github.com/visipedia/inat_sounds). Approved by the user for personal, non-commercial use.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.ingest.media import resample  # noqa: E402
from echomap.models.frontends import efficientat_mel  # noqa: E402
from echomap.models.runtime import session  # noqa: E402

BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/"
SR = 32000
WIN = 5 * SR
BN = 3 * SR


def load_meta(split: str) -> dict:
    r = requests.get(BASE + f"{split}.json.tar.gz", timeout=300)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        m = next(x for x in tf if x.isfile() and x.name.endswith(".json"))
        return json.load(tf.extractfile(m))


def read_wav(data: bytes) -> np.ndarray | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(min(n, sr * 60))
    except (wave.Error, EOFError):
        return None
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        return None
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, sr, SR) if sr != SR else x


def windows(x: np.ndarray, k: int) -> list[np.ndarray]:
    if len(x) < WIN:
        return [np.pad(x, (0, WIN - len(x)))]
    n = len(x) // WIN
    e = [float(np.mean(x[i * WIN:(i + 1) * WIN] ** 2)) for i in range(n)]
    return [x[i * WIN:(i + 1) * WIN] for i in np.argsort(e)[::-1][:k]]


def main():
    md, split, shard, nsh, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    kmax = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    meta = load_meta(split)
    cat_of = {a["audio_id"]: a["category_id"] for a in meta["annotations"]}
    by_name = {}
    for a in meta["audio"]:
        by_name[Path(a["file_name"]).name] = (a["id"], cat_of.get(a["id"]), a.get("license"))
    print(f"{split}: {len(meta['audio'])} recordings, {len(meta['categories'])} categories, licences "
          f"{[(l.get('id'), l.get('name')) for l in meta.get('licenses', [])]}", flush=True)
    melbank = np.load(Path(md) / "efficientat_mn10_as_melbank.npy")
    s_eat = session(Path(md) / "efficientat_mn10_as.onnx")
    s_bn = session(Path(md) / "birdnet_v3_fp16_pruned.onnx")
    print("BirdNET outputs:", [(o.name, o.shape) for o in s_bn.get_outputs()], flush=True)
    E1, E2, Y, A, L = [], [], [], [], []
    pend = []   # (windows, audio_id, cat, lic)

    def flush():
        if not pend:
            return
        wins = [w for p in pend for w in p[0]]
        mel = np.stack([efficientat_mel(w, melbank)[:, :500] for w in wins])[:, None].astype(np.float32)
        e1 = np.concatenate([s_eat.run(["probs", "embedding"], {"mel": mel[i:i + 16]})[1].reshape(-1, 960)
                             for i in range(0, len(mel), 16)])
        bn_in = np.stack([w[(WIN - BN) // 2:(WIN - BN) // 2 + BN] for w in wins]).astype(np.float32)
        outs = [s_bn.run(None, {"input": bn_in[i:i + 16]}) for i in range(0, len(bn_in), 16)]
        e2 = np.concatenate([o[1] for o in outs]).reshape(len(wins), -1) if len(outs[0]) > 1 else np.zeros((len(wins), 1))
        e2 = np.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)
        j = 0
        for ws, aid, cat, lic in pend:
            for _ in ws:
                E1.append(e1[j].astype(np.float16))
                E2.append(e2[j].astype(np.float16))
                Y.append(cat)
                A.append(aid)
                L.append(lic if lic is not None else -1)
                j += 1
        pend.clear()

    url = BASE + f"{split}.tar.gz"
    p = subprocess.Popen(f"curl -sSfL --retry 5 '{url}' | pigz -dc", shell=True, stdout=subprocess.PIPE, bufsize=1 << 20)
    t0, idx, done, skipped = time.time(), -1, 0, Counter()
    seen_names: list[str] = []
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".wav"):
                continue
            idx += 1
            if idx == 300 and not any(by_name.get(Path(n).name) for n in seen_names):
                raise SystemExit(f"archive names do not match the metadata, e.g. {seen_names[:3]} vs "
                                 f"{list(by_name)[:3]}")
            if idx < 300:
                seen_names.append(m.name)
            if idx % nsh != shard:
                continue
            info = by_name.get(Path(m.name).name)
            if not info or info[1] is None:
                skipped["no_annotation"] += 1
                continue
            x = read_wav(tf.extractfile(m).read())
            if x is None or len(x) < SR // 2:
                skipped["unreadable_or_short"] += 1
                continue
            pend.append((windows(x, kmax), *info))
            done += 1
            if len(pend) >= 32:
                flush()
            if done % 2000 == 0:
                print(f"{done} recordings, {len(Y)} windows, {time.time() - t0:.0f} s", flush=True)
    flush()
    p.wait()
    cats = {c["id"]: c for c in meta["categories"]}
    np.savez_compressed(out, eat=np.stack(E1), bn=np.stack(E2), y=np.array(Y, np.int32), audio_id=np.array(A, np.int64),
                        license=np.array(L, np.int32))
    if shard == 0:
        Path(out.with_suffix(".categories.json")).write_text(json.dumps(
            {"categories": list(cats.values()), "licenses": meta.get("licenses", [])}, ensure_ascii=False))
    print(f"DONE {split} shard {shard}/{nsh}: {done} recordings, {len(Y)} windows, skipped {dict(skipped)}, "
          f"licences {Counter(L).most_common()}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
