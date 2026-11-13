"""Embed a per-species sample of Xeno-canto recordings for EchoMap's animal head (CI only; audio never stored).

Usage: python tools/xc_embed.py <models_dir> <xc_meta.jsonl.gz> <shard> <n_shards> <out.npz> [per_species]

Per species: up to `per_species` recordings, best quality first (A, B, ...), spread over countries. Only the first
~45 s of each file are fetched (HTTP range request). Embeddings as in tools/inat_embed.py (EfficientAT 960-d on the
loudest 5 s windows + BirdNET+ 1280-d on their central 3 s), with species name, lat, lon and month per window.
Recordings are CC-licensed by their recordists (licence kept per row); used offline for personal, non-commercial use.
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.ingest.media import decode_mono  # noqa: E402
from echomap.models.frontends import efficientat_mel  # noqa: E402
from echomap.models.runtime import session  # noqa: E402
from tools.inat_embed import BN, WIN, windows  # noqa: E402

S = requests.Session()
S.headers["User-Agent"] = "EchoMap-xc-embed/1.0 (training an offline species classifier; partial downloads)"
QRANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def select(meta_path: str, per_species: int) -> list[dict]:
    by = defaultdict(list)
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("file") and "-nd" not in str(r.get("lic", "")).lower():   # no-derivatives licences excluded
                by[r["sci"].lower()].append(r)
    out = []
    for sp, rs in sorted(by.items()):
        rs.sort(key=lambda r: (QRANK.get(r.get("q"), 5), -int(r.get("id") or 0)))
        seen, pick = set(), []
        for r in rs:                       # one per country first, then the rest
            if r.get("cnt") not in seen:
                pick.append(r)
                seen.add(r.get("cnt"))
            if len(pick) >= per_species:
                break
        for r in rs:
            if len(pick) >= per_species:
                break
            if r not in pick:
                pick.append(r)
        out += pick
    return out


def main():
    md, meta, shard, nsh, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    per = int(sys.argv[6]) if len(sys.argv) > 6 else 25
    items = [r for i, r in enumerate(select(meta, per)) if i % nsh == shard]
    print(f"shard {shard}/{nsh}: {len(items)} recordings", flush=True)
    melbank = np.load(Path(md) / "efficientat_mn10_as_melbank.npy")
    s_eat = session(Path(md) / "efficientat_mn10_as.onnx")
    s_bn = session(Path(md) / "birdnet_v3_fp16_pruned.onnx")
    E1, E2, Y, LAT, LON, MON, RID = [], [], [], [], [], [], []
    tmp = Path(tempfile.mkdtemp())
    t0, ok, bad = time.time(), 0, 0
    for k, r in enumerate(items):
        try:
            resp = S.get(r["file"], headers={"Range": "bytes=0-900000"}, timeout=60)
            if resp.status_code not in (200, 206):
                bad += 1
                continue
            f = tmp / "a.bin"
            f.write_bytes(resp.content)
            x, _, _ = decode_mono(f, 32000, max_seconds=45)
        except Exception:  # noqa: BLE001  (truncated MP3 tails decode fine; anything else is skipped)
            bad += 1
            continue
        if len(x) < 16000:
            bad += 1
            continue
        ws = windows(x, 2)
        mel = np.stack([efficientat_mel(w, melbank)[:, :500] for w in ws])[:, None].astype(np.float32)
        e1 = s_eat.run(["probs", "embedding"], {"mel": mel})[1].reshape(len(ws), -1)
        bn_in = np.stack([w[(WIN - BN) // 2:(WIN - BN) // 2 + BN] for w in ws]).astype(np.float32)
        o = s_bn.run(None, {"input": bn_in})
        e2 = np.nan_to_num(o[1].reshape(len(ws), -1)) if len(o) > 1 else np.zeros((len(ws), 1))
        try:
            mon = int(str(r.get("date", ""))[5:7])
        except ValueError:
            mon = 0
        for j in range(len(ws)):
            E1.append(e1[j].astype(np.float16))
            E2.append(e2[j].astype(np.float16))
            Y.append(r["sci"].lower())
            LAT.append(float(r["lat"]) if r.get("lat") not in (None, "") else np.nan)
            LON.append(float(r["lon"]) if r.get("lon") not in (None, "") else np.nan)
            MON.append(mon)
            RID.append(int(r.get("id") or 0))
        ok += 1
        if k % 1000 == 0:
            print(f"  {k}/{len(items)}: ok {ok}, skipped {bad}, {time.time() - t0:.0f} s", flush=True)
        time.sleep(0.2)
    np.savez_compressed(out, eat=np.stack(E1), bn=np.stack(E2), yname=np.array(Y), lat=np.array(LAT, np.float32),
                        lon=np.array(LON, np.float32), month=np.array(MON, np.int8),
                        audio_id=np.array(RID, np.int64))
    print(f"DONE shard {shard}: {ok} recordings, {len(Y)} windows, skipped {bad}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
