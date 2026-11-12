"""Download EchoMap's models into a directory, verifying every checksum.  Used by CI runners
(never needs to run on the user's PC).  Usage: python tools/fetch_models.py <dest> [--langid]

Sources: official Zenodo / GitHub releases for BirdNET+, and this repository's own `models-v1`
release (ONNX conversions made by convert-models.yml) fetched with the runner's GITHUB_TOKEN via `gh`.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import requests

OFFICIAL = [
    ("https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx?download=1",
     "birdnet_v3_fp16_pruned.onnx", "md5", "8278be6e7d1a9fc1717cf953e269b2e3"),
    ("https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1",
     "birdnet_v3_labels.csv", "md5", "21ccf9e984a955bfc2d5c52faab6a6ea"),
    ("https://zenodo.org/records/20703646/files/TERMS_OF_USE.txt?download=1", "birdnet_v3_TERMS_OF_USE.txt", "md5",
     "d5949262d80b5749b49927767579b5f2"),
    ("https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET%2B_Geomodel_V3.0.4_Global_14K_FP16.onnx",
     "geomodel_v3.0.4_fp16.onnx", "sha256", "80e410f811bf4455e6cb1a2d48e394737d3828fe943a444646778495b67e7bab"),
    ("https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET%2B_Geomodel_V3.0.4_Global_14K_Labels.txt",
     "geomodel_v3.0.4_labels.txt", "sha256", "8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"),
    ("https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/LICENSE-MODELS.md",
     "geomodel_LICENSE-MODELS.md", "sha256", "c61a126ce3b9005b3ec0a74bd921f31214f0bec564f9067305d59e7c007796ca"),
]


def digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "EchoMap-CI"
    tag = os.environ.get("ECHOMAP_MODELS_TAG", "models-v1")
    # primary source: this repository's private release (mirrors the official files + CI conversions)
    subprocess.run(["gh", "release", "download", tag, "--dir", str(dest), "--clobber"], check=False)
    for url, name, algo, want in OFFICIAL:
        p = dest / name
        if not p.exists() or digest(p, algo) != want:
            with s.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        got = digest(p, algo)
        if got != want:
            sys.exit(f"CHECKSUM MISMATCH {name}: {got} != {want}")
        print(f"ok {name}")
    subprocess.run(["gh", "release", "download", "models-asr-v1", "--dir", str(dest), "--clobber"], check=False)
    asr_sums = dest / "SHA256SUMS_ASR.txt"
    if asr_sums.exists():
        for line in asr_sums.read_text().splitlines():
            if line.strip():
                h, n = line.split()
                if digest(dest / n.lstrip("*"), "sha256") != h:
                    sys.exit(f"CHECKSUM MISMATCH {n}")
        print("ok speech-to-text pack (models-asr-v1)")
    sums = (dest / "SHA256SUMS.txt").read_text().split("\n")
    for line in sums:
        if line.strip():
            h, n = line.split()
            n = n.lstrip("*")
            if digest(dest / n, "sha256") != h:
                sys.exit(f"CHECKSUM MISMATCH {n}")
            print(f"ok {n} (models release {tag})")


if __name__ == "__main__":
    main()
