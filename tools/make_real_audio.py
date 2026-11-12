"""Build real-audio test files for the end-to-end CI test (never committed). Usage: python tools/make_real_audio.py <dest>

- dutch_speech.wav   : ~40 s of Dutch read speech (FLEURS nl_nl test split, CC BY 4.0), 16 kHz
- street_dutch.wav   : 60 s of ESC-50 engine / car-horn / train / footsteps clips with the Dutch speech mixed in at -12 dB
- long_mix.wav       : 300 s (the 5-minute maximum) of looped ESC-50 clips incl. speech, for timing
"""
import io
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.ingest.media import decode_mono, resample  # noqa: E402

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/nl_nl/audio/test.tar.gz"
ESC = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{}.wav"
# engine, car horn, train, footsteps, church bells, siren, chirping birds, rain
ESC_CLIPS = ["1-18527-A-44", "1-17124-A-43", "1-119125-A-45", "1-155858-A-25", "1-13571-A-46", "1-31482-A-42", "1-100038-A-14", "1-17367-A-10"]
SR = 16000


def write(p: Path, x: np.ndarray, sr: int = SR):
    x = np.clip(x, -1, 1)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def fleurs(n_seconds: float, tmp: Path) -> np.ndarray:
    r = requests.get(FLEURS, stream=True, timeout=300, headers={"User-Agent": "EchoMap-CI/1.0"})
    r.raise_for_status()
    out, tot = [], 0.0
    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".wav"):
                (tmp / "u.wav").write_bytes(tf.extractfile(m).read())
                x, sr, _ = decode_mono(tmp / "u.wav", SR)
                out += [x, np.zeros(int(0.6 * SR), np.float32)]
                tot += len(x) / SR + 0.6
                if tot >= n_seconds:
                    break
    return np.concatenate(out)


def esc(tmp: Path) -> list[np.ndarray]:
    xs = []
    for c in ESC_CLIPS:
        try:
            r = requests.get(ESC.format(c), timeout=120)
            r.raise_for_status()
        except requests.RequestException as e:
            print("skip", c, e)
            continue
        (tmp / "e.wav").write_bytes(r.content)
        x, sr, _ = decode_mono(tmp / "e.wav", SR)
        xs.append(x)
    return xs


def main():
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_tmp"
    tmp.mkdir(exist_ok=True)
    nl = fleurs(40, tmp)
    nl = nl / (np.abs(nl).max() + 1e-9) * 0.7
    write(dest / "dutch_speech.wav", nl)
    clips = esc(tmp)
    bed = np.concatenate(clips[:4] * 3)[: 60 * SR]
    sp = np.zeros_like(bed)
    sp[5 * SR:5 * SR + min(len(nl), len(bed) - 5 * SR)] = nl[: len(bed) - 5 * SR]
    rms = lambda v: np.sqrt(np.mean(v[np.abs(v) > 1e-4] ** 2)) + 1e-9  # noqa: E731
    street = bed / rms(bed) * 0.08 + sp / rms(sp) * 0.08 * 10 ** (-12 / 20) * 4
    write(dest / "street_dutch.wav", street / (np.abs(street).max() + 1e-9) * 0.8)
    loop = np.concatenate(clips + [nl])
    long = np.tile(loop, int(np.ceil(300 * SR / len(loop))))[: 300 * SR]
    write(dest / "long_mix.wav", long / (np.abs(long).max() + 1e-9) * 0.8)
    for f in sorted(dest.glob("*.wav")):
        x, sr, _ = decode_mono(f, SR)
        print(f"{f.name}: {len(x) / SR:.1f} s")


if __name__ == "__main__":
    main()
