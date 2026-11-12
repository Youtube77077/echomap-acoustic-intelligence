"""Video files (MP4 with AAC audio): probe, stream-copy extraction and decoding (v1.0.8 regression: extraction crashed)."""
from pathlib import Path

import av
import numpy as np

from echomap.ingest import media


def make_mp4(path: Path, seconds: float = 3.0, sr: int = 16000):
    out = av.open(str(path), "w")
    v = out.add_stream("mpeg4", rate=10)
    v.width, v.height, v.pix_fmt = 160, 120, "yuv420p"
    a = out.add_stream("aac", rate=sr)
    a.layout = "mono"
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr)).astype(np.float32)
    for _ in range(int(seconds * 10)):
        for p in v.encode(av.VideoFrame.from_ndarray(np.zeros((120, 160, 3), np.uint8), format="rgb24")):
            out.mux(p)
    for s in range(0, len(x), 1024):
        f = av.AudioFrame.from_ndarray(x[s:s + 1024][None], format="flt", layout="mono")
        f.sample_rate = sr
        for p in a.encode(f):
            out.mux(p)
    for st in (v, a):
        for p in st.encode():
            out.mux(p)
    out.close()


def test_mp4_probe_extract_decode(tmp_path):
    src = tmp_path / "clip.mp4"
    make_mp4(src)
    m = media.probe(src)
    assert m["is_video"] and m["primary_audio"]["codec"] == "aac"
    ex = media.extract_audio_stream(src, tmp_path)
    assert ex["codec"] == "aac" and Path(ex["path"]).stat().st_size > 1000 and "stream index" in ex["method"]
    xs, sr = media.decode_channels(src, 16000, max_channels=2)
    assert sr == 16000 and xs.shape[1] > 2.5 * 16000
    spec = np.abs(np.fft.rfft(xs[0][:16000]))
    assert abs(np.argmax(spec) - 440) <= 3   # the 440 Hz tone survives decoding
