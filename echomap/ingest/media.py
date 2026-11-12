"""Media ingest: integrity hashing, metadata extraction, lossless audio extraction, decoding.

The original file is only ever opened read-only.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np

from ..config import MAX_DURATION_S, SUPPORTED_EXT, VIDEO_EXT


class IngestError(Exception):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
_DEVICE_KEYS = ("make", "model", "com.apple.quicktime.make", "com.apple.quicktime.model",
                "com.android.manufacturer", "com.android.model", "encoder", "handler_name",
                "com.apple.quicktime.software", "software", "originator", "comment")
_TIME_KEYS = ("creation_time", "date", "com.apple.quicktime.creationdate", "creationdate",
              "origination_date", "origination_time", "year", "ICRD", "date_recorded")
_LOC_KEYS = ("location", "com.apple.quicktime.location.ISO6709", "location-eng")


def probe(path: str | Path) -> dict:
    """Read technical metadata. Returns a JSON-serialisable dict."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise IngestError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXT))}")
    st = path.stat()
    try:
        c = av.open(str(path), mode="r")
    except av.FFmpegError as e:  # pragma: no cover - depends on file
        raise IngestError(f"Could not open media: {e}") from e
    with c:
        fmt = c.format
        duration = float(c.duration / av.time_base) if c.duration else None
        tags = {str(k): str(v) for k, v in (c.metadata or {}).items()}
        streams = []
        audio_streams = []
        for s in c.streams:
            d = {"index": s.index, "type": s.type, "codec": s.codec_context.name if s.codec_context else None,
                 "codec_long": getattr(s.codec_context.codec, "long_name", None) if s.codec_context else None,
                 "bit_rate": s.bit_rate or (s.codec_context.bit_rate if s.codec_context else None),
                 "duration_s": float(s.duration * s.time_base) if s.duration and s.time_base else None,
                 "tags": {str(k): str(v) for k, v in (s.metadata or {}).items()}}
            if s.type == "audio":
                cc = s.codec_context
                d.update({"sample_rate": cc.sample_rate, "channels": cc.channels,
                          "layout": cc.layout.name if cc.layout else None,
                          "sample_format": cc.format.name if cc.format else None,
                          "bits_per_sample": (cc.format.bytes * 8) if cc.format else None,
                          "bits_per_raw_sample": getattr(cc, "bits_per_raw_sample", None) or None})
                audio_streams.append(d)
            elif s.type == "video":
                cc = s.codec_context
                d.update({"width": cc.width, "height": cc.height,
                          "frame_rate": float(s.average_rate) if s.average_rate else None})
            streams.append(d)
        if not audio_streams:
            raise IngestError("The file contains no audio stream.")
        all_tags = dict(tags)
        for s in streams:
            for k, v in s["tags"].items():
                all_tags.setdefault(f"stream{s['index']}:{k}", v)
        lower = {k.lower().split(":")[-1]: v for k, v in all_tags.items()}
        embedded_times = {k: lower[k.lower()] for k in _TIME_KEYS if k.lower() in lower}
        device = {k: lower[k.lower()] for k in _DEVICE_KEYS if k.lower() in lower}
        location = None
        for k in _LOC_KEYS:
            if k.lower() in lower:
                m = _ISO6709.match(lower[k.lower()])
                if m:
                    location = {"raw": lower[k.lower()], "lat": float(m.group(1)), "lon": float(m.group(2)),
                                "note": "Embedded metadata claim; editable, not verified acoustically."}
        return {
            "file_name": path.name,
            "file_size_bytes": st.st_size,
            "extension": ext,
            "is_video": ext in VIDEO_EXT or any(s["type"] == "video" for s in streams),
            "container": fmt.name,
            "container_long": fmt.long_name,
            "duration_s": duration,
            "bit_rate": c.bit_rate,
            "container_tags": tags,
            "streams": streams,
            "primary_audio": audio_streams[0],
            "embedded_times": embedded_times,
            "device": device,
            "embedded_location": location,
            "filesystem_times": {
                "modified_utc": _iso(st.st_mtime),
                "created_utc": _iso(getattr(st, "st_birthtime", st.st_ctime)),
                "accessed_utc": _iso(st.st_atime),
                "note": "File-system times describe this copy of the file, not when the sound was recorded.",
            },
        }


def extract_audio_stream(src: str | Path, dst_dir: str | Path) -> dict:
    """Stream-copy the primary audio track into Matroska audio (.mka), with no re-encoding."""
    src, dst_dir = Path(src), Path(dst_dir)
    dst = dst_dir / (src.stem + "_audio_streamcopy.mka")
    with av.open(str(src), "r") as inp:
        ast = next(s for s in inp.streams if s.type == "audio")
        index, codec = ast.index, ast.codec_context.name   # read while open (PyAV 19 refuses after close)
        with av.open(str(dst), "w", format="matroska") as out:
            ost = out.add_stream_from_template(ast)
            for pkt in inp.demux(ast):
                if pkt.dts is None:
                    continue
                pkt.stream = ost
                out.mux(pkt)
    return {"path": str(dst), "method": "stream copy (no re-encode), audio stream index %d" % index,
            "codec": codec, "container": "matroska", "sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size}


def decode_mono(path: str | Path, target_sr: int | None = None, max_seconds: float = MAX_DURATION_S + 1):
    """Decode the primary audio stream to float32.

    Returns (mono, sr, channel_rms_dbfs). Multichannel audio is averaged to mono, and each channel's
    level is reported. Resampling (if target_sr is set) uses libswresample through PyAV.
    """
    with av.open(str(path), "r") as c:
        ast = next(s for s in c.streams if s.type == "audio")
        sr_in = ast.codec_context.sample_rate
        sr = target_sr or sr_in
        resampler = av.AudioResampler(format="fltp", layout=None, rate=sr)
        chunks: list[np.ndarray] = []
        total = 0
        limit = int(max_seconds * sr)
        for frame in c.decode(ast):
            for rf in resampler.resample(frame):
                a = rf.to_ndarray()  # (channels, n) planar float
                chunks.append(a.astype(np.float32, copy=False))
                total += a.shape[1]
            if total > limit:
                break
        for rf in resampler.resample(None):
            a = rf.to_ndarray()
            chunks.append(a.astype(np.float32, copy=False))
    if not chunks:
        raise IngestError("No audio samples could be decoded.")
    nch = max(ch.shape[0] for ch in chunks)
    data = np.concatenate([np.pad(ch, ((0, nch - ch.shape[0]), (0, 0))) for ch in chunks], axis=1)
    ch_rms = [float(20 * np.log10(np.sqrt(np.mean(data[i] ** 2)) + 1e-12)) for i in range(nch)]
    mono = data.mean(axis=0).astype(np.float32)
    return mono, sr, ch_rms


def decode_channels(path: str | Path, target_sr: int | None = None, max_channels: int = 2):
    """Decode up to `max_channels` channels as float32 (channels, n). Used for stereo direction estimates."""
    with av.open(str(path), "r") as c:
        ast = next(s for s in c.streams if s.type == "audio")
        sr = target_sr or ast.codec_context.sample_rate
        resampler = av.AudioResampler(format="fltp", layout=None, rate=sr)
        chunks = []
        for frame in c.decode(ast):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().astype(np.float32, copy=False))
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray().astype(np.float32, copy=False))
    if not chunks:
        return np.zeros((1, 0), np.float32), sr
    nch = min(max_channels, max(ch.shape[0] for ch in chunks))
    data = np.concatenate([np.pad(ch[:nch], ((0, nch - min(nch, ch.shape[0])), (0, 0))) for ch in chunks], axis=1)
    return data, sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resampling via libswresample (PyAV)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    frame = av.AudioFrame.from_ndarray(x[None, :].astype(np.float32), format="fltp", layout="mono")
    frame.sample_rate = sr_in
    rs = av.AudioResampler(format="fltp", layout="mono", rate=sr_out)
    out = [f.to_ndarray()[0] for f in rs.resample(frame)] + [f.to_ndarray()[0] for f in rs.resample(None)]
    y = np.concatenate(out) if out else np.zeros(0, np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def write_wav16(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Write mono 16-bit PCM WAV (standard library)."""
    import wave
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def check_duration(meta: dict, decoded_seconds: float) -> None:
    d = meta.get("duration_s") or decoded_seconds
    if d > MAX_DURATION_S + 0.5:
        raise IngestError(f"Recording is {d:.1f} s long. EchoMap supports recordings up to {MAX_DURATION_S/60:.0f} minutes.")
    if decoded_seconds < 0.5:
        raise IngestError("Recording is shorter than 0.5 s, which is too short to analyse.")


def is_lossless(meta: dict) -> bool:
    return (meta["primary_audio"].get("codec") or "").startswith(("pcm_", "flac", "alac", "wavpack"))


def safe_case_name(path: str | Path, digest: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem)[:40]
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{stem}_{digest[:8]}"


__all__ = ["probe", "decode_channels", "sha256_file", "extract_audio_stream", "decode_mono", "resample", "write_wav16",
           "check_duration", "IngestError", "safe_case_name", "is_lossless", "os"]
