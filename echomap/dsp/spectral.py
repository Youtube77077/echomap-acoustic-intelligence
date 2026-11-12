"""Spectral analysis: STFT, mel / log-frequency spectrograms, MFCC and frame features (pure numpy)."""
from __future__ import annotations

import numpy as np

DB_FLOOR = -120.0


def hann(n: int) -> np.ndarray:
    k = np.arange(n)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float32)


def frame_signal(x: np.ndarray, n: int, hop: int, center: bool = True) -> np.ndarray:
    if center:
        x = np.pad(x, (n // 2, n // 2), mode="constant")
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nf = 1 + (len(x) - n) // hop
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop][:nf]


def stft_power(x: np.ndarray, n_fft: int, hop: int, center: bool = True) -> np.ndarray:
    """Power spectrogram (n_fft//2+1, frames). Window-energy normalised so a full-scale sine is ~0 dBFS."""
    w = hann(n_fft)
    frames = frame_signal(x.astype(np.float32), n_fft, hop, center)
    out = np.empty((n_fft // 2 + 1, frames.shape[0]), np.float32)
    norm = (w.sum() / 2.0) ** 2
    step = 2048
    for i in range(0, frames.shape[0], step):
        s = np.fft.rfft(frames[i:i + step] * w, axis=-1)
        out[:, i:i + step] = ((s.real ** 2 + s.imag ** 2) / norm).T
    return out


def power_to_db(p: np.ndarray) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(p, 1e-14)), DB_FLOOR).astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular HTK mel filterbank (n_mels, n_fft//2+1), area-normalised."""
    fmax = fmax or sr / 2
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        dn = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, dn))
        fb[i] *= 2.0 / max(hi - lo, 1e-9)
    return fb


def dct_ortho(n_in: int, n_out: int) -> np.ndarray:
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    m = np.cos(np.pi / n_in * (n + 0.5) * k) * np.sqrt(2.0 / n_in)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


def frame_features(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int | None = None) -> dict:
    """Per-frame descriptors. Returns arrays of shape (frames,) plus mfcc (13, frames)."""
    hop = hop or n_fft // 4
    p = stft_power(x, n_fft, hop)
    freqs = np.linspace(0, sr / 2, p.shape[0]).astype(np.float32)
    mag = np.sqrt(p)
    tot = mag.sum(axis=0) + 1e-12
    centroid = (freqs[:, None] * mag).sum(axis=0) / tot
    bandwidth = np.sqrt((((freqs[:, None] - centroid[None]) ** 2) * mag).sum(axis=0) / tot)
    cum = np.cumsum(p, axis=0)
    rolloff = freqs[np.minimum((cum < 0.85 * cum[-1:]).sum(axis=0), len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(p + 1e-14), axis=0)) / (np.mean(p, axis=0) + 1e-14)
    frames = frame_signal(x, n_fft, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-14)
    peak = np.max(np.abs(frames), axis=1)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    fb = mel_filterbank(sr, n_fft, 40, 20.0, min(sr / 2, 16000))
    logmel = np.log(fb @ p + 1e-10)
    mfcc = dct_ortho(40, 13) @ logmel
    n = min(len(rms), p.shape[1])
    return {"hop_s": hop / sr, "rms": rms[:n], "peak": peak[:n], "zcr": zcr[:n], "centroid": centroid[:n],
            "bandwidth": bandwidth[:n], "rolloff": rolloff[:n], "flatness": flatness[:n], "mfcc": mfcc[:, :n]}


def summarize(feats: dict) -> dict:
    def st(a):
        a = np.asarray(a, np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "p10": float(np.percentile(a, 10)),
                "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}
    rms_db = 20 * np.log10(feats["rms"] + 1e-12)
    return {
        "rms_dbfs": st(rms_db),
        "peak_dbfs": float(20 * np.log10(feats["peak"].max() + 1e-12)),
        "zero_crossing_rate": st(feats["zcr"]),
        "spectral_centroid_hz": st(feats["centroid"]),
        "spectral_bandwidth_hz": st(feats["bandwidth"]),
        "spectral_rolloff85_hz": st(feats["rolloff"]),
        "spectral_flatness": st(feats["flatness"]),
        "mfcc_mean": [float(v) for v in feats["mfcc"].mean(axis=1)],
        "mfcc_std": [float(v) for v in feats["mfcc"].std(axis=1)],
    }


def ltas(x: np.ndarray, sr: int, n_fft: int = 8192) -> dict:
    """Long-term average spectrum in dBFS on a log-spaced frequency axis (for plotting)."""
    p = stft_power(x, n_fft, n_fft // 2)
    m = p.mean(axis=1)
    freqs = np.linspace(0, sr / 2, len(m))
    fq = np.geomspace(20, sr / 2, 200)
    idx = np.clip(np.searchsorted(freqs, fq), 1, len(m) - 1)
    edges = np.concatenate([[1], (idx[1:] + idx[:-1]) // 2, [len(m)]])
    vals = [float(10 * np.log10(m[edges[i]:max(edges[i + 1], edges[i] + 1)].mean() + 1e-14)) for i in range(len(fq))]
    return {"freq_hz": [round(float(f), 1) for f in fq], "dbfs": [round(v, 2) for v in vals]}


def band_energy_db(p: np.ndarray, sr: int, f_lo: float, f_hi: float) -> np.ndarray:
    n = p.shape[0]
    a = int(f_lo / (sr / 2) * (n - 1))
    b = max(a + 1, int(f_hi / (sr / 2) * (n - 1)))
    return power_to_db(p[a:b].sum(axis=0))


def spectrogram_view(x: np.ndarray, sr: int, t0: float, t1: float, fmin: float, fmax: float,
                     scale: str = "linear", width: int = 1200, height: int = 360,
                     db_range: tuple[float, float] = (-110.0, -10.0)) -> dict:
    """Render a region of the spectrogram to a uint8 matrix (height x width), row 0 = highest frequency."""
    t0 = max(0.0, t0)
    t1 = min(len(x) / sr, max(t1, t0 + 0.01))
    fmax = min(fmax, sr / 2)
    fmin = max(0.0 if scale == "linear" else 10.0, min(fmin, fmax - 1))
    width = int(np.clip(width, 64, 4096))
    height = int(np.clip(height, 64, 1024))
    dur = t1 - t0
    hop = max(1, int(dur * sr / width))
    # frequency resolution: at least the vertical pixel spacing for the lowest displayed band
    need_df = (fmax - fmin) / height if scale == "linear" else max(fmin * 0.06, 2.0)
    n_fft = int(2 ** np.ceil(np.log2(max(sr / max(need_df, 1e-3), 256))))
    n_fft = int(np.clip(n_fft, 256, 16384))
    n_fft = max(n_fft, 2 ** int(np.ceil(np.log2(hop * 2))))
    a = int(t0 * sr)
    b = int(t1 * sr)
    pad = n_fft // 2
    seg = x[max(0, a - pad):min(len(x), b + pad)]
    lead = a - max(0, a - pad)
    seg = np.pad(seg, (pad - lead, 0))
    p = stft_power(seg, n_fft, hop, center=False)
    p = p[:, :width] if p.shape[1] >= width else np.pad(p, ((0, 0), (0, width - p.shape[1])))
    freqs = np.linspace(0, sr / 2, p.shape[0])
    if scale == "log":
        edges = np.geomspace(fmin, fmax, height + 1)
    elif scale == "mel":
        edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), height + 1))
    else:
        edges = np.linspace(fmin, fmax, height + 1)
    lo = np.searchsorted(freqs, edges[:-1])
    hi = np.maximum(np.searchsorted(freqs, edges[1:]), lo + 1)
    lo = np.clip(lo, 0, p.shape[0] - 1)
    hi = np.clip(hi, 1, p.shape[0])
    cs = np.vstack([np.zeros((1, p.shape[1]), np.float64), np.cumsum(p, axis=0, dtype=np.float64)])
    rows = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    # rows needing finer resolution than FFT bins: interpolate
    db = power_to_db(rows.astype(np.float32))[::-1]
    lo_db, hi_db = db_range
    u8 = np.clip((db - lo_db) / (hi_db - lo_db) * 255.0, 0, 255).astype(np.uint8)
    return {"t0": t0, "t1": t1, "fmin": float(fmin), "fmax": float(fmax), "scale": scale, "width": width,
            "height": height, "db_min": lo_db, "db_max": hi_db, "n_fft": n_fft, "hop": hop,
            "freq_resolution_hz": sr / n_fft, "time_resolution_s": hop / sr, "data": u8}


def waveform_envelope(x: np.ndarray, buckets: int = 4000) -> dict:
    n = len(x)
    buckets = int(min(buckets, n))
    edges = np.linspace(0, n, buckets + 1).astype(int)
    mins = np.minimum.reduceat(x, edges[:-1])
    maxs = np.maximum.reduceat(x, edges[:-1])
    return {"min": [round(float(v), 4) for v in mins], "max": [round(float(v), 4) for v in maxs]}
