"""Transients, periodicity, energy segmentation and SNR estimation."""
from __future__ import annotations

import numpy as np

from .spectral import mel_filterbank, power_to_db, stft_power


def onset_envelope(x: np.ndarray, sr: int, fps: int = 100) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength on a log-mel spectrogram. Returns (env, frame_period_s)."""
    hop = sr // fps
    n_fft = 1024 if sr <= 24000 else 2048
    p = stft_power(x, n_fft, hop)
    fb = mel_filterbank(sr, n_fft, 64, 30.0, min(sr / 2, 16000))
    lm = np.log1p(1000.0 * (fb @ p))
    flux = np.maximum(0.0, np.diff(lm, axis=1)).mean(axis=0)
    env = np.concatenate([[0.0], flux]).astype(np.float32)
    return env, hop / sr


def pick_peaks(env: np.ndarray, period: float, k: float = 3.0, min_gap_s: float = 0.05, win_s: float = 1.0):
    """Adaptive threshold (moving median + k * MAD) peak picking. Returns (indices, strengths)."""
    n = len(env)
    w = max(3, int(win_s / period) | 1)
    padded = np.pad(env, (w // 2, w // 2), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, w)
    med = np.median(view, axis=1)[:n]
    mad = np.median(np.abs(view - med[:, None]), axis=1)[:n] + 1e-6
    thr = med + k * mad * 1.4826
    cand = np.where((env > thr) & (env >= np.roll(env, 1)) & (env >= np.roll(env, -1)))[0]
    gap = max(1, int(min_gap_s / period))
    keep = []
    for i in cand[np.argsort(-env[cand])]:
        if all(abs(i - j) >= gap for j in keep):
            keep.append(i)
    keep = np.array(sorted(keep), int)
    return keep, (env[keep] - thr[keep]) if len(keep) else np.zeros(0)


def onsets(x: np.ndarray, sr: int) -> dict:
    env, per = onset_envelope(x, sr)
    idx, strength = pick_peaks(env, per)
    return {"times_s": [round(float(i * per), 3) for i in idx], "strength": [round(float(s), 4) for s in strength],
            "frame_period_s": per, "_env": env}


def periodicity(env: np.ndarray, period: float, min_lag_s: float = 0.1, max_lag_s: float = 5.0, top: int = 5):
    """Autocorrelation of the onset envelope. Returns the strongest repetition periods."""
    e = env - env.mean()
    n = len(e)
    if n < 10:
        return []
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0] + 1e-12
    lo, hi = int(min_lag_s / period), min(n - 1, int(max_lag_s / period))
    seg = ac[lo:hi]
    out = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.15:
            out.append((float(seg[i]), (i + lo) * period))
    out.sort(reverse=True)
    return [{"period_s": round(p, 3), "rate_hz": round(1 / p, 3), "autocorr": round(a, 3)} for a, p in out[:top]]


def snr_estimate(x: np.ndarray, sr: int, frame_s: float = 0.05) -> dict:
    n = max(1, int(frame_s * sr))
    m = len(x) // n
    if m < 4:
        return {"snr_db": None, "note": "Recording too short for SNR estimate."}
    fr = x[:m * n].reshape(m, n)
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-14)
    noise = float(np.percentile(db, 10))
    sig = float(np.percentile(db, 95))
    return {"snr_db": round(sig - noise, 1), "noise_floor_dbfs": round(noise, 1), "signal_level_dbfs": round(sig, 1),
            "method": "Percentile estimate: 95th percentile minus 10th percentile of 50 ms frame energy. "
                      "This is an estimate: steady background sounds count as 'noise' and continuous signals reduce it."}


def energy_segments(x: np.ndarray, sr: int, above_db: float = 8.0, min_dur: float = 0.12, merge_gap: float = 0.25):
    """Segments whose frame energy exceeds the noise floor by `above_db`."""
    hop = int(0.02 * sr)
    p = stft_power(x, 1024 if sr <= 24000 else 2048, hop)
    db = power_to_db(p.sum(axis=0))
    floor = np.percentile(db, 15)
    on = db > floor + above_db
    segs = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * hop / sr <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    out = []
    for a, b in merged:
        t0, t1 = a * hop / sr, b * hop / sr
        if t1 - t0 >= min_dur:
            out.append({"start": round(t0, 3), "end": round(t1, 3),
                        "peak_db_above_floor": round(float(db[a:b].max() - floor), 1)})
    return out, float(floor)


def band_profile(x: np.ndarray, sr: int, t0: float, t1: float, f_lo: float = 50.0, f_hi: float | None = None) -> dict:
    """Frequency characteristics of a segment: peak frequency, centroid, and the 5-95 % energy band."""
    a, b = int(max(0, t0) * sr), int(min(len(x), t1 * sr))
    seg = x[a:b]
    if len(seg) < 256:
        return {}
    n_fft = 4096 if len(seg) >= 4096 else int(2 ** np.floor(np.log2(len(seg))))
    p = stft_power(seg, n_fft, n_fft // 4).mean(axis=1)
    freqs = np.linspace(0, sr / 2, len(p))
    f_hi = f_hi or sr / 2
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not m.any():
        return {}
    pf, ff = p[m], freqs[m]
    c = np.cumsum(pf) / (pf.sum() + 1e-14)
    return {"peak_hz": round(float(ff[np.argmax(pf)]), 1),
            "centroid_hz": round(float((ff * pf).sum() / (pf.sum() + 1e-14)), 1),
            "band_5_95_hz": [round(float(ff[np.searchsorted(c, 0.05)]), 1),
                             round(float(ff[min(len(ff) - 1, np.searchsorted(c, 0.95))]), 1)],
            "level_dbfs": round(float(10 * np.log10(np.mean(seg ** 2) + 1e-14)), 1)}
