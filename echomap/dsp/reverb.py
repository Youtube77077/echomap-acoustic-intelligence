"""Blind reverberation and wind-noise analysis.

Reverberation: decay slopes after impulsive events, a Schroeder-style backward-integrated
envelope with a linear fit between -5 and -25 dB, extrapolated to RT60. This is a coarse
indicator, not an ISO 3382 measurement: it needs clean impulsive sounds with ≥ 20 dB of dynamic
range, and background noise biases it.
"""
from __future__ import annotations

import numpy as np

from .spectral import band_energy_db, stft_power


def rt60_blind(x: np.ndarray, sr: int, onset_times: list[float], max_events: int = 40) -> dict:
    hop = int(0.005 * sr)
    p = stft_power(x, 1024 if sr <= 24000 else 2048, hop)
    env_db = band_energy_db(p, sr, 500, min(4000, sr / 2 - 1))
    floor = np.percentile(env_db, 10)
    est = []
    for t in onset_times[:400]:
        i = int(t * sr / hop)
        j = min(len(env_db), i + int(0.08 * sr / hop))
        if j - i < 2:
            continue
        pk = i + int(np.argmax(env_db[i:j]))
        if env_db[pk] - floor < 22:
            continue
        pre = env_db[max(0, i - int(0.06 * sr / hop)):max(1, i - int(0.01 * sr / hop))]
        if len(pre) == 0 or np.mean(pre) > env_db[pk] - 15:
            continue  # not an impulsive onset out of relative quiet: a decay cannot be observed
        tail = env_db[pk:pk + int(3.0 * sr / hop)]
        lin = 10 ** (tail / 10)
        sch = np.cumsum(lin[::-1])[::-1]
        sch_db = 10 * np.log10(sch / sch[0] + 1e-14)
        stop = np.where(tail < floor + 5)[0]
        stop = stop[0] if len(stop) else len(tail)
        sch_db = sch_db[:stop]
        a = np.where(sch_db <= -5)[0]
        b = np.where(sch_db <= -25)[0]
        if not len(a) or not len(b) or b[0] - a[0] < 4:
            continue
        seg = sch_db[a[0]:b[0]]
        tt = np.arange(len(seg)) * hop / sr
        slope = np.polyfit(tt, seg, 1)[0]
        if slope >= -5:
            continue
        r = float(-60.0 / slope)
        if 0.05 <= r <= 8.0:
            est.append(r)
        if len(est) >= max_events:
            break
    if len(est) < 5:
        return {"rt60_s": None, "n_decays": len(est),
                "interpretation": "Not estimable: too few clean impulsive decays with sufficient dynamic range."}
    med = float(np.median(est))
    if med < 0.3:
        interp = "Short decay: consistent with outdoors, a small furnished room, or a close source"
    elif med < 0.8:
        interp = "Moderate decay: consistent with a typical room or a semi-enclosed space"
    elif med < 1.8:
        interp = "Long decay: consistent with a large room, hall, church, station or tunnel"
    else:
        interp = "Very long decay: consistent with a very large reverberant space (church, cavern, underground station)"
    return {"rt60_s": round(med, 2), "iqr_s": [round(float(np.percentile(est, 25)), 2), round(float(np.percentile(est, 75)), 2)],
            "n_decays": len(est), "interpretation": interp,
            "method": "Blind estimate from free decays after transients (500-4000 Hz band, Schroeder integration, -5 to -25 dB fit)."}


def wind_analysis(x: np.ndarray, sr: int) -> dict:
    """Microphone wind noise: intermittent, strong energy below ~150 Hz relative to the mid band."""
    hop = int(0.05 * sr)
    p = stft_power(x, 2048 if sr >= 16000 else 1024, hop)
    lf = band_energy_db(p, sr, 20, 150)
    mf = band_energy_db(p, sr, 500, 4000)
    ratio = lf - mf
    windy = ratio > 10
    k = 10
    sm = np.convolve(lf, np.ones(k) / k, mode="same")
    base = np.percentile(sm, 30)
    gusts = []
    i = 0
    while i < len(sm):
        if sm[i] > base + 6:
            j = i
            while j < len(sm) and sm[j] > base + 3:
                j += 1
            if (j - i) * hop / sr >= 0.5:
                gusts.append({"start": round(i * hop / sr, 2), "end": round(j * hop / sr, 2),
                              "peak_db_above_baseline": round(float(sm[i:j].max() - base), 1)})
            i = j
        else:
            i += 1
    return {"fraction_frames_lf_dominant": round(float(windy.mean()), 3), "gusts": gusts,
            "lf_envelope": {"hop_s": hop / sr, "db": [round(float(v), 1) for v in sm[::4]]},
            "note": ("Low-frequency dominance and gust events are measured as relative levels (dB). Wind speed "
                     "(m/s), direction and temperature cannot be inferred from ordinary audio. Handling noise and "
                     "traffic rumble can also produce low-frequency energy.")}
