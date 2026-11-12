"""Direction estimates from stereo recordings (GCC-PHAT time difference plus level difference).

Physics and limits:
- Mono recordings carry no direction information, so nothing is estimated.
- With two channels, a time difference gives only a *lateral* angle (left/right of the microphone
  axis) with front/back ambiguity, and it needs the microphone spacing. The spacing is usually
  unknown, so EchoMap reports the angle for a plausible range of spacings (cone), not a point.
- Coincident (XY/MS) pairs carry direction mostly in level differences. Phone stereo is often synthetic.
- Elevation and distance cannot be determined from a stereo pair. Distance is never estimated here.
"""
from __future__ import annotations

import numpy as np

C_SOUND = 343.0


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_tau: float):
    n = len(a) + len(b)
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    m = int(max_tau * sr)
    cc = np.concatenate((cc[-m:], cc[:m + 1]))
    k = int(np.argmax(np.abs(cc)))
    peak = float(np.abs(cc[k]))
    sharp = peak / (np.median(np.abs(cc)) + 1e-12)
    return (k - m) / sr, peak, sharp


def channel_report(xs: np.ndarray, sr: int) -> dict:
    if xs.ndim != 2 or xs.shape[0] < 2:
        return {"channels": 1, "supported": False,
                "reason": "Mono recording: no inter-channel cues, so source direction cannot be estimated."}
    a, b = xs[0], xs[1]
    diff = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-12))
    corr = float(np.corrcoef(a[:: max(1, len(a) // 200000)], b[:: max(1, len(b) // 200000)])[0, 1]) if len(a) > 10 else 1.0
    if diff < 1e-3:
        return {"channels": 2, "supported": False, "inter_channel_correlation": round(corr, 4),
                "reason": "Both channels are identical (dual-mono): no direction information."}
    return {"channels": 2, "supported": True, "inter_channel_correlation": round(corr, 4),
            "relative_channel_difference": round(diff, 4)}


def event_direction(xs: np.ndarray, sr: int, t0: float, t1: float, spacing_range_m=(0.1, 0.3)) -> dict:
    a = xs[0, int(t0 * sr):int(t1 * sr)]
    b = xs[1, int(t0 * sr):int(t1 * sr)]
    if len(a) < int(0.1 * sr):
        return {"supported": False, "reason": "event shorter than 0.1 s"}
    max_tau = spacing_range_m[1] / C_SOUND * 1.2
    tau, peak, sharp = gcc_phat(a, b, sr, max_tau)
    ild = float(20 * np.log10((np.sqrt(np.mean(a ** 2)) + 1e-12) / (np.sqrt(np.mean(b ** 2)) + 1e-12)))
    res = {"supported": True, "itd_ms": round(tau * 1000, 3), "ild_db": round(ild, 2), "gcc_peak_sharpness": round(sharp, 1)}
    if sharp < 4:
        res["lateral"] = "undetermined"
        res["reason"] = "No clear inter-channel delay (diffuse, reverberant or coincident-microphone recording)."
    else:
        angles = []
        for d in spacing_range_m:
            s = np.clip(tau * C_SOUND / d, -1, 1)
            angles.append(float(np.degrees(np.arcsin(s))))
        lo, hi = sorted(angles)
        side = "left" if tau > 0 else "right" if tau < 0 else "centre"
        res.update({"lateral": side, "lateral_angle_deg_range": [round(lo, 1), round(hi, 1)],
                    "assumption": f"Spaced pair with unknown spacing {spacing_range_m[0]}-{spacing_range_m[1]} m; "
                                  "channel 1 = left. Front/back ambiguous; elevation and distance not determinable."})
    if abs(ild) > 3:
        res["level_cue"] = f"{'left' if ild > 0 else 'right'} channel louder by {abs(ild):.1f} dB"
    return res
