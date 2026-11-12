"""Bell strike analysis: nominal-partial estimation per strike and ring reconstruction.

Church bells are not harmonic. A well-tuned bell's main partials relative to the *nominal* are:
hum 0.25, prime 0.5, tierce 0.5946 (minor third above the prime), quint 0.749, nominal 1.0,
superquint ~1.5, octave nominal 2.0. The perceived strike note is close to an octave below the
nominal, and Dove's Guide lists bells by nominal frequency. EchoMap therefore fits this partial template
to each strike instead of relying on one frequency, and only accepts a strike when the nominal plus
at least two other partials are present.
"""
from __future__ import annotations

import numpy as np

TEMPLATE = [(0.25, 0.5, "hum"), (0.5, 0.6, "prime"), (0.5946, 0.6, "tierce"), (0.749, 0.2, "quint"),
            (1.0, 1.0, "nominal"), (1.5, 0.3, "superquint"), (2.0, 0.3, "octave nominal")]
NOTE_NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]


def note_name(f: float) -> str:
    n = int(round(12 * np.log2(f / 440.0) + 69))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def _peaks_db(x: np.ndarray, sr: int, t: float, dur: float = 0.45):
    a = int((t + 0.02) * sr)
    seg = x[a:a + int(dur * sr)]
    if len(seg) < int(0.15 * sr):
        return None
    n = 1 << int(np.ceil(np.log2(max(len(seg), 8192) * 2)))
    w = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * w, n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    keep = freqs < 9000.0
    freqs, spec = freqs[keep], spec[keep]
    db = 20 * np.log10(spec + 1e-12)
    k = 41
    pad = np.pad(db, (k // 2, k // 2), mode="edge")
    bg = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)[:len(db)]
    return freqs, db - bg


def strike_nominal(x: np.ndarray, sr: int, t: float, fmin: float = 250.0, fmax: float = 4000.0):
    r = _peaks_db(x, sr, t)
    if r is None:
        return None
    freqs, prom = r
    cand_idx = np.where((freqs >= fmin) & (freqs <= min(fmax, sr / 4)) & (prom > 12) &
                        (prom >= np.roll(prom, 1)) & (prom >= np.roll(prom, -1)))[0]
    if len(cand_idx) == 0:
        return None
    cand_idx = cand_idx[np.argsort(-prom[cand_idx])][:25]
    best = None
    for ci in cand_idx:
        f = freqs[ci]
        score, found = 0.0, []
        for ratio, wgt, name in TEMPLATE:
            ft = f * ratio
            if ft >= sr / 2 - 100 or ft < 40:
                continue
            m = (freqs >= ft * 0.985) & (freqs <= ft * 1.015)
            if not m.any():
                continue
            pk = float(prom[m].max())
            if pk > 8:
                score += wgt * min(pk, 40) / 40
                found.append(name)
        if "nominal" in found and len(found) >= 3 and (best is None or score > best[0]):
            best = (score, f, found)
    if best is None:
        return None
    return {"t": round(t, 3), "nominal_hz": round(float(best[1]), 2), "partials_found": best[2], "score": round(best[0], 3)}


def analyse_strikes(x: np.ndarray, sr: int, onset_times: list[float], tol: float = 0.012) -> dict:
    strikes = [s for s in (strike_nominal(x, sr, t) for t in onset_times[:300]) if s]
    if len(strikes) < 4:
        return {"valid_strikes": len(strikes), "bells": [], "ring_size_observed": 0,
                "reason": "Fewer than 4 strikes with a bell partial structure (nominal plus 2 partials)."}
    f = np.array([s["nominal_hz"] for s in strikes])
    order = np.argsort(f)
    clusters: list[list[int]] = []
    for i in order:
        if clusters and abs(np.log(f[i] / np.median(f[clusters[-1]]))) < tol:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    counts = [len(c) for c in clusters]
    min_count = max(2, int(0.3 * np.median(counts)))
    bells = [{"nominal_hz": round(float(np.median(f[c])), 2), "note": note_name(float(np.median(f[c]))),
              "strikes": len(c), "spread_cents": round(float(1200 * np.std(np.log2(f[c]))), 1)}
             for c in clusters if len(c) >= min_count]
    bells.sort(key=lambda b: b["nominal_hz"])
    steps = [round(12 * np.log2(bells[i + 1]["nominal_hz"] / bells[i]["nominal_hz"]), 2) for i in range(len(bells) - 1)]
    diatonic = bool(steps) and all(0.6 <= s <= 2.4 for s in steps)
    return {"valid_strikes": len(strikes), "bells": bells, "ring_size_observed": len(bells),
            "tenor_nominal_hz": bells[0]["nominal_hz"] if bells else None,
            "semitone_steps": steps, "diatonic_like": diatonic, "strikes": strikes[:200],
            "method": "Partial-template fit per strike (hum, prime, tierce, quint, nominal, superquint, octave), "
                      "clustered at 1.2% tolerance."}
