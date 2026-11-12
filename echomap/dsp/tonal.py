"""Narrowband / tonal analysis: mains hum (ENF), tone tracking, siren patterns, periodic signals, engine f0."""
from __future__ import annotations

import numpy as np

from .spectral import stft_power


def _interp_peak(spec: np.ndarray, k: int) -> float:
    """Quadratic interpolation of a spectral peak (on log power). Returns a fractional bin."""
    if k <= 0 or k >= len(spec) - 1:
        return float(k)
    a, b, c = np.log(spec[k - 1] + 1e-20), np.log(spec[k] + 1e-20), np.log(spec[k + 1] + 1e-20)
    d = a - 2 * b + c
    return float(k + (0.5 * (a - c) / d if d != 0 else 0.0))


# ----------------------------------------------------------------------------- mains hum / ENF
def mains_hum(x_2k: np.ndarray, sr: int = 2000) -> dict:
    """Detect 50 vs 60 Hz mains hum (and harmonics) and extract an ENF trace.

    x_2k: audio resampled to `sr` (2 kHz is enough for harmonics up to 240 Hz).
    """
    dur = len(x_2k) / sr
    # Welch averaging: at least 7 half-overlapping segments so noise peaks cannot pass as hum
    seg_s = float(np.clip(dur / 4.0, 1.0, 8.0))
    seg = int(2 ** np.round(np.log2(seg_s * sr)))
    hop = seg // 2
    p = stft_power(x_2k, seg, hop, center=False)
    psd = p.mean(axis=1) if p.shape[1] < 9 else np.median(p, axis=1)
    df = sr / seg
    freqs = np.arange(len(psd)) * df
    out = {"freq_resolution_hz": round(df, 4), "families": {}}
    for nominal in (50, 60):
        harm = []
        for h in (1, 2, 3, 4):
            f0 = nominal * h
            if f0 + 3 > sr / 2:
                break
            pk = (freqs >= f0 - 0.6 * h ** 0.5) & (freqs <= f0 + 0.6 * h ** 0.5)
            bg = ((freqs >= f0 - 4) & (freqs <= f0 - 1.5)) | ((freqs >= f0 + 1.5) & (freqs <= f0 + 4))
            if not pk.any() or not bg.any():
                continue
            k = np.where(pk)[0][np.argmax(psd[pk])]
            snr = 10 * np.log10(psd[k] / (np.median(psd[bg]) + 1e-20) + 1e-20)
            fpk = _interp_peak(psd, k) * df
            harm.append({"harmonic": h, "freq_hz": round(fpk, 3), "snr_db": round(float(snr), 1)})
        strong = [hh for hh in harm if hh["snr_db"] >= 8]
        score = sum(max(0.0, hh["snr_db"] - 8) for hh in harm)
        out["families"][nominal] = {"harmonics": harm, "n_strong": len(strong), "score": round(score, 1)}
    f50, f60 = out["families"][50], out["families"][60]
    best = 50 if f50["score"] >= f60["score"] else 60
    fb, fo = out["families"][best], out["families"][110 - best]
    fund = next((h for h in fb["harmonics"] if h["harmonic"] == 1), None)
    n_seg = p.shape[1]
    detected = ((fb["n_strong"] >= 3 or (fb["n_strong"] >= 2 and fund and fund["snr_db"] >= 15))
                and fb["score"] > 2 * fo["score"] + 6 and n_seg >= 5)
    out["detected"] = bool(detected)
    out["segments_averaged"] = int(n_seg)
    out["duration_s"] = round(dur, 2)
    out["nominal_hz"] = best if detected else None
    # ENF trace from the strongest harmonic of the winning family
    if detected:
        h = max(fb["harmonics"], key=lambda r: r["snr_db"])
        fc = best * h["harmonic"]
        band = (freqs >= fc - 1.0) & (freqs <= fc + 1.0)
        idx = np.where(band)[0]
        trace = []
        for t in range(p.shape[1]):
            col = p[:, t]
            k = idx[np.argmax(col[idx])]
            trace.append(_interp_peak(col, k) * df / h["harmonic"])
        trace = np.array(trace)
        out["enf_trace"] = {"t_s": [round(i * hop / sr + seg / sr / 2, 2) for i in range(len(trace))],
                            "freq_hz": [round(float(v), 4) for v in trace], "harmonic_used": h["harmonic"]}
        dev = trace - best
        out["enf_stats"] = {"median_offset_hz": round(float(np.median(dev)), 4), "std_hz": round(float(dev.std()), 4)}
        out["mains_consistent"] = bool(abs(np.median(dev)) < 0.25 and dev.std() < 0.15)
        out["confidence_note"] = ("Peak SNRs over the harmonic series are deterministic measurements. A stable tone "
                                  "within ±0.25 Hz of nominal is consistent with mains interference, but motors, "
                                  "transformers in imported devices, or synthetic tones can mimic it.")
    return out


# ----------------------------------------------------------------------------- tone tracking
def tone_tracks(x: np.ndarray, sr: int, f_lo: float = 250.0, f_hi: float = 8000.0, prom_db: float = 12.0,
                hop_s: float = 0.01, min_dur: float = 0.04) -> list[dict]:
    """Link prominent narrowband peaks across frames into tonal tracks."""
    hop = max(1, int(hop_s * sr))
    n_fft = 2048 if sr >= 32000 else 1024
    p = stft_power(x, n_fft, hop)
    db = 10 * np.log10(p + 1e-14)
    freqs = np.linspace(0, sr / 2, p.shape[0])
    a, b = np.searchsorted(freqs, f_lo), np.searchsorted(freqs, min(f_hi, sr / 2 - 1))
    sub = db[a:b]
    # local background: mean level (dB) over +-15 bins excluding +-2 around the bin (cumulative sums, O(N))
    w, ex = 15, 2
    pad = np.pad(sub, ((w, w), (0, 0)), mode="edge").astype(np.float64)
    cs = np.vstack([np.zeros((1, pad.shape[1])), np.cumsum(pad, axis=0)])
    k = np.arange(sub.shape[0]) + w
    tot = cs[k + w + 1] - cs[k - w]
    inner = cs[k + ex + 1] - cs[k - ex]
    bg = (tot - inner) / (2 * w + 1 - (2 * ex + 1))
    prom = (sub - bg).astype(np.float32)
    active: list[dict] = []
    done: list[dict] = []
    df = freqs[1]
    for t in range(sub.shape[1]):
        col = prom[:, t]
        pk = np.where((col > prom_db) & (col >= np.roll(col, 1)) & (col >= np.roll(col, -1)))[0]
        pk = pk[np.argsort(-col[pk])][:6]
        used = set()
        for tr in active:
            best, bd = None, 1e9
            for k in pk:
                if k in used:
                    continue
                d = abs((k + a) * df - tr["f"][-1])
                if d < bd and d <= max(40.0, 0.06 * tr["f"][-1]):
                    best, bd = k, d
            if best is not None:
                used.add(best)
                tr["f"].append((best + a) * df)
                tr["db"].append(float(sub[best, t]))
                tr["end"] = t
                tr["miss"] = 0
            else:
                tr["miss"] += 1
        for k in pk:
            if k not in used:
                active.append({"start": t, "end": t, "f": [(k + a) * df], "db": [float(sub[k, t])], "miss": 0})
        still = []
        for tr in active:
            (still if tr["miss"] <= 2 else done).append(tr)
        active = still
    done.extend(active)
    out = []
    for tr in done:
        dur = (tr["end"] - tr["start"] + 1) * hop / sr
        if dur >= min_dur:
            f = np.array(tr["f"])
            out.append({"start": round(tr["start"] * hop / sr, 3), "end": round((tr["end"] + 1) * hop / sr, 3),
                        "duration": round(dur, 3), "f_median": round(float(np.median(f)), 1),
                        "f_min": round(float(f.min()), 1), "f_max": round(float(f.max()), 1),
                        "level_db": round(float(np.median(tr["db"])), 1),
                        "contour": [round(float(v), 1) for v in f[:: max(1, len(f) // 200)]],
                        "contour_hop_s": hop / sr * max(1, len(f) // 200)})
    return sorted(out, key=lambda r: r["start"])


def periodic_tone_patterns(tracks: list[dict], tol_f: float = 0.03, max_cv: float = 0.08, min_count: int = 5):
    """Find repeating short tones (beeps) with stable pitch, regular period and clear gaps.

    Conditions (all must hold): each pulse is pitch-stable (f_max/f_min ≤ 1.04), pulses repeat with
    period CV ≤ max_cv, duty cycle ≤ 0.7, and at least `min_count` pulses. Harmonically related
    patterns with the same timing are merged and the strongest is kept, so a single beeper does not
    produce one pattern per harmonic.
    """
    short = [t for t in tracks if t["duration"] <= 0.6 and t["f_max"] <= 1.04 * max(t["f_min"], 1)]
    short = sorted(short, key=lambda r: -r["level_db"])[:1500]
    short.sort(key=lambda r: r["start"])
    used = set()
    patterns = []
    for i, t in enumerate(short):
        if i in used:
            continue
        grp = [i] + [j for j in range(i + 1, len(short))
                     if j not in used and abs(short[j]["f_median"] - t["f_median"]) <= tol_f * t["f_median"]]
        if len(grp) < min_count:
            continue
        starts = np.array([short[j]["start"] for j in grp])
        durs_all = np.array([short[j]["duration"] for j in grp])
        keep = [0]
        for k in range(1, len(starts)):
            if starts[k] - starts[keep[-1]] > 0.08:
                keep.append(k)
        starts, durs = starts[keep], durs_all[keep]
        if len(starts) < min_count:
            continue
        iv = np.diff(starts)
        med = np.median(iv)
        regular = iv[np.abs(iv - med) < 0.2 * med]
        if len(regular) < min_count - 1 or len(regular) < 0.75 * len(iv):
            continue
        cv = regular.std() / (regular.mean() + 1e-9)
        duty = float(np.median(durs) / regular.mean())
        if cv > max_cv or duty > 0.7:
            continue
        used.update(grp)
        patterns.append({"kind": "periodic_tone", "f_hz": round(float(np.median([short[j]["f_median"] for j in grp])), 1),
                         "period_s": round(float(regular.mean()), 3), "rate_hz": round(float(1 / regular.mean()), 3),
                         "period_cv": round(float(cv), 3), "pulse_duration_s": round(float(np.median(durs)), 3),
                         "duty_cycle": round(duty, 2), "count": int(len(starts)), "start": float(starts[0]),
                         "end": round(float(starts[-1] + np.median(durs)), 3),
                         "level_db": round(float(np.median([short[j]["level_db"] for j in grp])), 1)})
    # merge harmonics / duplicates sharing timing
    patterns.sort(key=lambda p: -p["level_db"])
    merged = []
    for p in patterns:
        dup = None
        for q in merged:
            same_t = abs(p["period_s"] - q["period_s"]) < 0.03 * q["period_s"] and min(p["end"], q["end"]) - max(p["start"], q["start"]) > 0.5 * (p["end"] - p["start"])
            ratio = max(p["f_hz"], q["f_hz"]) / max(1.0, min(p["f_hz"], q["f_hz"]))
            if same_t and abs(ratio - round(ratio)) < 0.06:
                dup = q
                break
        if dup:
            dup.setdefault("harmonics_hz", []).append(p["f_hz"])
        else:
            merged.append(p)
    return sorted(merged, key=lambda p: p["start"])


def click_trains(onset_times: list[float], min_count: int = 5, max_cv: float = 0.1, max_len: int = 400):
    """Regularly spaced transients (e.g. ticking pedestrian signals, clocks, turn indicators).

    Greedy left-to-right scan, O(n * max_len): a train is extended while its interval CV stays below max_cv.
    """
    t = np.asarray(onset_times, dtype=float)
    out = []
    n = len(t)
    i = 0
    while i <= n - min_count:
        iv = np.diff(t[i:i + min_count])
        m = iv.mean()
        if not (0.08 <= m <= 3.0) or iv.std() / (m + 1e-9) > max_cv:
            i += 1
            continue
        j = i + min_count
        while j < n and j - i < max_len:
            iv2 = np.diff(t[i:j + 1])
            if iv2.std() / (iv2.mean() + 1e-9) > max_cv:
                break
            j += 1
        iv = np.diff(t[i:j])
        out.append({"kind": "click_train", "start": round(float(t[i]), 3), "end": round(float(t[j - 1]), 3),
                    "period_s": round(float(iv.mean()), 3), "rate_hz": round(float(1 / iv.mean()), 3),
                    "period_cv": round(float(iv.std() / iv.mean()), 3), "count": int(j - i)})
        i = j
    return out


def siren_pattern(x: np.ndarray, sr: int, t0: float, t1: float) -> dict:
    """Classify a siren's pitch pattern: two-tone alternation, slow sweep (wail), fast sweep (yelp)."""
    seg = x[int(t0 * sr):int(t1 * sr)]
    if len(seg) < sr // 2:
        return {"pattern": "undetermined", "reason": "segment too short"}
    hop = int(0.01 * sr)
    n_fft = 4096 if sr >= 32000 else 2048
    p = stft_power(seg, n_fft, hop)
    freqs = np.linspace(0, sr / 2, p.shape[0])
    m = (freqs >= 300) & (freqs <= 2000)
    f = freqs[m][np.argmax(p[m], axis=0)]
    f = np.convolve(f, np.ones(5) / 5, mode="same")
    if len(f) < 50:
        return {"pattern": "undetermined", "reason": "segment too short"}
    lo, hi = np.percentile(f, 10), np.percentile(f, 90)
    if hi / max(lo, 1) < 1.08:
        return {"pattern": "steady tone", "f_hz": round(float(np.median(f)), 1)}
    hist, edges = np.histogram(f, bins=24, range=(lo * 0.95, hi * 1.05))
    mid = (lo + hi) / 2
    frac_mid = np.mean((f > lo + 0.25 * (hi - lo)) & (f < hi - 0.25 * (hi - lo)))
    crossings = np.where(np.diff(np.sign(f - mid)) != 0)[0]
    if len(crossings) < 2:
        return {"pattern": "single sweep", "f_range_hz": [round(float(lo), 1), round(float(hi), 1)]}
    cycle = 2 * np.median(np.diff(crossings)) * hop / sr
    res = {"f_range_hz": [round(float(lo), 1), round(float(hi), 1)], "cycle_s": round(float(cycle), 3),
           "fraction_time_in_transition": round(float(frac_mid), 3)}
    if frac_mid < 0.15:
        res["pattern"] = "two-tone alternation (hi-lo)"
        res["interval_ratio"] = round(float(hi / lo), 3)
    elif cycle < 0.8:
        res["pattern"] = "fast sweep (yelp-like)"
    else:
        res["pattern"] = "slow sweep (wail-like)"
    return res


# ----------------------------------------------------------------------------- engine harmonics
def engine_f0_track(x_4k: np.ndarray, sr: int = 4000, t0: float = 0.0, t1: float | None = None,
                    f_lo: float = 12.0, f_hi: float = 250.0) -> dict:
    """Harmonic-sum fundamental tracking for engine sounds (firing frequency)."""
    t1 = t1 if t1 is not None else len(x_4k) / sr
    seg = x_4k[int(t0 * sr):int(t1 * sr)]
    if len(seg) < sr // 2:
        return {"available": False, "reason": "segment shorter than 0.5 s"}
    n_fft, hop = 4096, int(0.05 * sr)
    p = stft_power(seg, n_fft, hop)
    lp = np.log(p + 1e-12)
    lp -= np.median(lp, axis=0, keepdims=True)
    df = sr / n_fft
    cands = np.arange(f_lo, f_hi, 0.5)
    H = 8
    w = 1.0 / np.arange(1, H + 1) ** 0.5
    score = np.zeros((len(cands), p.shape[1]))
    for h in range(1, H + 1):
        k = np.clip(np.round(cands * h / df).astype(int), 0, p.shape[0] - 1)
        score += w[h - 1] * lp[k]
    # subharmonic penalty: prefer the highest f0 among near-equal octave candidates
    best = np.argmax(score, axis=0)
    f0 = cands[best]
    sal = (score.max(axis=0) - np.median(score, axis=0)) / (score.std(axis=0) + 1e-9)
    good = sal > 2.5
    if good.sum() < 5:
        return {"available": False, "reason": "no stable harmonic series found (broadband or masked sound)"}
    t = t0 + np.arange(len(f0)) * hop / sr
    f0s = f0.astype(float)
    f0s[~good] = np.nan
    # median smoothing
    sm = np.array([np.nanmedian(f0s[max(0, i - 3):i + 4]) if np.isfinite(f0s[max(0, i - 3):i + 4]).any() else np.nan
                   for i in range(len(f0s))])
    valid = np.isfinite(sm)
    fv, tv = sm[valid], t[valid]
    # local slopes (% per second) over 1 s windows
    states = []
    win = int(1.0 / (hop / sr))
    for i in range(0, len(fv) - win, max(1, win // 2)):
        a, b = fv[i:i + win], tv[i:i + win]
        if len(a) < win // 2:
            continue
        sl = np.polyfit(b, a, 1)[0] / (np.mean(a) + 1e-9) * 100
        lab = "accelerating" if sl > 3 else ("decelerating" if sl < -3 else "steady")
        states.append({"start": round(float(b[0]), 2), "end": round(float(b[-1]), 2), "slope_pct_per_s": round(float(sl), 1),
                       "state": lab})
    gear = []
    for i in range(1, len(fv)):
        if tv[i] - tv[i - 1] < 0.3 and fv[i] < 0.85 * fv[i - 1]:
            pre = fv[max(0, i - 10):i]
            post = fv[i:i + 10]
            if len(pre) > 4 and len(post) > 4 and np.polyfit(range(len(pre)), pre, 1)[0] > 0 and np.polyfit(range(len(post)), post, 1)[0] > 0:
                gear.append(round(float(tv[i]), 2))
    med = float(np.nanmedian(fv))
    jumps = np.abs(np.diff(fv)) / (fv[:-1] + 1e-9) if len(fv) > 1 else np.zeros(1)
    continuity = float(np.mean(jumps < 0.08)) * float(valid.mean())
    rpm = {f"{c}-cyl 4-stroke": round(med * 120 / c) for c in (2, 3, 4, 6, 8)}
    return {"available": True, "f0_median_hz": round(med, 1), "f0_range_hz": [round(float(np.nanmin(fv)), 1), round(float(np.nanmax(fv)), 1)],
            "t_s": [round(float(v), 2) for v in tv[::2]], "f0_hz": [round(float(v), 1) for v in fv[::2]],
            "states": states, "possible_gear_changes_s": gear, "continuity": round(continuity, 3),
            "rpm_if": rpm,
            "note": ("f0 is the strongest harmonic-series fundamental, usually the engine firing frequency. RPM values "
                     "are conditional on an assumed cylinder count (RPM = f0 x 120 / cylinders for 4-stroke) and are "
                     "not identifications. Gear-change times are heuristic: a sudden >15% f0 drop between rising segments.")}


# ----------------------------------------------------------------------------- AVAS (electric-vehicle warning sound)
def avas_like(tracks: list[dict], min_dur: float = 1.5, min_corr: float = 0.85) -> list[dict]:
    """Detect AVAS-like sounds: concurrent, sustained tonal components (160 Hz-5 kHz) whose frequencies
    drift together, as required by UN R138 (a tone shifting on average >= 0.8 % per km/h between
    5 and 20 km/h). Returns candidate segments. This is a pattern detector, not an identification.
    """
    long = [t for t in tracks if t["duration"] >= min_dur and 160 <= t["f_median"] <= 5000 and len(t["contour"]) >= 8]
    out, used = [], set()
    for i, a in enumerate(long):
        for j in range(i + 1, len(long)):
            b = long[j]
            if i in used or j in used:
                continue
            t0, t1 = max(a["start"], b["start"]), min(a["end"], b["end"])
            if t1 - t0 < min_dur or abs(a["f_median"] - b["f_median"]) < 0.05 * a["f_median"]:
                continue
            ratio = max(a["f_median"], b["f_median"]) / min(a["f_median"], b["f_median"])
            if any(abs(ratio - p_ / q_) < 0.02 * ratio for q_ in range(1, 7) for p_ in range(q_ + 1, 7 * q_) if p_ <= 6):
                continue  # small-integer ratio: partials of one harmonic source (voice, bird, siren, engine)

            def seg(t):
                n = len(t["contour"])
                ts = t["start"] + np.arange(n) * t["contour_hop_s"]
                m = (ts >= t0) & (ts <= t1)
                return ts[m], np.array(t["contour"])[m]
            ta, fa = seg(a)
            tb, fb = seg(b)
            if len(fa) < 6 or len(fb) < 6:
                continue
            grid_t = np.linspace(t0, t1, 30)
            ra = np.interp(grid_t, ta, fa)
            rb = np.interp(grid_t, tb, fb)
            ra, rb = ra / ra.mean(), rb / rb.mean()
            drift = float(ra.max() - ra.min())
            if drift < 0.01 or drift > 0.6 or ra.std() < 1e-6 or rb.std() < 1e-6:
                continue  # static tones (beepers) or wild sweeps (sirens, birds) are excluded
            corr = float(np.corrcoef(ra, rb)[0, 1])
            ratio_stability = float(np.std(ra / rb))
            if corr >= min_corr and ratio_stability < 0.02:
                used.update((i, j))
                slope = float(np.polyfit(grid_t, ra, 1)[0] * 100)
                out.append({"start": round(t0, 2), "end": round(t1, 2), "f_hz": sorted([a["f_median"], b["f_median"]]),
                            "relative_drift": round(drift, 3), "drift_pct_per_s": round(slope, 2), "co_drift_corr": round(corr, 3)})
    return out
