"""Forensic integrity checks on the decoded signal and the container metadata.

Each check reports what was measured, the threshold used, and how to interpret it. These are
indicators that warrant attention, not proof of editing. Lossy codecs, device DSP and ordinary
editing can all produce them.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .spectral import stft_power


def _parse_time(s: str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def bandwidth_cutoff(x: np.ndarray, sr: int) -> dict:
    """Highest frequency with content above the noise floor. A steep cut-off well below Nyquist
    (e.g. ~16 kHz at 44.1/48 kHz) is typical of lossy encoding (MP3/AAC) somewhere in the file's history."""
    n_fft = 8192 if sr >= 32000 else 4096
    p = stft_power(x, n_fft, n_fft // 2).mean(axis=1)
    db = 10 * np.log10(p + 1e-20)
    freqs = np.linspace(0, sr / 2, len(db))
    sm = np.convolve(db, np.ones(9) / 9, mode="same")
    floor = np.percentile(sm[freqs > sr * 0.45], 50) if (freqs > sr * 0.45).any() else sm.min()
    above = np.where(sm > floor + 12)[0]
    fc = float(freqs[above[-1]]) if len(above) else 0.0
    # steepness: drop over the 1 kHz above the cut-off
    i = int(np.searchsorted(freqs, fc))
    j = int(np.searchsorted(freqs, min(sr / 2, fc + 1000)))
    steep = float(sm[max(0, i - 3):i + 1].mean() - sm[j - 3:j + 1].mean()) if j > i + 3 else 0.0
    lossy_like = fc < 0.9 * (sr / 2) and steep > 20 and fc > 4000
    return {"effective_bandwidth_hz": round(fc), "nyquist_hz": sr / 2, "edge_steepness_db_per_khz": round(steep, 1),
            "lossy_history_indicated": bool(lossy_like),
            "interpretation": ("Steep spectral cut-off well below Nyquist: consistent with lossy encoding (MP3/AAC/Opus) at "
                               "some point in the file's history." if lossy_like else
                               "No steep low-pass cut-off detected below Nyquist.")}


def clipping(x: np.ndarray) -> dict:
    a = np.abs(x)
    peak = float(a.max()) if len(a) else 0.0
    thr = 0.999 * peak if peak > 0.9 else 1.01
    hits = np.concatenate([[0], (a >= thr).astype(np.int8), [0]])
    edges = np.diff(hits)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    lengths = ends - starts
    runs = int(np.sum(lengths >= 3))
    longest = int(lengths.max()) if len(lengths) else 0
    return {"peak": round(peak, 5), "flat_top_runs_ge3": runs, "longest_run_samples": longest,
            "clipping_indicated": bool(runs >= 5),
            "interpretation": ("Repeated flat-topped peaks: the signal was clipped (overload at recording or processing)."
                               if runs >= 5 else "No significant clipping.")}


def dropouts(x: np.ndarray, sr: int, min_ms: float = 20.0) -> dict:
    """Runs of exact digital zeros inside the recording (not at the edges): possible dropouts or splices."""
    z = np.concatenate([[0], (x == 0.0).astype(np.int8), [0]])
    n_min = int(min_ms / 1000 * sr)
    n = len(x)
    e = np.diff(z)
    st, en = np.where(e == 1)[0], np.where(e == -1)[0]
    keep = ((en - st) >= n_min) & (st > sr * 0.05) & (en < n - sr * 0.05)
    out = [{"start": round(a / sr, 3), "end": round(b / sr, 3)} for a, b in zip(st[keep], en[keep])]
    return {"digital_silence_gaps": out[:50], "count": len(out),
            "interpretation": ("Exact digital silence inside the recording: may indicate dropouts, muting or editing points."
                               if out else "No internal digital-silence gaps.")}


def dc_offset(x: np.ndarray) -> dict:
    m = float(np.mean(x)) if len(x) else 0.0
    return {"dc_offset": round(m, 6), "significant": bool(abs(m) > 0.01),
            "interpretation": "Significant DC offset (hardware or processing artefact)." if abs(m) > 0.01 else "Negligible DC offset."}


def metadata_consistency(meta: dict, decoded_s: float) -> list[dict]:
    issues = []
    d = meta.get("duration_s")
    if d and abs(d - decoded_s) > max(0.5, 0.02 * d):
        issues.append({"check": "duration", "finding": f"Container duration {d:.2f} s vs decoded {decoded_s:.2f} s",
                       "interpretation": "Container header disagrees with the audio data (truncation, concatenation or a broken header)."})
    tags = {k.lower(): v for k, v in (meta.get("embedded_times") or {}).items()}
    fs = meta.get("filesystem_times") or {}
    for k, v in tags.items():
        t = _parse_time(str(v))
        m = _parse_time((fs.get("modified_utc") or "")[:19].replace("T", " "))
        if t and m and t.replace(tzinfo=None) > m.replace(tzinfo=None):
            issues.append({"check": "time order", "finding": f"Embedded '{k}' {v} is later than the file modification time",
                           "interpretation": "Inconsistent clocks or metadata rewritten after the file was saved."})
        if t and t.year < 1995:
            issues.append({"check": "implausible date", "finding": f"Embedded '{k}' = {v}",
                           "interpretation": "Device clock not set (default date)."})
    enc = " ".join(str(v) for v in (meta.get("device") or {}).values()).lower()
    if any(s in enc for s in ("lavf", "lavc", "ffmpeg", "audacity", "adobe", "audition", "reaper", "garageband")):
        issues.append({"check": "processing software", "finding": f"Encoder/software tag: {enc[:120]}",
                       "interpretation": "The file was written or re-encoded by editing or conversion software, so it is not "
                                         "necessarily the device original."})
    if not meta.get("embedded_times"):
        issues.append({"check": "missing creation time", "finding": "No embedded recording/creation time",
                       "interpretation": "Common for exported or converted files. Recording time cannot be taken from metadata."})
    return issues


def run(x: np.ndarray, sr: int, meta: dict) -> dict:
    checks = {"bandwidth": bandwidth_cutoff(x, sr), "clipping": clipping(x), "dropouts": dropouts(x, sr),
              "dc_offset": dc_offset(x), "metadata": metadata_consistency(meta, len(x) / sr)}
    flags = []
    if checks["bandwidth"]["lossy_history_indicated"] and str(meta.get("primary_audio", {}).get("codec", "")).startswith(("pcm", "flac")):
        flags.append("Lossless container with a lossy-like spectrum: the audio was probably transcoded from a lossy source.")
    if checks["clipping"]["clipping_indicated"]:
        flags.append("Clipping present.")
    if checks["dropouts"]["count"]:
        flags.append(f"{checks['dropouts']['count']} internal digital-silence gap(s).")
    flags += [i["finding"] for i in checks["metadata"] if i["check"] in ("duration", "time order", "processing software")]
    checks["summary_flags"] = flags
    checks["note"] = ("These are indicators that warrant attention, not proof of tampering. Editing detection would need "
                      "dedicated methods (e.g. ENF continuity against grid data, double-compression analysis).")
    return checks
