"""Local speech transcription (Whisper base, ONNX int8; greedy decoding). Runs offline on CPU.

Faint and distant speech is attempted. Those segments are speech-band filtered (100-4000 Hz), noise-reduced
by spectral subtraction and level-normalised before decoding. Each transcript gets quality measures:
- no_speech_prob : Whisper's probability that the segment contains no speech
- avg_logprob    : mean token log-probability (low means the model is guessing)
- compression    : text zlib compression ratio (high means repetitive, a typical hallucination loop)
Quality label: clear / uncertain / likely spurious. Whisper is known to invent plausible text on noise,
music or very faint audio, so only "clear" transcripts should be relied on, and even those can contain errors.
"""
from __future__ import annotations

import base64
import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .frontends import whisper_log_mel
from .runtime import session

FILES = ["whisper_base_asr_encoder_int8.onnx", "whisper_base_asr_decoder_int8.onnx", "whisper_base_asr_tokens.json",
         "whisper_base_mel_filters.npy"]


@lru_cache(maxsize=2)
def _table(md: str) -> dict:
    t = json.loads((Path(md) / "whisper_base_asr_tokens.json").read_text(encoding="utf-8"))
    t["bytes"] = [base64.b64decode(b) for b in t["tokens_b64"]]
    return t


def available(md: str) -> bool:
    return all((Path(md) / f).exists() for f in FILES)


def enhance(x16k: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Speech-band filter + spectral subtraction + RMS normalisation (for faint/distant speech)."""
    n = len(x16k)
    if n < 1024:
        return x16k
    nfft, hop = 512, 128
    w = np.hanning(nfft).astype(np.float32)
    pad = np.pad(x16k, (nfft, nfft))
    frames = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::hop] * w
    S = np.fft.rfft(frames, axis=1)
    mag, ph = np.abs(S), np.angle(S)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f >= 100) & (f <= 4000)
    energy = (mag[:, band] ** 2).sum(axis=1)
    noise = np.median(mag[energy <= np.percentile(energy, 15)], axis=0) if len(mag) > 10 else np.zeros(mag.shape[1])
    clean = np.maximum(mag - 1.5 * noise, 0.08 * mag)
    clean[:, ~band] *= 0.05
    y_frames = np.fft.irfft(clean * np.exp(1j * ph), nfft, axis=1) * w
    y = np.zeros(len(pad), np.float32)
    norm = np.zeros(len(pad), np.float32)
    for i, fr in enumerate(y_frames):
        y[i * hop:i * hop + nfft] += fr
        norm[i * hop:i * hop + nfft] += w ** 2
    y = (y / np.maximum(norm, 1e-6))[nfft:nfft + n]
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    return np.clip(y * (0.1 / rms), -1, 1).astype(np.float32)


def transcribe(x16k: np.ndarray, md: str, lang: str | None = None, max_tokens: int = 220) -> dict:
    md = str(md)
    t = _table(md)
    ids = t["ids"]
    se = session(Path(md) / FILES[0])
    sd = session(Path(md) / FILES[1])
    mel_f = np.load(Path(md) / FILES[3])
    enc = se.run(None, {"input_features": whisper_log_mel(x16k, mel_f)[None]})[0]

    def step(seq):
        return sd.run(None, {"input_ids": np.array([seq], np.int64), "encoder_hidden_states": enc})[0][0].astype(np.float64)

    first = step([ids["sot"]])
    pf = np.exp(first - first.max())
    pf /= pf.sum()
    no_speech = float(pf[ids["nospeech"]])
    if lang is None or f"<|{lang}|>" not in t["lang_to_id"]:
        lang_ids = np.array(list(t["lang_to_id"].values()))
        k = int(lang_ids[np.argmax(first[lang_ids])])
        lang = next(c for c, v in t["lang_to_id"].items() if v == k).strip("<|>")
    seq = [ids["sot"], t["lang_to_id"][f"<|{lang}|>"], ids["transcribe"], ids["notimestamps"]]
    sup = np.array(t["suppress_tokens"], int)
    logps = []
    for i in range(max_tokens):
        lg = step(seq)
        if len(sup):
            lg[sup] = -np.inf
        lg[ids["notimestamps"] + 1:] = -np.inf
        if i == 0 and t["begin_suppress_tokens"]:
            lg[t["begin_suppress_tokens"]] = -np.inf
        m = lg.max()
        lp = lg - m - np.log(np.exp(lg - m).sum())
        nxt = int(np.argmax(lg))
        if nxt == ids["eot"]:
            break
        logps.append(float(lp[nxt]))
        seq.append(nxt)
        if len(seq) > 24 and seq[-8:] == seq[-16:-8] == seq[-24:-16]:  # repetition loop
            break
    text = b"".join(t["bytes"][k] for k in seq[4:] if k < t["special_start"]).decode("utf-8", "replace").strip()
    raw = text.encode("utf-8")
    comp = len(raw) / max(1, len(zlib.compress(raw))) if raw else 0.0
    avg = float(np.mean(logps)) if logps else -10.0
    if not text or (no_speech > 0.6 and avg < -1.0) or comp > 2.4:
        quality = "likely spurious"
    elif no_speech < 0.4 and avg > -0.7:
        quality = "clear"
    else:
        quality = "uncertain"
    return {"text": text, "language": lang, "no_speech_prob": round(no_speech, 3), "avg_logprob": round(avg, 3),
            "compression_ratio": round(comp, 2), "n_tokens": len(logps), "quality": quality}


def speech_candidates(sed: dict, speech_idx: list[int], thr: float = 0.06, min_len: float = 0.6, gap: float = 1.0):
    """Low-threshold speech regions (so that faint and distant speech is attempted)."""
    s = sed["frame_scores"][:, speech_idx].max(axis=1)
    hop = sed["hop_s"]
    on = s >= thr
    regs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if regs and (i - regs[-1][1]) * hop <= gap:
                regs[-1][1] = j
            else:
                regs.append([i, j])
            i = j
        else:
            i += 1
    out = []
    for a, b in regs:
        t0, t1 = a * hop, b * hop
        if t1 - t0 >= min_len:
            out.append((t0, t1, float(s[a:b].max())))
    return out


def run_all(x16k: np.ndarray, sed: dict, speech_idx: list[int], md: str, lang_hint=None) -> list[dict]:
    res = []
    for t0, t1, score in speech_candidates(sed, speech_idx):
        t = t0
        while t < t1 - 0.3:
            e = min(t1, t + 28.0)
            seg = x16k[int(t * 16000):int(e * 16000)]
            level = float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)) if len(seg) else -120
            faint = score < 0.3 or level < -40
            r = transcribe(enhance(seg) if faint else seg, md, lang_hint(t, e) if lang_hint else None)
            res.append({"start": round(t, 2), "end": round(e, 2), "speech_score": round(score, 3), "level_dbfs": round(level, 1),
                        "enhanced": faint, **r})
            t = e
    return res
