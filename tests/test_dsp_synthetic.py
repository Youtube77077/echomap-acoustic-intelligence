"""Deterministic tests with synthetic signals whose ground truth is known exactly."""
import numpy as np
import pytest

from echomap.dsp import events, reverb, spectral, tonal
from echomap.models.frontends import efficientat_mel, whisper_log_mel

SR = 32000
rng = np.random.default_rng(0)


def tone(f, dur, sr=SR, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_spectrogram_peak_frequency():
    x = tone(1000, 2.0) + rng.normal(0, 0.001, int(2 * SR)).astype(np.float32)
    v = spectral.spectrogram_view(x, SR, 0, 2, 0, 4000, "linear", 400, 400)
    row = v["data"].mean(axis=1).argmax()  # row 0 = fmax
    f = v["fmax"] - (row + 0.5) / v["height"] * (v["fmax"] - v["fmin"])
    assert abs(f - 1000) < 20


@pytest.mark.parametrize("scale", ["linear", "log", "mel"])
def test_spectrogram_scales_shape(scale):
    x = tone(440, 1.0)
    v = spectral.spectrogram_view(x, SR, 0, 1, 50, 8000, scale, 300, 200)
    assert v["data"].shape == (200, 300)


def test_waveform_envelope_bounds():
    x = tone(100, 1.0, amp=0.5)
    e = spectral.waveform_envelope(x, 100)
    assert max(e["max"]) == pytest.approx(0.5, abs=0.01) and min(e["min"]) == pytest.approx(-0.5, abs=0.01)


def test_features_centroid_tracks_tone():
    s = spectral.summarize(spectral.frame_features(tone(3000, 1.0), SR))
    assert abs(s["spectral_centroid_hz"]["p50"] - 3000) < 150


@pytest.mark.parametrize("nominal", [50, 60])
def test_mains_hum_detection(nominal):
    sr = 2000
    t = np.arange(30 * sr) / sr
    x = (0.02 * np.sin(2 * np.pi * nominal * t) + 0.01 * np.sin(2 * np.pi * 2 * nominal * t + 1) +
         0.008 * np.sin(2 * np.pi * 3 * nominal * t + 2) + rng.normal(0, 0.01, len(t))).astype(np.float32)
    m = tonal.mains_hum(x, sr)
    assert m["detected"] and m["nominal_hz"] == nominal and m["mains_consistent"]


def test_mains_hum_absent_in_noise():
    x = rng.normal(0, 0.05, 30 * 2000).astype(np.float32)
    assert not tonal.mains_hum(x, 2000)["detected"]


def test_periodic_beeps_1hz_locator_pattern():
    x = rng.normal(0, 0.002, 12 * SR).astype(np.float32)
    for k in range(11):
        a = int((0.5 + k * 1.0) * SR)
        x[a:a + int(0.1 * SR)] += tone(880, 0.1)
    tracks = tonal.tone_tracks(x, SR)
    pats = tonal.periodic_tone_patterns(tracks)
    assert pats, "no periodic pattern found"
    p = pats[0]
    assert abs(p["period_s"] - 1.0) < 0.03 and abs(p["f_hz"] - 880) < 30 and p["count"] >= 8


def test_click_train():
    times = [0.5 + 0.2 * i for i in range(20)]
    ct = tonal.click_trains(times)
    assert ct and abs(ct[0]["period_s"] - 0.2) < 1e-6 and ct[0]["count"] == 20


def test_click_train_scales_linearly():
    import time
    times = sorted(rng.uniform(0, 300, 3000).tolist())  # irregular onsets, worst case for the old O(n^3) scan
    t0 = time.perf_counter()
    tonal.click_trains(times)
    assert time.perf_counter() - t0 < 2.0


def test_two_tone_siren():
    sr = SR
    x = np.concatenate([tone(440 if i % 2 == 0 else 587, 0.6) for i in range(10)])
    pat = tonal.siren_pattern(x, sr, 0, len(x) / sr)
    assert "two-tone" in pat["pattern"]


def test_wail_siren():
    t = np.arange(8 * SR) / SR
    f = 900 + 400 * np.sin(2 * np.pi * t / 4.0)
    x = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    assert "slow sweep" in tonal.siren_pattern(x, SR, 0, 8)["pattern"]


def test_engine_f0_rising():
    sr = 4000
    t = np.arange(6 * sr) / sr
    f0 = 25 + 10 * t  # 25 -> 85 Hz firing frequency
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(h * ph) / h for h in range(1, 7)).astype(np.float32) * 0.2
    r = tonal.engine_f0_track(x, sr)
    assert r["available"]
    assert any(s["state"] == "accelerating" for s in r["states"])
    assert 40 < r["f0_median_hz"] < 70


def test_onsets_and_periodicity():
    x = rng.normal(0, 0.001, 10 * SR).astype(np.float32)
    for k in range(20):
        a = int((0.25 + k * 0.5) * SR)
        x[a:a + 200] += 0.8
    on = events.onsets(x, SR)
    assert 18 <= len(on["times_s"]) <= 22
    per = events.periodicity(on["_env"], on["frame_period_s"])
    assert any(abs(p["period_s"] - 0.5) < 0.03 for p in per)


def test_snr_estimate_orders():
    quiet = rng.normal(0, 0.001, 5 * SR).astype(np.float32)
    loud = quiet.copy()
    loud[SR:2 * SR] += tone(1000, 1.0, amp=0.5)
    assert events.snr_estimate(loud, SR)["snr_db"] > events.snr_estimate(quiet, SR)["snr_db"] + 20


def test_rt60_synthetic_decay():
    x = rng.normal(0, 1e-4, 15 * SR).astype(np.float32)
    rt = 0.8
    n = int(2.5 * SR)
    env = 10 ** (-3 * np.arange(n) / SR / rt)  # -60 dB at rt seconds
    for k in range(6):
        a = int((0.3 + k * 1.9) * SR)
        x[a:a + n] += (rng.normal(0, 0.5, n) * env).astype(np.float32)
    on = [0.3 + k * 1.9 for k in range(6)]
    r = reverb.rt60_blind(x, SR, on)
    assert r["rt60_s"] is not None and abs(r["rt60_s"] - rt) < 0.25


def test_frontend_shapes():
    mb = np.ones((128, 513), np.float32) / 513
    assert efficientat_mel(tone(1000, 5.0), mb).shape == (128, 500)  # pre-emphasis drops one sample, as in the reference
    mf = np.ones((80, 201), np.float32) / 201
    assert whisper_log_mel(tone(500, 2.0, sr=16000), mf).shape == (80, 3000)


def _codrift(f1, f2, dur=5.0, drift_pct_s=2.0):
    t = np.arange(int(dur * SR)) / SR
    k = 1 + drift_pct_s / 100 * t
    x = sum(0.15 * np.sin(2 * np.pi * np.cumsum(f * k) / SR) for f in (f1, f2))
    return (x + rng.normal(0, 0.003, len(t))).astype(np.float32)


def test_avas_like_detects_codrifting_tones():
    tr = tonal.tone_tracks(_codrift(520, 1300), SR)
    assert tonal.avas_like(tr), "co-drifting AVAS-style tones not detected"


def test_avas_like_ignores_static_tones_and_sirens():
    static = tone(800, 5.0) + tone(1900, 5.0)
    assert not tonal.avas_like(tonal.tone_tracks(static, SR))
    t = np.arange(8 * SR) / SR
    f = 900 + 400 * np.sin(2 * np.pi * t / 4.0)
    wail = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    assert not tonal.avas_like(tonal.tone_tracks(wail, SR))
