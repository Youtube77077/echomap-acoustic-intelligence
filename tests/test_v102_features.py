"""Tests for v1.0.2 features: integrity, stereo direction, bells + Dove, relations, text search, footprints."""
import numpy as np
import pytest

from echomap.dsp import bells, integrity, spatial

SR = 32000
rng = np.random.default_rng(3)


def test_integrity_detects_lowpassed_lossy_like_spectrum_and_clipping():
    x = rng.normal(0, 0.1, 5 * 48000).astype(np.float32)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / 48000)
    X[f > 16000] = 0  # MP3-like 16 kHz shelf
    y = np.fft.irfft(X, len(x)).astype(np.float32)
    assert integrity.bandwidth_cutoff(y, 48000)["lossy_history_indicated"]
    assert not integrity.bandwidth_cutoff(x, 48000)["lossy_history_indicated"]
    c = np.clip(x * 20, -1, 1)
    assert integrity.clipping(c)["clipping_indicated"]
    assert not integrity.clipping(x * 0.5)["clipping_indicated"]


def test_integrity_dropouts():
    x = rng.normal(0, 0.1, 3 * SR).astype(np.float32)
    x[SR:SR + 2000] = 0
    d = integrity.dropouts(x, SR)
    assert d["count"] == 1 and abs(d["digital_silence_gaps"][0]["start"] - 1.0) < 0.01


def test_stereo_direction_lateral_side_and_mono_unsupported():
    s = rng.normal(0, 0.3, 2 * SR).astype(np.float32)
    lag = 12  # samples: left leads (sound from the left)
    left, right = s, np.concatenate([np.zeros(lag, np.float32), s[:-lag]])
    xs = np.stack([left, right])
    assert spatial.channel_report(xs, SR)["supported"]
    d = spatial.event_direction(xs, SR, 0.2, 1.8)
    assert d["lateral"] == "right" or d["lateral"] == "left"
    assert abs(abs(d["itd_ms"]) - lag / SR * 1000) < 0.05
    assert not spatial.channel_report(np.stack([s, s]), SR)["supported"]
    assert not spatial.channel_report(s[None], SR)["supported"]


def _bell(nominal, sr=SR, dur=1.2):
    t = np.arange(int(dur * sr)) / sr
    parts = [(0.25, 0.6, 0.9), (0.5, 0.8, 1.2), (0.5946, 0.7, 1.6), (0.749, 0.3, 2.0), (1.0, 1.0, 2.5), (1.5, 0.4, 4.0), (2.0, 0.3, 5.0)]
    return sum(a * np.exp(-t * dcy) * np.sin(2 * np.pi * nominal * r * t) for r, a, dcy in parts).astype(np.float32) * 0.2


def test_bell_partial_template_recovers_nominals_of_a_ring():
    noms = [1000.0, 891.0, 794.0, 749.0, 667.0, 594.0]  # descending diatonic-like ring
    x = np.zeros(int(20 * SR), np.float32)
    onsets = []
    t = 0.5
    for rnd in range(3):
        for f in noms:
            seg = _bell(f)
            a = int(t * SR)
            x[a:a + len(seg)] += seg
            onsets.append(t)
            t += 0.9
    x += rng.normal(0, 0.002, len(x)).astype(np.float32)
    r = bells.analyse_strikes(x, SR, onsets)
    got = sorted(b["nominal_hz"] for b in r["bells"])
    assert r["ring_size_observed"] == 6
    for g, w in zip(got, sorted(noms)):
        assert abs(1200 * np.log2(g / w)) < 15  # within 15 cents


def test_dove_match_ranks_exact_ring_first_when_data_present():
    from echomap.geo import bells_dove
    data = bells_dove.rings()
    if not data["rings"]:
        pytest.skip("Dove data not built")
    ring = next(r for r in data["rings"] if r.get("nominals") and r["bells"] == 6 and r.get("lat"))
    ana = {"bells": [{"nominal_hz": f} for f in ring["nominals"]]}
    m = bells_dove.match(ana)
    assert any(c["ring_id"] == ring["ring_id"] for c in m["candidates"][:5])
    assert data["meta"]["rings_in_dove_csv"] > 7000


def test_relations_level_crossing_hypothesis():
    from echomap.detect import relations
    recs = [{"id": "D0001", "label": "Repeating tone/beep", "category": "electronic_signals", "start": 1.0, "end": 40.0, "duration": 39.0},
            {"id": "D0002", "label": "Train", "category": "railway", "start": 12.0, "end": 30.0, "duration": 18.0}]
    r = relations.build(recs, 60.0, [-40.0] * 600, 0.1)
    assert any(h["pattern"] == "railway level-crossing sequence" for h in r["hypotheses"])


def test_text_search_maps_query_to_audioset_classes():
    from echomap.detect import textsearch
    from echomap.detect.taxonomy import class_table
    names = [c["name"] for c in class_table()]
    fs = np.zeros((20, 527), np.float32)
    fs[5:9, names.index("Church bell")] = 0.6
    sed = {"frame_scores": fs, "hop_s": 0.5}
    r = textsearch.search("a church bell ringing in the distance", sed, names)
    assert "Church bell" in [c["name"] for c in r["mapped_classes"]]
    assert r["intervals"] and abs(r["intervals"][0]["start"] - 2.5) < 0.01


def test_geo_footprints_and_unlocated():
    from echomap.geo import engine as ge
    recs = [{"id": "D0001", "label": "Spoken language: Dutch", "language_code": "nl", "language_name": "Dutch", "confidence": 0.9,
             "tier": "probable", "category": "language", "category_title": "Language detection", "evidence": {}},
            {"id": "D0002", "label": "Car", "confidence": 0.7, "tier": "probable", "category": "vehicles",
             "category_title": "Vehicles", "evidence": {}}]
    r = ge.GeoEngine("x").run(recs, {"detected": False}, {})
    assert r["footprints"] and r["footprints"][0]["record"] == "D0001"
    assert {c["iso2"] for c in r["footprints"][0]["countries"]} & {"NL", "BE", "SR"}
    assert [u["record"] for u in r["unlocated"]] == ["D0002"]
    import base64
    bits = np.unpackbits(np.frombuffer(base64.b64decode(r["footprints"][0]["bits"]), np.uint8))
    row, col = int((90 - 52.37) / 0.5), int((4.9 + 180) / 0.5)  # Amsterdam
    assert bits[row * 720 + col] == 1


def test_capability_modes():
    from echomap.capabilities import CapabilityLog
    q = CapabilityLog("quick")
    assert q.wants("sed") and not q.wants("birdnet")
    e = CapabilityLog("experimental")
    assert e.wants("discovery") and e.wants("references")
    assert any(i["id"] == "rf" and i["status"] == "requires_hardware" for i in e.to_json()["items"])


def test_fingerprint_is_deterministic_and_nonempty():
    from echomap.detect.references import fingerprint
    x = rng.normal(0, 0.1, 8000 * 5).astype(np.float32)
    a, b = fingerprint(x), fingerprint(x)
    assert a and a == b


def test_asr_enhancement_and_candidates():
    from echomap.models import asr
    x = (rng.normal(0, 0.001, 3 * 16000) + 0.003 * np.sin(2 * np.pi * 300 * np.arange(3 * 16000) / 16000)).astype(np.float32)
    y = asr.enhance(x)
    assert len(y) == len(x) and np.isfinite(y).all() and np.sqrt(np.mean(y ** 2)) > 5 * np.sqrt(np.mean(x ** 2))
    fs = np.zeros((20, 527), np.float32)
    fs[4:8, 0] = 0.08  # faint speech score still becomes a candidate
    regs = asr.speech_candidates({"frame_scores": fs, "hop_s": 0.5}, [0])
    assert regs and abs(regs[0][0] - 2.0) < 1e-6


def test_asr_transcribes_when_pack_installed():
    import os
    md = os.environ.get("ECHOMAP_MODELS")
    from echomap.models import asr
    if not md or not asr.available(md):
        pytest.skip("speech-to-text pack not installed")
    r = asr.transcribe(np.zeros(16000 * 3, np.float32), md, "en")
    assert r["quality"] in ("likely spurious", "uncertain")  # silence must never come out as a 'clear' transcript
