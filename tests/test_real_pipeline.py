"""End-to-end assertions on the real pipeline with real audio (CI only; see .github/workflows/realaudio.yml).

tools/profile_case.py writes the case summary to ECHOMAP_PROFILE_JSON; this test checks that the analysis finished in
time and that the results a user needs actually come out: transcript, language, labelled sounds and a location.
"""
import json
import os
from pathlib import Path

import pytest

P = os.environ.get("ECHOMAP_PROFILE_JSON")
pytestmark = pytest.mark.skipif(not P or not Path(P).exists(), reason="real-audio profile not available (CI only)")

# wall-clock limits on GitHub's 4-core runner, as a multiple of the recording length (plus model loading)
LIMIT = {"quick": 1.0, "standard": 2.0, "deep": 3.0}


@pytest.fixture(scope="module")
def d():
    return json.loads(Path(P).read_text(encoding="utf-8"))


def test_completed_in_time(d):
    assert d["state"] == "complete", d["error"]
    assert d["total_s"] <= LIMIT[d["mode"]] * d["audio_s"] + 30, f"{d['total_s']:.0f} s for {d['audio_s']:.0f} s of audio"


def test_findings_present(d):
    f = d["findings"]
    assert f and f["where"]["headline"]
    assert f["sounds"], "no sound described"


def test_no_spurious_speech_label(d):
    assert "Speech synthesizer" not in d["speech_label_records"]


@pytest.mark.skipif(os.environ.get("ECHOMAP_FILE") not in ("dutch_speech.wav", "street_dutch.wav", "street_dutch.mp4", "long_mix.wav"),
                    reason="no speech in this file")
def test_dutch_transcribed_and_identified(d):
    assert d["clear_transcripts"] >= 1, "no clear transcript"
    assert any(w in d["transcript_text"].lower() for w in ("organisaties", "kunnen", "een")), d["transcript_text"][:200]
    assert any(code == "nl" for _, code, _ in d["languages"]), d["languages"]


@pytest.mark.skipif(os.environ.get("ECHOMAP_FILE") not in ("dutch_speech.wav", "street_dutch.wav", "street_dutch.mp4"),
                    reason="speech-only location check")
def test_location_netherlands_or_belgium(d):
    assert d["geo_status"] == "estimate", d["geo_summary"]
    top = " ".join(d["geo_top"][:2])
    assert "Netherlands" in top or "Belgium" in top, d["geo_top"]


@pytest.mark.skipif(os.environ.get("ECHOMAP_FILE") not in ("street_dutch.wav", "street_dutch.mp4"), reason="street clip only")
def test_vehicle_type_described(d):
    assert d["vehicle_typed"] >= 1
