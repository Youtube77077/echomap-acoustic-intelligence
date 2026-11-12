"""v1.0.3: point clues (tram networks, spoken place names), likely places, exact-border footprints, vector clue maps."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

import echomap.geo.engine as ge
from echomap.geo import places
from echomap.geo.topo import country_at
from echomap.report import images


def _rec(**kw):
    b = {"id": "D0001", "label": "x", "confidence": 0.9, "tier": "probable", "evidence": {}, "category": "language",
         "start": 0, "end": 10, "category_title": "x"}
    b.update(kw)
    return b


CZ_NETS = [  # small fixture (real OSM-derived values for these cities, rounded)
    {"name": "Prague", "cc": "CZ", "admin1": "Prague", "lat": 50.07, "lon": 14.44, "ways": 2624, "extent_km": 8.5},
    {"name": "Brno", "cc": "CZ", "admin1": "South Moravian", "lat": 49.2, "lon": 16.6, "ways": 1306, "extent_km": 6.5},
    {"name": "Ostrava", "cc": "CZ", "admin1": "Moravian-Silesian", "lat": 49.82, "lon": 18.24, "ways": 684, "extent_km": 6.8},
    {"name": "Dresden", "cc": "DE", "admin1": "Saxony", "lat": 51.05, "lon": 13.74, "ways": 2300, "extent_km": 7.5},
    {"name": "Melbourne", "cc": "AU", "admin1": "Victoria", "lat": -37.81, "lon": 144.96, "ways": 5200, "extent_km": 14.0},
]


@pytest.fixture
def nets(monkeypatch):
    monkeypatch.setattr(places, "tram_networks", lambda: {"meta": {}, "networks": CZ_NETS})


def test_country_at_uses_real_borders():
    assert country_at(50.08, 14.42) == "CZ"
    assert country_at(51.05, 13.74) == "DE"
    assert country_at(51.5, -0.12) == "GB"
    assert country_at(0.0, -30.0) is None


def test_tram_hypothesis_rule_and_reference():
    recs = [_rec(id="D1", label="Rail transport", category="vehicles", confidence=0.7, start=2, end=9),
            _rec(id="D2", label="Ding", category="bells", confidence=0.5, start=4, end=5)]
    th = places.tram_hypothesis(recs, None)
    assert th and th["c"] == pytest.approx(0.3) and th["record"] == "D1"
    refs = {"events": [{"record": "D1", "matches": [{"category": "tram", "tier": "strong acoustic similarity", "similarity": 0.95,
                                                      "title": "Tram in Prague.ogg"}]}]}
    assert places.tram_hypothesis(recs, refs)["c"] == pytest.approx(0.55)
    assert places.tram_hypothesis([_rec(label="Rail transport", category="vehicles", confidence=0.7)], None) is None


def test_place_mentions_cues_and_common_words():
    seg = lambda t: [{"text": t, "quality": "clear", "language": "en", "start": 0, "end": 3}]
    m = places.place_mentions(seg("Next stop is Brno, main station."))
    assert [x["name"] for x in m] == ["Brno"] and m[0]["cue"] == "next stop" and places.mention_confidence(m[0]) >= 0.55
    assert places.place_mentions(seg("Reading is fun. Best day ever in Mobile")) == []
    assert places.place_mentions([{"text": "Next stop Brno", "quality": "likely spurious", "language": "cs"}]) == []
    names = {x["name"] for x in places.place_mentions(seg("We met in Sao Paulo and Rio de Janeiro"))}
    assert names == {"Sao Paulo", "Rio de Janeiro"}


def test_prague_tram_example_ranks_czech_tram_cities(nets):
    recs = [_rec(id="D1", label="Mains hum (50 Hz electrical network)", tier="confirmed", category="electronic_signals"),
            _rec(id="D2", label="Speech", language_code="cs", language_name="Czech", confidence=0.95),
            _rec(id="D3", label="Rail transport", category="vehicles", confidence=0.7, start=2, end=9),
            _rec(id="D4", label="Ding", category="bells", confidence=0.5, start=4, end=5)]
    extra = {"tram": places.tram_hypothesis(recs, None)}
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": 50}, {}, extra)
    P = r["likely_places"]["places"]
    assert P[0]["name"] == "Prague" and P[1]["name"] == "Brno"
    assert {q["country"] for q in P[:3]} == {"Czechia"}
    assert next(q for q in P if q["name"] == "Melbourne")["share"] < 0.01
    assert abs(sum(q["share"] for q in P) - 1.0) < 1e-3
    fp = {f["label"]: f for f in r["footprints"]}
    assert fp["Language Czech (cs)"]["shape"]["kind"] == "countries" and "CZ" in fp["Language Czech (cs)"]["shape"]["iso"]
    assert fp["Tram / streetcar (hypothesis)"]["shape"]["kind"] == "points"


def test_spoken_city_name_moves_the_best_place(nets):
    recs = [_rec(id="D1", label="Mains hum (50 Hz electrical network)", tier="confirmed", category="electronic_signals"),
            _rec(id="D2", label="Speech", language_code="cs", language_name="Czech", confidence=0.95),
            _rec(id="D3", label="Rail transport", category="vehicles", confidence=0.7, start=2, end=9),
            _rec(id="D4", label="Ding", category="bells", confidence=0.5, start=4, end=5)]
    segs = [{"text": "Příští zastávka Brno, Česká.", "quality": "clear", "language": "cs", "start": 3, "end": 6}]
    extra = {"tram": places.tram_hypothesis(recs, None), "place_mentions": places.place_mentions(segs)}
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": 50}, {}, extra)
    assert r["likely_places"]["places"][0]["name"] == "Brno"


def test_vector_clue_maps_render_world_and_zoom(nets):
    recs = [_rec(id="D1", label="Mains hum (50 Hz electrical network)", tier="confirmed", category="electronic_signals"),
            _rec(id="D2", label="Speech", language_code="cs", language_name="Czech", confidence=0.95),
            _rec(id="D3", label="Rail transport", category="vehicles", confidence=0.7, start=2, end=9),
            _rec(id="D4", label="Ding", category="bells", confidence=0.5, start=4, end=5)]
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": 50}, {}, {"tram": places.tram_hypothesis(recs, None)})
    views = dict(images.clue_map_views(r, width=700))
    assert set(views) == {"world", "zoom"}
    z = Image.open(io.BytesIO(base64.b64decode(views["zoom"].split(",", 1)[1]))).convert("RGB")
    a = np.asarray(z)
    # the #1 place circle is drawn in red somewhere in the zoomed view
    red = (a[..., 0] > 200) & (a[..., 1] < 110) & (a[..., 2] < 90)
    assert red.sum() > 50
    x0, y0, x1, y1 = images.default_view(r)
    assert x0 < 14.44 < x1 and y0 < 50.07 < y1 and (x1 - x0) < 20
