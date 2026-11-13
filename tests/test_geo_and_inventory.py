"""Geographic engine, gazetteer, inventory logic, taxonomy and media ingest tests (no models needed)."""
import numpy as np
import pytest

from echomap.detect import taxonomy
from echomap.detect.inventory import Inventory, runs, tier_for
from echomap.geo import engine as ge
from echomap.geo import gazetteer
from echomap.geo.grid import grid, haversine_km


def test_grid_country_lookup():
    g = grid()
    i = np.argmin(haversine_km(52.37, 4.89, g.lat, g.lon))  # Amsterdam
    assert g.iso[i] == "NL" and g.mains[i] == "50"
    j = np.argmin(haversine_km(40.71, -74.0, g.lat, g.lon))  # New York
    assert g.iso[j] == "US" and g.mains[j] == "60"
    k = np.argmin(haversine_km(34.69, 135.5, g.lat, g.lon))  # Osaka (west Japan, 60 Hz)
    assert g.iso[k] == "JP" and g.mains[k] == "60"
    m = np.argmin(haversine_km(35.68, 139.69, g.lat, g.lon))  # Tokyo (east Japan, 50 Hz)
    assert g.mains[m] == "50"


def test_haversine():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.2)


def test_gazetteer():
    r = gazetteer.search_city("Amsterdam", "NL")
    assert r and r[0]["country"] == "NL" and abs(r[0]["lat"] - 52.37) < 0.1
    assert gazetteer.nearest_city(48.857, 2.352)["name"].startswith("Paris")


def _rec(**kw):
    base = {"id": "D0001", "label": "x", "confidence": 0.9, "tier": "probable", "evidence": {}, "category": "language"}
    base.update(kw)
    if base.get("language_code"):   # hand-specified language clue = clear speech
        base["evidence"] = {**base["evidence"], "speech_score_max": 0.9}
        base.setdefault("transcripts", [{"quality": "clear", "language": base["language_code"]}])
    return base


def test_geo_no_evidence_gives_no_estimate():
    r = ge.GeoEngine("nonexistent").run([], {"detected": False}, {})
    assert r["status"] == "no_estimate" and not r["candidates"]


def test_geo_mains_60_alone_is_not_a_reliable_estimate_but_points_to_60hz_grids():
    r = ge.GeoEngine("nonexistent").run([_rec(label="Mains hum (60 Hz electrical network)", tier="confirmed",
                                              category="electronic_signals")],
                                        {"detected": True, "nominal_hz": 60}, {})
    # 60 Hz grids cover a large share of the land area: evidence alone narrows it < 2x -> no estimate
    assert r["status"] == "no_estimate" and not r["candidates"]
    isos = {c["iso2"] for c in r["weak_candidates"][0]["countries"]}
    assert isos & {"US", "CA", "MX", "BR", "CO", "VE", "PE", "SA", "KR", "TW", "PH"}


def test_geo_mains_plus_language_gives_estimate():
    recs = [_rec(label="Mains hum (60 Hz electrical network)", tier="confirmed", category="electronic_signals"),
            _rec(id="D0002", language_code="ko", language_name="Korean", confidence=0.9)]
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": 60}, {})
    assert r["status"] == "estimate" and r["candidates"][0]["countries"][0]["iso2"] in ("KR", "KP")


def test_geo_language_plus_user_country():
    recs = [_rec(language_code="nl", language_name="Dutch", confidence=0.9)]
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": 50},
                                         {"location": {"mode": "country", "country": "BE"}})
    assert r["candidates"][0]["countries"][0]["iso2"] == "BE"


def test_geo_city_prior_gives_local_resolution():
    city = gazetteer.search_city("Utrecht", "NL")[0]
    recs = [_rec(language_code="nl", language_name="Dutch", confidence=0.8)]
    r = ge.GeoEngine("nonexistent").run(recs, {"detected": False}, {"location": {"mode": "city", "country": "NL", "city": city}})
    lc = r["local_refinement"]["candidates"][0]
    assert haversine_km(lc["centre"]["lat"], lc["centre"]["lon"], city["lat"], city["lon"]) < 15
    assert lc["uncertainty_radius_km"] <= 25


def test_low_confidence_contributes_little():
    hi = ge.GeoEngine("x").run([_rec(language_code="ja", language_name="Japanese", confidence=0.95)], {"detected": False}, {})
    lo = ge.GeoEngine("x").run([_rec(language_code="ja", language_name="Japanese", confidence=0.05)], {"detected": False}, {})
    assert hi["information_ratio"] > lo["information_ratio"] * 3


def test_weeks_from_context():
    w, _ = ge.context_weeks({"date": {"mode": "exact", "value": "2025-05-10T07:30"}})
    assert w == [18]
    w, _ = ge.context_weeks({"date": {"mode": "unknown"}})
    assert w is None


def test_tiers():
    assert tier_for(0.8, True) == "confirmed"
    assert tier_for(0.8, False) == "probable"
    assert tier_for(0.4, True) == "probable"
    assert tier_for(0.2, False) == "possible"
    assert tier_for(0.0, False, measurement=True) == "confirmed"


def test_runs_bridge_gaps():
    m = np.array([1, 1, 0, 1, 0, 0, 0, 1], bool)
    assert runs(m, gap=2) == [(0, 4), (7, 8)]


def test_taxonomy_covers_all_classes_and_categories():
    t = taxonomy.class_table()
    assert len(t) == 527
    keys = {k for k, _ in taxonomy.CATEGORIES}
    assert all(c["category"] in keys for c in t)
    names = {c["name"]: c["category"] for c in t}
    assert names["Bus"] == "public_transport" and names["Train"] == "railway" and names["Church bell"] == "bells_alarms"


def test_association_merges_parent_child_not_unrelated():
    inv = Inventory(10.0)
    raw = [
        {"name": "Vehicle", "ancestors": ["Sounds of things"], "alt_group": 0, "start": 1, "end": 4, "score": .8, "depth": 1},
        {"name": "Bus", "ancestors": ["Motor vehicle (road)", "Vehicle"], "alt_group": 0, "start": 1.5, "end": 4, "score": .6, "depth": 3},
        {"name": "Rain", "ancestors": ["Natural sounds"], "alt_group": 9, "start": 1, "end": 4, "score": .7, "depth": 2},
    ]
    groups = inv.associate(raw)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
