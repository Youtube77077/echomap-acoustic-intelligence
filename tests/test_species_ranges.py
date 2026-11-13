"""GBIF-based ranges and season check for species outside the BirdNET+ geomodel (v1.0.9)."""
from echomap.geo import engine as E


def test_gbif_range_and_season(monkeypatch):
    rng = {"rana temporaria": {"k": 1, "n": 5000, "c": {"NL": 900, "DE": 2000, "GB": 1500, "US": 1},
                               "m": [10, 20, 300, 900, 900, 700, 500, 400, 200, 50, 10, 10]}}
    monkeypatch.setattr(E, "species_ranges", lambda: rng)
    recs = [{"id": "D0001", "label": "Amphibian vocalization", "category": "birds_wildlife", "category_title": "Birds",
             "tier": "probable", "confidence": 0.9, "scientific_name": "Rana temporaria", "common_name": "Common frog",
             "corroborated_by": ["D0002 Frog"], "start": 0, "end": 5, "evidence": {}}]
    g = E.GeoEngine("/nonexistent").run(recs, {"detected": False},
                                        {"date": {"mode": "exact", "value": "2025-01-10"}, "location": {"mode": "unknown"}}, {})
    used = [it for it in g["evidence_items"] if it["group"] == "biogeography"]
    assert used and used[0]["source"].startswith("GBIF") and used[0]["used"]
    assert "unusual" in recs[0]["season_check"]          # 10 of 4000 records in January
    c = (g.get("candidates") or g.get("weak_candidates"))[0]
    assert c["centre"]["lat"] > 45 and -12 < c["centre"]["lon"] < 20   # NW Europe, not the US
