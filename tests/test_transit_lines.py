"""Announced stops -> transit line + stop-level pin (v1.0.9), with a tiny synthetic GTFS-derived pack."""
import gzip
import json

from echomap.geo import transit_lines as T


def _pack(tmp_path, monkeypatch):
    stops = [["Centraal Station", 52.3789, 4.9001], ["Dam", 52.3731, 4.8926], ["Paleisstraat", 52.3732, 4.8898],
             ["Koningsplein", 52.3672, 4.8890], ["Leidseplein", 52.3633, 4.8830], ["Amsterdam, Leidseplein", 52.3623, 4.8809],
             ["Centraal Station", 52.0894, 5.1100]]   # a second "Centraal Station" (Utrecht): generic within NL
    ops = [["GVB", "NL", "Amsterdam"], ["Connexxion", "NL", ""]]
    lines = [[0, "tram", "2", "Lijn 2", [0, 1, 2, 3, 4]], [0, "tram", "12", "Lijn 12", [3, 4]],
             [1, "bus", "N47", "Nachtlijn", [5, 6]]]
    p = tmp_path / "EchoMap-transit-v1.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as g:
        json.dump({"stops": stops, "operators": ops, "lines": lines, "feeds": []}, g)
    monkeypatch.setattr(T, "find_pack", lambda md=None: p)
    T._load.cache_clear()


def seg(text):
    return [{"text": text, "quality": "clear", "start": 1.0}]


def test_two_stops_on_one_line(tmp_path, monkeypatch):
    _pack(tmp_path, monkeypatch)
    r = T.match(seg("we gaan langs Paleisstraat en dan Koningsplein"))
    assert r and r["lines"][0]["line"] == "Tram 2" and r["lines"][0]["operator"] == "GVB"
    assert set(r["lines"][0]["stops_heard"]) == {"Paleisstraat", "Koningsplein"} and r["confidence"] == 0.7


def test_single_stop_needs_cue_and_tram_preferred(tmp_path, monkeypatch):
    _pack(tmp_path, monkeypatch)
    assert T.match(seg("ik was gisteren bij het Leidseplein")) is None          # no cue, one stop: not enough
    r = T.match(seg("Volgende halte: Leidseplein"), prefer_modes=["tram"])
    assert r and r["heard"][0]["cue"] == "volgende halte"
    assert r["lines"][0]["mode"] == "tram" and abs(r["pins"][0]["lat"] - 52.36) < 0.01


def test_unrelated_speech_matches_nothing(tmp_path, monkeypatch):
    _pack(tmp_path, monkeypatch)
    assert T.match(seg("ik heb een broodje gegeten op het werk")) is None
    assert T.match([{"text": "Volgende halte: Leidseplein", "quality": "likely spurious"}]) is None


def test_geo_engine_pins_the_stop(tmp_path, monkeypatch):
    _pack(tmp_path, monkeypatch)
    from echomap.geo.engine import GeoEngine
    tl = T.match(seg("Volgende halte Paleisstraat, daarna Koningsplein"), prefer_modes=["tram"])
    recs = [{"id": "D0001", "label": "Spoken language: Dutch", "language_code": "nl", "language_name": "Dutch",
             "confidence": 0.95, "category": "language", "category_title": "Language", "tier": "confirmed",
             "start": 0, "end": 10, "evidence": {}}]
    g = GeoEngine("/nonexistent").run(recs, {"detected": False}, {"date": {"mode": "unknown"}, "location": {"mode": "unknown"}},
                                      {"transit_lines": tl})
    places = (g.get("likely_places") or {}).get("places") or []
    from echomap.geo.grid import haversine_km
    assert places and float(haversine_km(52.370, 4.889, places[0]["lat"], places[0]["lon"])) < 3.0
    T._load.cache_clear()
