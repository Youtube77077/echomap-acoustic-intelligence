"""Offline picture pack (v1.0.9): lookup rules and reading, with a tiny fake pack."""
import json
import zipfile

from echomap import images


def _pack(tmp_path):
    entries = {k: {"img": f"img/{i}.webp", "title": f"{k}.jpg", "licence": "CC BY-SA 4.0", "artist": "A", "source": "s"}
               for i, k in enumerate(["sp:turdus merula", "vt:tram", "as:Car", "as:Siren", "as:Train"])}
    p = tmp_path / "EchoMap-images-v1.zip"
    with zipfile.ZipFile(p, "w") as z:
        for e in entries.values():
            z.writestr(e["img"], b"RIFFxxxxWEBP")
        z.writestr("index.json", json.dumps({"entries": entries}))
    images._open.cache_clear()
    return p


def test_picture_rules(tmp_path):
    _pack(tmp_path)
    recs = [
        {"label": "Bird vocalization", "category": "birds_wildlife", "scientific_name": "Turdus merula"},
        {"label": "Train", "category": "railway", "evidence": {"vehicle_type": {"top": "tram", "confident": True}}},
        {"label": "Train", "category": "railway", "evidence": {"vehicle_type": {"top": "tram", "confident": False}}},
        {"label": "Car", "category": "vehicles"},
        {"label": "Siren-like tonal pattern", "category": "bells_alarms"},
        {"label": "Spoken language: Dutch", "category": "language", "language_code": "nl"},
        {"label": "Mains hum 50 Hz", "category": "electronic_signals"},
    ]
    n = images.attach(recs, str(tmp_path))
    keys = [r.get("image", {}).get("key") for r in recs]
    assert keys == ["sp:turdus merula", "vt:tram", "as:Train", "as:Car", "as:Siren", None, None]
    assert n == 5 and recs[0]["image"]["licence"] == "CC BY-SA 4.0"
    assert images.read("vt:tram", str(tmp_path)).startswith(b"RIFF")
    assert images.read("as:Nothing", str(tmp_path)) is None
    images._open.cache_clear()


def test_no_pack_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "find_pack", lambda md=None: None)
    recs = [{"label": "Car", "category": "vehicles"}]
    assert images.attach(recs, str(tmp_path)) == 0 and "image" not in recs[0]
