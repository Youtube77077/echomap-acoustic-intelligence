"""Worked examples for the documentation and UI screenshots (CI): synthetic *clue sets* run through the real engine.

Usage: python tools/example_geo.py <out_dir>
Writes <out_dir>/examples.json (geo results) and <out_dir>/example-<name>-{world,zoom}.png (report renderer).
The clues are hand-specified (no audio): they show what the engine concludes IF these sounds were detected.
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import echomap.geo.engine as ge  # noqa: E402
from echomap.geo import places  # noqa: E402
from echomap.report import images  # noqa: E402


def rec(**kw):
    b = {"id": "D0001", "label": "x", "confidence": 0.9, "tier": "probable", "evidence": {}, "category": "language",
         "start": 0, "end": 10, "category_title": "x"}
    b.update(kw)
    return b


HUM = lambda hz: rec(id="D1", label=f"Mains hum ({hz} Hz electrical network)", tier="confirmed", category="electronic_signals")
LANG = lambda code, name: rec(id="D2", label="Speech", language_code=code, language_name=name, confidence=0.95)
TRAM = [rec(id="D3", label="Rail transport", category="vehicles", confidence=0.7, start=2, end=9),
        rec(id="D4", label="Ding", category="bells", confidence=0.5, start=4, end=5)]

EXAMPLES = {
    "prague-tram": ("Tram bell + rail sound, Czech speech, 50 Hz hum", [HUM(50), LANG("cs", "Czech")] + TRAM, 50, []),
    "tram-announcement": ("Same, plus the announcement 'Příští zastávka Brno'", [HUM(50), LANG("cs", "Czech")] + TRAM, 50,
                          [{"text": "Příští zastávka Brno, Česká.", "quality": "clear", "language": "cs", "start": 3, "end": 6}]),
    "melbourne-tram": ("Tram bell + rail sound, English speech, 50 Hz hum", [HUM(50), LANG("en", "English")] + TRAM, 50, []),
    "spanish-60hz": ("Spanish speech and 60 Hz hum (no tram)", [HUM(60), LANG("es", "Spanish")], 60, []),
}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (title, recs, hz, segs) in EXAMPLES.items():
        extra = {"tram": places.tram_hypothesis(recs, None), "place_mentions": places.place_mentions(segs)}
        g = ge.GeoEngine("nonexistent").run(recs, {"detected": True, "nominal_hz": hz}, {}, extra)
        g["example_title"] = title
        res[name] = g
        for view, uri in images.clue_map_views(g):
            (out / f"example-{name}-{view}.png").write_bytes(base64.b64decode(uri.split(",", 1)[1]))
        top = [(q["name"], q["country"], f"{q['share'] * 100:.1f}%") for q in (g.get("likely_places") or {}).get("places", [])[:5]]
        print(name, "|", g.get("summary", "")[:110], "|", top)
    (out / "examples.json").write_text(json.dumps(res, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
