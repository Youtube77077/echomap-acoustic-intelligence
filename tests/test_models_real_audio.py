"""Model and end-to-end tests on real labelled recordings. Run in CI, which provides
ECHOMAP_MODELS (models folder) and ECHOMAP_AUDIO (test clips from the test-audio-v1 release).

Assertions check that the *ground-truth* class appears among the model's top predictions. Results
for every clip are written to tests/_eval.json for docs/TEST_RESULTS.md, and failures are reported
truthfully rather than hidden.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

MODELS = os.environ.get("ECHOMAP_MODELS")
AUDIO = os.environ.get("ECHOMAP_AUDIO")
pytestmark = pytest.mark.skipif(not (MODELS and AUDIO), reason="models/audio not available (CI only)")
MANIFEST = json.loads((Path(__file__).parent / "test_audio_manifest.json").read_text(encoding="utf-8"))
EVAL = {}


def _clip(name):
    from echomap.ingest.media import decode_mono
    return decode_mono(Path(AUDIO) / name, 32000)[0]


@pytest.mark.parametrize("item", [c for c in MANIFEST["clips"] if "audioset_expected" in c["truth"]], ids=lambda c: c["file"])
def test_sed_topk_contains_truth(item):
    from echomap.detect.taxonomy import class_table
    from echomap.models import sed
    x = _clip(item["file"])
    r = sed.run(x, MODELS)
    labels = sed.load_labels(MODELS)
    mx = r["scores"].max(axis=0)
    order = np.argsort(-mx)
    top = [labels[i]["name"] for i in order[:10]]
    want = item["truth"]["audioset_expected"]
    tab = {c["name"]: c for c in class_table()}
    # accept the expected class or any of its descendants among the top 10
    ok = any(n == want or want in tab[n]["ancestors"] for n in top)
    EVAL[item["file"]] = {"truth": want, "top5": [(labels[i]["name"], round(float(mx[i]), 3)) for i in order[:5]], "hit_top10": ok}
    _dump()
    assert ok, f"{want} not in top-10: {top}"


def test_birdnet_soundscape_species():
    from echomap.models import birdnet
    x = _clip("soundscape.wav")
    r = birdnet.run(x, MODELS)
    L = birdnet.labels(MODELS)
    mx = r["scores"].max(axis=0)
    top = [(L[i]["common"], L[i]["class"], round(float(mx[i]), 3)) for i in np.argsort(-mx)[:10]]
    EVAL["soundscape.wav:birdnet"] = {"top10": top}
    _dump()
    assert sum(1 for _, c, s in top if c == "Aves" and s >= 0.5) >= 3


def test_birdnet_crickets_insect():
    from echomap.models import birdnet
    x = _clip("esc50_crickets.wav")
    r = birdnet.run(x, MODELS)
    L = birdnet.labels(MODELS)
    i = int(r["scores"].max(axis=0).argmax())
    EVAL["esc50_crickets.wav:birdnet"] = {"top1": (L[i]["common"], L[i]["class"])}
    _dump()
    assert L[i]["class"] == "Insecta"


def test_end_to_end_pipeline_and_report(tmp_path):
    from echomap import config
    from echomap.pipeline import Case
    s = config.Settings(models_dir=MODELS, cases_dir=str(tmp_path))
    c = Case(str(Path(AUDIO) / "esc50_church_bells.wav"), s)
    c.prepare()
    c.analyse()
    assert c.state == "analysed", c.error
    assert c.result["inventory"], "empty inventory"
    assert all(r["start"] <= r["end"] for r in c.result["inventory"])
    c.contextualise({"date": {"mode": "unknown"}, "location": {"mode": "unknown"}})
    assert c.state == "complete", c.error
    html = Path(c.result["report_html"]).read_text(encoding="utf-8")
    for section in ["SHA-256", "Complete detection inventory", "Event timeline", "Model information", "Full processing history"]:
        assert section in html
    assert (c.dir / "case.json").exists()
    EVAL["e2e:church_bells"] = {"records": [(r["label"], r["tier"], round(r["confidence"], 2)) for r in c.result["inventory"]][:30],
                                "geo": c.result["geo"].get("summary")}
    _dump()


def _dump():
    (Path(__file__).parent / "_eval.json").write_text(json.dumps(EVAL, indent=1), encoding="utf-8")
