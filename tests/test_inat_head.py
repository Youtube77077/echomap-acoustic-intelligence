"""iNatSounds species head runtime (v1.0.9) with a tiny synthetic head file."""
import json

import numpy as np

from echomap.detect.inventory import Inventory
from echomap.models import inat_head


def _fake_head(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    d, C = 8, 3
    W = np.zeros((d, C), np.float32)
    W[0, 1] = 50.0   # first PCA direction strongly means class 1
    np.savez_compressed(tmp_path / "inat_head.npz", classes=np.array(["Gryllus campestris", "Rana temporaria", "Bubo bubo"]),
                        common=np.array(["Field cricket", "Common frog", "Eurasian eagle-owl"]),
                        groups=np.array(["Insecta", "Amphibia", "Aves"]), shipped=np.array(["eat"]), dim=np.array(d),
                        eat_mu=np.zeros(960, np.float16), eat_P=np.eye(960, d, dtype=np.float16), eat_W=W.astype(np.float16),
                        eat_b=np.zeros(C, np.float32), eat_thr=np.array(0.5),
                        metrics_json=np.array(json.dumps({"eat": {"top1": 0.42, "top5": 0.7}})))
    monkeypatch.setattr(inat_head, "DATA", tmp_path / "inat_head.npz")
    inat_head.head.cache_clear()


def test_inat_head_adds_species_record(tmp_path, monkeypatch):
    _fake_head(tmp_path, monkeypatch)
    e = np.zeros(960, np.float32)
    e[0] = 1.0
    r = inat_head.classify(e)
    assert r["top"][0]["sci"] == "Rana temporaria" and r["top"][0]["p"] > 0.9 and r["metrics"]["top1"] == 0.42
    sed = {"starts": np.array([0.0, 5.0]), "ends": np.array([5.0, 10.0]), "embeddings": np.stack([e, e])}
    inv = Inventory(10.0)
    frog = inv.add(category="birds_wildlife", label="Frog", start=1, end=4, confidence=0.5, tier="probable")
    frog["_audioset_names"] = ["Frog"]
    n = inat_head.add_records(inv, sed, None, lambda *a: {})
    assert n == 1
    sp = [x for x in inv.records if x.get("scientific_name") == "Rana temporaria"]
    assert sp and "Common frog" in sp[0]["specific"] and sp[0]["taxon_class"] == "Amphibia"
    inat_head.head.cache_clear()


def test_inat_head_corroborates_birdnet_instead_of_duplicating(tmp_path, monkeypatch):
    _fake_head(tmp_path, monkeypatch)
    e = np.zeros(960, np.float32)
    e[0] = 1.0
    sed = {"starts": np.array([0.0]), "ends": np.array([5.0]), "embeddings": np.stack([e])}
    inv = Inventory(5.0)
    inv.add(category="birds_wildlife", label="Amphibian vocalization", start=0, end=5, confidence=0.3, tier="possible",
            scientific_name="Rana temporaria", common_name="Common frog")
    assert inat_head.add_records(inv, sed, None, lambda *a: {}) == 0
    assert any("iNatSounds" in c for c in inv.records[0].get("corroborated_by", []))
    inat_head.head.cache_clear()


def test_no_head_file_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(inat_head, "DATA", tmp_path / "missing.npz")
    inat_head.head.cache_clear()
    assert inat_head.classify(np.ones(960)) is None
    assert inat_head.add_records(Inventory(1.0), {"starts": [], "ends": []}, None, lambda *a: {}) == 0
    inat_head.head.cache_clear()
