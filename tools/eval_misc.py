"""Signal, bell and unknown-sound evaluations (CI only).

signals : reference-library crossing-signal / level-crossing / siren clips through the pipeline (Deep mode with the
          reference search disabled to avoid leakage): which detectors fire, and periodic pattern statistics.
bells   : bell clips whose title names a place. Is that place among the compatible Dove's Guide rings, and at what rank?
unknown : synthetic out-of-ontology sounds (FSK modem bursts, random FM textures) mixed into park-like noise:
          does discovery flag them, and does any model give them a high-confidence label?
Usage: python tools/eval_misc.py <models_dir> <clips_dir> <out_json>
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap import config  # noqa: E402
from echomap.ingest.media import write_wav16  # noqa: E402
from echomap.pipeline import Case  # noqa: E402


def run_case(path, models, mode, tmp):
    s = config.Settings(models_dir=models, cases_dir=str(tmp), analysis_mode=mode)
    c = Case(str(path), s)
    c.prepare()
    c.analyse(auto_geo=False)
    return c


def main():
    models, clips, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    tmp = Path("/tmp/evalcases")
    refs = json.load(gzip.open(Path(__file__).resolve().parents[1] / "echomap/data/references.json.gz", "rt"))["refs"]
    res = {"signals": [], "bells": [], "unknown": []}
    import echomap.detect.references as R
    R.library.cache_clear()
    R.DATA = Path("/nonexistent")  # disable reference search for these clips (they ARE the references)
    for r in refs:
        if r["category"] not in ("crossing signal", "level crossing", "siren", "bells"):
            continue
        f = clips / r["clip_file"]
        if not f.exists():
            continue
        try:
            c = run_case(f, models, "deep", tmp)
        except Exception as e:  # noqa: BLE001
            continue
        inv = c.result.get("inventory", [])
        labels = [d["label"] for d in inv if d["tier"] != "unclassified"]
        row = {"title": r["title"], "category": r["category"], "duration_s": r["duration_s"],
               "periodic_detected": any(("Repeating" in l or "Periodic" in l or "ticks" in l) for l in labels),
               "siren_detected": any("iren" in l for l in labels), "bell_detected": any("ell" in l for l in labels),
               "top_labels": labels[:6], "patterns": c.result.get("signal", {}).get("periodic_patterns", [])[:2]}
        if r["category"] == "bells":
            b = c.result.get("bells") or {}
            m = b.get("dove_match") or {}
            words = [w for w in re.findall(r"[A-Za-z]{4,}", r["title"]) if w.lower() not in
                     ("bells", "bell", "church", "ringing", "ring", "tower", "cathedral", "with", "from")]
            rank = None
            for i, cnd in enumerate(m.get("candidates", []) + []):
                if any(w.lower() in (cnd.get("place") or "").lower() for w in words):
                    rank = i + 1
                    break
            res["bells"].append({"title": r["title"], "bells_reconstructed": (b.get("analysis") or {}).get("ring_size_observed"),
                                 "valid_strikes": (b.get("analysis") or {}).get("valid_strikes"), "n_compatible": m.get("n_compatible"),
                                 "place_words": words, "place_rank_in_top15": rank})
        res["signals"].append(row)
        print(json.dumps(row)[:240], flush=True)
    # unknown sounds
    rng = np.random.default_rng(7)
    sr = 32000
    for k in range(6):
        t = np.arange(12 * sr) / sr
        bg = np.convolve(rng.normal(0, 0.01, len(t)), np.ones(30) / 30, mode="same").astype(np.float32)
        x = bg.copy()
        if k % 2 == 0:  # FSK modem-like bursts
            bits = rng.integers(0, 2, 400)
            f = np.repeat(np.where(bits, 2100, 1300), sr // 300)[: 4 * sr]
            sig = 0.2 * np.sin(2 * np.pi * np.cumsum(f) / sr)
        else:  # random FM texture
            fm = 3000 + 1500 * np.sin(2 * np.pi * np.cumsum(rng.normal(0, 2, 4 * sr)) / sr)
            sig = 0.2 * np.sin(2 * np.pi * np.cumsum(fm) / sr)
        x[4 * sr:8 * sr] += sig.astype(np.float32)
        p = tmp / f"unknown_{k}.wav"
        tmp.mkdir(parents=True, exist_ok=True)
        write_wav16(p, x, sr)
        c = run_case(p, models, "experimental", tmp)
        inv = c.result.get("inventory", [])
        in_win = [d for d in inv if d["start"] < 8 and d["end"] > 4]
        high = [d["label"] for d in in_win if d["tier"] in ("confirmed", "probable") and d["kind"] == "model"]
        dis = c.result.get("discovery", {}).get("events", [])
        flagged = [u for u in dis if u["start"] < 8 and u["end"] > 4]
        res["unknown"].append({"case": k, "kind": "FSK bursts" if k % 2 == 0 else "random FM texture",
                               "flagged_by_discovery": bool(flagged), "max_novelty": max([u["novelty"] for u in flagged], default=None),
                               "high_confidence_model_labels": high, "families": c.result.get("discovery", {}).get("families")})
        print(res["unknown"][-1], flush=True)
    sig = res["signals"]
    summary = {"signals_n": len(sig),
               "periodic_detected_rate_crossing": round(np.mean([s["periodic_detected"] for s in sig if s["category"] in ("crossing signal", "level crossing")] or [0]), 3),
               "siren_detected_rate": round(np.mean([s["siren_detected"] for s in sig if s["category"] == "siren"] or [0]), 3),
               "bell_detected_rate": round(np.mean([s["bell_detected"] for s in sig if s["category"] == "bells"] or [0]), 3),
               "bells_n": len(res["bells"]), "bells_place_found_in_top15": sum(1 for b in res["bells"] if b["place_rank_in_top15"]),
               "unknown_flagged_rate": round(np.mean([u["flagged_by_discovery"] for u in res["unknown"]]), 3),
               "unknown_forced_high_conf_labels": sum(1 for u in res["unknown"] if u["high_confidence_model_labels"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d"), "summary": summary, **res, "method": __doc__.strip()},
                              indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
