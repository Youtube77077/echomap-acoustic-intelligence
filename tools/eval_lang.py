"""Language-identification evaluation on FLEURS real speech (google/fleurs, CC BY 4.0). CI only.

For each language: N test utterances, scored on full utterances, 3 s excerpts, and full utterances with white noise
at 5 dB SNR. Reports top-1 accuracy per condition and the most frequent confusions.
Usage: python tools/eval_lang.py <models_dir> <out_json> [n_per_lang]
"""
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.ingest.media import decode_mono  # noqa: E402
from echomap.models import langid  # noqa: E402

LANGS = {"en_us": "en", "de_de": "de", "fr_fr": "fr", "es_419": "es", "nl_nl": "nl", "it_it": "it", "pt_br": "pt",
         "ru_ru": "ru", "pl_pl": "pl", "tr_tr": "tr", "ar_eg": "ar", "hi_in": "hi", "ja_jp": "ja", "cmn_hans_cn": "zh",
         "ko_kr": "ko", "sw_ke": "sw", "fi_fi": "fi", "sv_se": "sv", "uk_ua": "uk", "vi_vn": "vi"}
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{}/audio/test.tar.gz"


def utterances(code, n):
    r = requests.get(BASE.format(code), stream=True, timeout=300, headers={"User-Agent": "EchoMap-evaluation/1.0"})
    r.raise_for_status()
    got = 0
    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".wav"):
                data = tf.extractfile(m).read()
                p = Path("/tmp/fleurs.wav")
                p.write_bytes(data)
                x, sr, _ = decode_mono(p, 16000)
                yield x
                got += 1
                if got >= n:
                    break


def main():
    models, out = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    rng = np.random.default_rng(0)
    res, conf = {}, {}
    for code, wl in LANGS.items():
        acc = {"full": 0, "3s": 0, "noisy_5dB": 0}
        k = 0
        try:
            for x in utterances(code, n):
                k += 1
                xn = x + rng.normal(0, np.sqrt(np.mean(x ** 2) / 10 ** 0.5), len(x)).astype(np.float32)
                for cond, sig in (("full", x), ("3s", x[: 3 * 16000]), ("noisy_5dB", xn)):
                    top = langid.identify(sig, models, top=1)[0]["code"]
                    acc[cond] += int(top == wl)
                    if top != wl and cond == "full":
                        conf[f"{wl}->{top}"] = conf.get(f"{wl}->{top}", 0) + 1
        except Exception as e:  # noqa: BLE001
            res[code] = {"error": str(e)}
            continue
        res[code] = {"whisper_code": wl, "n": k, **{c: round(v / max(1, k), 3) for c, v in acc.items()}}
        print(code, res[code], flush=True)
    ok = [v for v in res.values() if "n" in v]
    overall = {c: round(sum(v[c] * v["n"] for v in ok) / max(1, sum(v["n"] for v in ok)), 3) for c in ("full", "3s", "noisy_5dB")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d"), "dataset": "google/fleurs test split (CC BY 4.0)",
                               "model": "Whisper base LID-only int8 (EchoMap)", "overall_top1": overall, "per_language": res,
                               "top_confusions": sorted(conf.items(), key=lambda kv: -kv[1])[:20], "method": __doc__.strip()},
                              indent=1), encoding="utf-8")
    print("overall", overall)


if __name__ == "__main__":
    main()
