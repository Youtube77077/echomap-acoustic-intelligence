"""Model catalogue: what each model is, where it comes from, licence, size and resource notes."""
from __future__ import annotations

from pathlib import Path

MODELS = {
    "sed": {
        "name": "EfficientAT mn10_as (AudioSet, 527 classes)",
        "files": ["efficientat_mn10_as.onnx", "efficientat_mn10_as_melbank.npy", "audioset_labels.csv"],
        "source": "https://github.com/fschmid56/EfficientAT (weights mn10_as_mAP_471.pt, release v0.0.1); "
                  "ONNX converted by this repository's convert-models workflow (release models-v1)",
        "licence": "MIT (c) 2022 Florian Schmid",
        "published_accuracy": "mAP 0.471 on the AudioSet evaluation set (publisher's figure)",
        "size_mb": 19.8, "ram_mb": "~200", "required": True,
        "limitations": "Trained on 10 s weakly labelled clips; time resolution comes from 3 s sliding windows "
                       "(0.5 s hop). Scores are not calibrated probabilities. AudioSet labels are noisy.",
    },
    "birdnet": {
        "name": "BirdNET+ V3.0 developer preview 3.1 (~11.5K species, FP16 pruned)",
        "files": ["birdnet_v3_fp16_pruned.onnx", "birdnet_v3_labels.csv"],
        "source": "https://zenodo.org/records/20703646 (Cornell Lab of Ornithology / TU Chemnitz)",
        "licence": "CC BY-SA 4.0, with Terms of Use (no poaching or military use)",
        "published_accuracy": "None published for this preview ('may not reflect final performance')",
        "size_mb": 71.5, "ram_mb": "~400", "required": False,
        "limitations": "Developer preview. Limited handling of non-target sounds (rain, wind, engines). The species "
                       "list needs cleanup (some taxonomic classes are wrong). Call vs song is not distinguished. "
                       "Japanese and Canadian crossing signals imitate bird calls.",
    },
    "geomodel": {
        "name": "BirdNET+ Geomodel V3.0.4 (~14K species, FP16)",
        "files": ["geomodel_v3.0.4_fp16.onnx", "geomodel_v3.0.4_labels.txt"],
        "source": "https://github.com/birdnet-team/geomodel/releases/tag/v3.0.4",
        "licence": "Apache-2.0 for model artifacts (LICENSE-MODELS.md); code MIT",
        "published_accuracy": "None published",
        "size_mb": 7.9, "ram_mb": "~100", "required": False,
        "limitations": "Predicts where and when a species is expected from occurrence data (citizen-science bias "
                       "toward well-surveyed regions). Not proof of presence or absence.",
    },
    "langid": {
        "name": "Whisper base, language-identification-only graph (99 languages, int8)",
        "files": ["whisper_base_lid_int8.onnx", "whisper_base_mel_filters.npy", "whisper_base_languages.json"],
        "source": "https://huggingface.co/openai/whisper-base, converted by this repository's convert-models "
                  "workflow (release models-v1)",
        "licence": "MIT (c) OpenAI",
        "published_accuracy": "Whisper paper (Radford et al. 2022): LID accuracy varies strongly by language and "
                              "degrades on short or low-resource speech",
        "size_mb": 51.5, "ram_mb": "~300", "required": False,
        "limitations": "Cannot transcribe (by design). int8 quantisation shifts probabilities by up to ~0.18 vs "
                       "FP32 (parity test). Overlapping speakers are not separated. Related languages are confused "
                       "(e.g. Dutch/German in the CI test on synthetic speech).",
    },
}


def status(models_dir: str | Path) -> dict:
    d = Path(models_dir)
    out = {}
    for key, m in MODELS.items():
        present = [f for f in m["files"] if (d / f).exists()]
        out[key] = {**m, "installed": len(present) == len(m["files"]), "missing": [f for f in m["files"] if f not in present]}
    return out
