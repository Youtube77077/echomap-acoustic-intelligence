# Model Registry

Last verification: 2026-11-12. Machine-readable version: `echomap/models/registry.py`.
All models run locally through ONNX Runtime 1.31 on CPU, are loaded on demand, and are released after each case.

| Model | Version | Provider / repo | Licence | Architecture | Covers | Input | Output | Size | RAM (est.) | Speed (4-vCPU runner) | Known limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EfficientAT mn10_as | weights v0.0.1 (mAP .471) | Florian Schmid, github.com/fschmid56/EfficientAT | MIT | MobileNetV3, 4.88 M params, distilled from PaSST | 527 AudioSet classes, 960-d embedding | 32 kHz mono, 128-mel (numpy frontend, parity-tested) | sigmoid scores per 5 s window, 0.5 s hop | 19.5 MB | ~200 MB | ~1 s for 30 s of audio | Saturates on windows ≤ 3 s (EchoMap uses 5 s); weak 10 s labels; scores not calibrated |
| BirdNET+ V3.0-preview3.1 (FP16 pruned) | preview 3.1, June 2026 | Cornell Lab / TU Chemnitz, zenodo 20703646 | CC BY-SA 4.0 + Terms of Use | CNN (official ONNX) | ~11,560 labels: birds, insects, amphibians, mammals, reptiles | 32 kHz, 3 s chunks | probabilities + 1280-d embeddings | 71.5 MB | ~400 MB | ~8 s for 120 s | Developer preview, no published accuracy; label list needs cleanup; weak on non-target noise |
| BirdNET+ Geomodel V3.0.4 (FP16) | 3.0.4 | birdnet-team/geomodel | Apache-2.0 (model artifacts) | MLP with circular encodings | 14,082 species occurrence by (lat, lon, week) | lat, lon, week 1-48 | occurrence probabilities | 7.9 MB | ~100 MB | 12 weekly passes over ~4k points in a few seconds | Citizen-science sampling bias; not proof of presence or absence |
| Whisper base, LID-only graph (int8) | openai/whisper-base | OpenAI (converted by convert-models.yml) | MIT | encoder + 1 decoder step + language rows | 99 languages | 16 kHz log-mel, 30 s | language softmax | 51.5 MB | ~300 MB | ~0.3 s per 30 s region | Related-language confusion; short or noisy speech; int8 shifts probabilities by up to 0.18 |
| Whisper base, speech-to-text (int8) | openai/whisper-base | OpenAI (converted by convert-asr.yml) | MIT | encoder + no-cache decoder, greedy | transcription in 99 languages | 16 kHz log-mel, ≤ 28 s chunks | text + no-speech prob + avg logprob | ~75 MB | ~400 MB | depends on amount of speech | Hallucination on noise or faint audio (quality labels); base model is weaker than larger Whisper models |

## Evaluated but not bundled
| Model | Why not (yet) |
|---|---|
| TinyCLAP (PhiNet student, 17.8 MB, Apache-2.0; verified on HF fpaissan/tinyCLAP) | Text embeddings need the MS CLAP 2022 teacher (2.3 GB; licence stated as MIT / CC-BY-3.0-US / MS-PL in different places). Conversion pipeline planned. EchoMap offers AudioSet-label text search meanwhile. |
| PANNs CNN14 | 327 MB, lower mAP (.431) than mn10 |
| AST / BEATs / HTS-AT | ~90 M params (350 MB+); gains ~1-2 mAP points |
| CED-tiny/mini | Strong (mAP .481-.490) but no maintained ONNX export path |
| Demucs / AudioSep / SAM Audio / Open-Unmix | 100 MB to several GB, GPU-oriented; separation artefacts. Band-limited listening offered instead |
| SpeechBrain ECAPA VoxLingua107 | Better LID candidate (Apache-2.0, ~85 MB); needs a PyTorch export with a custom frontend |
| BirdNET-Analyzer V2.4 | TFLite only (no maintained Windows runtime); CC BY-NC-SA |
| Google Perch v2 | TensorFlow only |
