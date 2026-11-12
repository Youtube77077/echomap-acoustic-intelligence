# EchoMap Test Results

All runs below happened on GitHub Actions runners (Ubuntu 24.04, plus Windows Server 2025 for the build), never on the user's PC.
Raw logs: workflows `test`, `analyze`, `convert-models` and `build-windows`. CI uploads `tests/_eval.json` as an artifact.
No accuracy percentages are claimed beyond what is listed here. Every figure below was measured on the stated clips.

## 1. Model conversion parity (convert-models, run 43250341866)

| Check | Result |
|---|---|
| EfficientAT numpy mel vs PyTorch AugmentMelSTFT | max abs error ≤ 5.7e-5 (3 signals), identical frame counts |
| EfficientAT ONNX vs PyTorch probabilities | max abs error ≤ 4.6e-6 |
| Whisper numpy log-mel vs HF WhisperFeatureExtractor | max abs error ≤ 3.1e-5 |
| Whisper LID-only graph vs full model (FP32) | max abs error ≤ 1.5e-6 |
| Whisper LID int8 vs full FP32 | top-1 identical on all 5 test utterances; probabilities shift by up to 0.18 |

**Known failure (model, not conversion):** on espeak-ng synthetic speech, Whisper-base itself (FP32 and int8 alike) labels French as **Thai** and Dutch as **German**. Robotic TTS is not representative of real speech, but the confusion shows the model's limits on short or unnatural speech.

## 2. Unit and synthetic-signal tests (`tests/test_dsp_synthetic.py`, `tests/test_geo_and_inventory.py`)

All passed on the latest run, including:
- mains hum 50 and 60 Hz detection, and no detection in pure noise;
- a 1 Hz 880 Hz locator-tone pattern recovered (period within ±0.03 s);
- click trains;
- two-tone and wail siren classification, plus the siren scanner;
- engine f0 acceleration;
- RT60 of a synthetic 0.8 s decay (within ±0.25 s);
- AVAS co-drifting tones detected, while static tones, sirens and harmonic partials are rejected;
- geo engine:
  - 60 Hz evidence favours 60 Hz countries with a radius > 100 km;
  - a city prior gives a local candidate within 15 km at ≤ 25 km radius;
  - low-confidence evidence carries little weight;
  - Tokyo/Osaka mains split;
- taxonomy covers all 527 classes;
- association merges parent and child labels but not unrelated sounds.

Three earlier failures were bugs in the tests themselves (an array overrun, an off-by-one frame count, and GeoNames "Paris 04" districts). They were fixed, and district entries (PPLX) were removed from the gazetteer.

## 3. Real labelled recordings: AudioSet tagger (ESC-50 fold-1 clips, 5 s each)

Criterion: the ESC-50 class (mapped to AudioSet) or a descendant appears in the top-10 predictions. Result: **15/15 hits**. Many are hits only at low scores, so whether each one is *listed* (score ≥ 0.15) matters more:

| Clip (truth) | Top prediction(s) | Listed as event? |
|---|---|---|
| chirping birds | Bird 0.33, Chirp/tweet 0.22 | yes |
| thunderstorm | Thunder 0.25 | yes |
| airplane | Aircraft 0.37, Fixed-wing aircraft 0.34 | yes |
| train | Train 0.70, Rail transport 0.68, Clickety-clack 0.42 | yes |
| church bells | Church bell 0.52, Bell 0.44 | yes |
| rain | Rain 0.58, Rain on surface 0.49 | yes |
| car horn | Vehicle horn 0.19 | yes (possible) |
| clock tick | Clock 0.30 | yes (possible) |
| sea waves | Wind 0.24, Ocean 0.22, Waves/surf 0.21 | yes (possible) |
| engine | Vehicle 0.30, Truck 0.06 | Vehicle only |
| footsteps | Air horn 0.21 (wrong), Walk/footsteps 0.14 | **no**: steps appear as *unclassified* with a "Walk, footsteps 0.14" hint |
| helicopter | Speech 0.23 (wrong), Helicopter 0.06 | **no** |
| siren | Speech 0.24 (wrong), Siren 0.03 | **no** from AudioSet. The DSP siren scanner detects it (yelp-like, 0.55) |
| wind | nothing above 0.05 | **no** |
| crickets | Insect 0.09 | no from AudioSet; BirdNET finds *Tettigonia viridissima* 0.85 |

## 4. Real recordings: BirdNET+ V3 species

| Clip | Result |
|---|---|
| BirdNET example soundscape (North American) | Dark-eyed Junco 0.96, Black-capped Chickadee 0.92, Blue Jay 0.92, House Finch 0.91, Northern Cardinal 0.86, American Goldfinch 0.83 (all North American species) |
| ESC-50 crickets | *Tettigonia viridissima* (great green bush-cricket) 0.85, correct taxon class |
| ESC-50 chirping birds | Great-tailed Grackle 0.91 (species truth unknown for this clip) |

**Known false positives at low score:**
- 0.19 "Humpback Whale" on church bells;
- 0.18–0.25 finches/sparrows on a helicopter clip;
- 0.31 "Red Tanager" on rain.

These are listed as *possible* but are **excluded from geolocation**: species need ≥ 0.40, and uncorroborated species get half weight. That rule was added after the bells clip was wrongly placed near Antarctica.

## 5. Geolocation

| Clip | Evidence | Result |
|---|---|---|
| BirdNET soundscape (North American birds) | 6+ North American species | Best candidates US/Canada (centres in the northern US / southern Canada), radius 900–1000 km, labelled "multi-country / continental" |
| ESC-50 chirping birds | Great-tailed Grackle 0.91 | Mexico / southern US region, radius 800 km |
| Clips without geographic evidence (train, rain, engine, …) | none | "No reliable geographic estimate" |
| Sea waves (weak coastal evidence only) | waves 0.21 | "No reliable estimate": land area narrowed only 1.00x (threshold 2x) |

**Fixed failure:** before the mains-hum statistics were corrected, short noisy clips produced false "50/60 Hz mains hum" detections. One rain clip was "confirmed" as 60 Hz hum, which skewed geography. Mains detection now needs ≥ 5 Welch segments, ≥ 3 harmonics at ≥ 8 dB, and ≥ 10 s for "confirmed". After the fix, none of the 15 ESC-50 clips reports hum.

**Not yet done:** a quantitative geolocation evaluation on recordings with known coordinates. A proposal is pending approval in the capability registry.

## 6. Performance (GitHub Ubuntu runner, 4 vCPU)
- 5 s clip: analysis ≈ 1.0 s.
- 120 s soundscape: analysis 13.5 s, geography 8.8 s (was 141 s before the geomodel lattice and caching fixes).
- An O(n³) click-train scan that made long recordings hang was found and fixed. A regression test now enforces linear scaling.

## 7. Windows build
See the `build-windows` workflow run. Status is recorded in the release notes once complete.
