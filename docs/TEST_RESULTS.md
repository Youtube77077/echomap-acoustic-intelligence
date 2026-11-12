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

## 7. Windows build and built-exe tests (build-windows run 43268562053, Windows Server 2025)
- The full test suite passed on Windows before packaging.
- PyInstaller one-folder build: **app ZIP 75.3 MB** (no models); **models ZIP 117.2 MB**.
- Smoke test of `EchoMap.exe`:
  - the API answers with the token and rejects requests without it;
  - the UI HTML is served.
- Real analysis through the built exe's API (soundscape.wav):
  - 57 inventory records;
  - geo best candidate US/Canada (radius 1000 km);
  - spectrogram endpoint returned 800x300 pixels (60,340 non-zero);
  - report HTML 446 KB containing all sections.
- Headless Edge screenshots (artifact `ui-screenshots`): analysis tab (waveform, timeline) and inventory tab (all categories, tiers, confidence bars) render correctly.
- **Not yet verified visually:** the spectrogram and globe screenshots were captured before their asynchronous data arrived. The spectrogram endpoint itself is verified by the API check above, but interactive rendering inside the native WebView2 window has not been seen in CI.

## 8. Open test gaps (planned)
- Video input (MP4/MOV/MKV/WEBM) extraction test file.
- Real-speech language test set (e.g. FLEURS, CC BY 4.0, streamed in CI). Needs approval.
- Geolocation evaluation on recordings with known coordinates. Needs approval.
- Real crossing-signal, EV pass-by, and change-ringing recordings. Need approval.


## 9. Cloud evaluations (v1.0.2)

### Language identification: FLEURS real speech (`docs/eval/language.json`)
20 languages, 15 test utterances each, Whisper-base LID-only int8:

| Condition | Top-1 accuracy |
|---|---|
| Full utterances | **98.0%** |
| First 3 s only | **70.0%** |
| Full utterance + white noise at 5 dB SNR | **76.7%** |

- **Full utterances:** 18 of 20 languages at 100%. Swedish was 93% and Swahili 67% (confused with Yoruba, Malay, Japanese and Indonesian).
- **Short or noisy clips:** Swahili, Swedish, Ukrainian and Finnish degrade most.
- **Caveat:** FLEURS is read speech, cleaner than field audio.

### Geolocation: iNaturalist bird recordings with known coordinates (`docs/eval/geo.json`)
47 research-grade recordings from 8 world regions (CC0 / CC BY / CC BY-SA), Standard mode:

| Metric | Date unknown | Date known |
|---|---|---|
| Recordings given an estimate (others: "no reliable estimate") | 45% | 47% |
| Median error of best candidate centre | 1,004 km | 986 km |
| Interquartile range of the error | 705 – 1,689 km | 494 – 2,746 km |
| Truth inside best circle (raw radii, before calibration) | 43% | 41% |
| Truth inside any candidate circle | 71% | 68% |
| Truth inside the "most clues agree" zone | 73% | 71% |

- **Finding:** the raw radii were overconfident. A calibration factor of 1.75 (the 68th percentile of error/radius) is now applied so the best circle is about a 68% region on this data. The sample is small (43 estimates), so treat the factor as provisional.
- **Leakage caveat:** the BirdNET geomodel is trained on citizen-science occurrences that may include these observations.
- **What this means:** single-species bird recordings typically localise to a region of about 1,000 km. Combining species, language, mains frequency and bells narrows that. A city prior gives local precision.
