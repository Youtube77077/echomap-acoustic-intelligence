# EchoMap: Technology Research (Phase 1)

Research date: 2026-11-12. Covers what each technology can actually do, what it cannot, and why it was or was not selected.
Accuracy figures below come from the publishers' own evaluations unless stated otherwise. EchoMap's own test results are in `docs/TEST_RESULTS.md`, and no accuracy figure in this repository is invented.

---

## 0. Constraints that drove every decision

| Constraint | Value measured on the target PC | Consequence |
|---|---|---|
| Free disk on C: | ~3 GB | No PyTorch or TensorFlow on the PC. Models run as ONNX through the already-installed ONNX Runtime, and any conversion runs in GitHub Actions. |
| Python | 3.14.8 (python.org), with numpy 2.5, scipy 1.18, onnxruntime 1.31.1, Pillow, PyMuPDF, shapely, requests and PyInstaller already installed | Reuse them where possible. Only add packages with a clear need. |
| CPU / RAM / GPU | Ryzen 5 7535HS / 15 GB / RTX 3050 Laptop 4 GB | CPU inference is enough for 5-minute recordings, so no CUDA stack (~2 GB) is needed. |
| Web runtime | Microsoft Edge WebView2 157 is installed, as is .NET Framework 4.8.1 | An HTML UI inside a native window via pywebview, without bundling Chromium. |
| Max recording length | 5 minutes | All analysis can run in memory. A 5-min mono 32 kHz signal is 38 MB as float32. |

---

## 1. General sound event / acoustic scene classification

The goal is to recognise as many sound types as possible, with timestamps.

| Candidate | What it is | Capabilities (publisher figures) | Limitations | License | Size | Verdict |
|---|---|---|---|---|---|---|
| **EfficientAT `mn10_as`** (Schmid, Koutini, Widmer, JKU Linz). [Repo](https://github.com/fschmid56/EfficientAT), [paper](https://arxiv.org/abs/2211.04772) | MobileNetV3 distilled from PaSST transformer ensembles on AudioSet | 527 AudioSet classes, **mAP 47.1** on the AudioSet eval set, 4.88 M params, 0.54 GMACs per 10 s clip | Trained on 10 s weakly-labelled clips, so time resolution comes only from sliding windows (EchoMap uses ~0.5 s). AudioSet labels are noisy, and some classes (for example "Electric vehicle" or brand names) do not exist. | MIT (code and weights) | ~20 MB as FP32 ONNX | **Selected (default)**. Best accuracy-per-MB found that has a clean licence. |
| EfficientAT `mn40_as` / `dymn20_as` | Larger variants | mAP 48.4 / 49.1 | 68 M / 40 M params, roughly 270 / 160 MB | MIT | - | Optional "high accuracy" pack in future, not the default. |
| PANNs CNN14 (Kong et al. 2020) | CNN on AudioSet | mAP 43.1, plus a frame-wise SED variant | 80 M params, 327 MB, and less accurate than mn10 | MIT code, CC BY 4.0 weights | 327 MB | Rejected: 16x larger and less accurate. |
| YAMNet (Google) | MobileNetV1 on AudioSet | 521 classes, ~3.7 M params | Google does not publish an AudioSet mAP figure. It is a TF Hub model, so ONNX needs tf2onnx, and it is generally reported as weaker than PANNs. | Apache-2.0 | 15 MB | Rejected in favour of the stronger mn10. |
| AST (Gong et al. 2021) | ViT on AudioSet | mAP 48.5 | 87 M params, slow on CPU | BSD-3 | ~350 MB | Rejected (size). |
| BEATs (Microsoft 2022) | Self-supervised audio transformer | mAP ~48-50 | ~90 M params, needs PyTorch | MIT | ~360 MB | Rejected (size). |
| CED-tiny/mini (Xiaomi 2023). [Paper](https://arxiv.org/abs/2308.11957) | Distilled ViT | mAP 48.1 (5.5 M) / 49.0 (9.6 M) | Needs a third-party `ced_model` package, and no ONNX export exists | Apache-2.0 | 22-40 MB | Strong alternative. Kept as a future option, because export tooling is less mature than for the plain CNN. |
| CLAP (LAION / Microsoft) | Audio-text contrastive model | Zero-shot labels from free text | Zero-shot accuracy is well below supervised taggers, and it is large (~600 MB) | CC0 / MIT | 600 MB+ | Rejected for the default. It would help with "unknown" sounds, so it is noted for future work. |

**AudioSet ontology** ([ontology.json](https://github.com/audioset/ontology), CC BY-SA 4.0): the 632-node hierarchy (for example *Vehicle → Motor vehicle (road) → Bus*). EchoMap uses it to group detections into categories and to de-duplicate parent/child labels, such as "Vehicle" plus "Bus" on the same event.

What AudioSet covers that matters here: bird vocalization, chirp, crow, pigeon, owl, insects, frogs, dog; car, car passing by, bus, truck, motorcycle, train, rail transport, railroad car, subway/metro, train horn/whistle, train wheels squealing, aircraft, helicopter, jet, boat, ship, bicycle, skateboard; engine, light/medium/heavy engine, idling, accelerating/revving, engine starting; siren, police car, ambulance, fire engine, civil defense siren, reversing beeps, air brake, air horn; church bell, change ringing, bicycle bell, chime, clock, tick-tock, alarm, buzzer, beep; wind, wind noise (microphone), rustling leaves, rain, raindrop, thunder, waves/surf, stream, waterfall, drip, fire; footsteps, door, crowd, chatter, applause; music and instrument classes; *Inside small room*, *Inside large room or hall*, *Inside public space*, *Outside urban or manmade*, *Outside rural or natural*; mains hum, static, sine wave.

What AudioSet does **not** cover, so EchoMap must say "not supported" instead of guessing: vehicle manufacturer, model or generation; diesel vs petrol; electric buses or EVs as a class; specific bell towers; species of birds (that is BirdNET's job).

---

## 2. Bioacoustics: birds and other wildlife

| Candidate | Facts | Verdict |
|---|---|---|
| **BirdNET+ V3.0 developer preview 3.1** (K. Lisa Yang Center, Cornell Lab of Ornithology and TU Chemnitz). [Repo](https://github.com/birdnet-team/birdnet-V3.0-dev), [Zenodo 10.5281/zenodo.20703646](https://zenodo.org/records/20703646) | About 11K classes: birds plus insects, amphibians, mammals and reptiles. **Official ONNX files**: FP16-pruned 71.5 MB (the file the official web demo uses), FP16 271.6 MB, FP32 541.6 MB. Input is raw mono waveform at 32 kHz in 3 s chunks (96,000 samples, batch [N,T] float32). Output is per-class scores, plus embeddings when available. Labels CSV columns: `idx;id;sci_name;com_name;class;order`. License: **CC BY-SA 4.0** for models and MIT for code, with Terms of Use prohibiting poaching and military use. | **Selected**. It is the only official BirdNET release that runs on Windows through ONNX Runtime without TensorFlow. Needs ONNX Runtime API ≥ 29, and 1.31.1 is installed. |
| Its limitations, stated by the authors and carried into the app | "Developer preview": models, labels and code will change, and it was "trained on a subset of data and may not reflect final performance". No published accuracy figure. Limited handling of non-target sounds (rain, wind, engines) and no noise class. The species list "needs cleanup", and some taxonomic class columns are wrong (for example a plover labelled Insecta). It does not separate call from song. | EchoMap shows every BirdNET result as **"possible species"**, never "confirmed", and reports "call/song: not supported by model". |
| BirdNET-Analyzer V2.4 (6,522 species) | TFLite model with a CC BY-NC-SA licence. TFLite has no maintained Windows Python runtime (tflite-runtime and ai-edge-litert are Linux/macOS only), and full TensorFlow is ~2 GB. | Rejected on Windows: storage and licence. |
| **BirdNET Geomodel V3** ([repo](https://github.com/birdnet-team/geomodel)) | Spatio-temporal species range model. Input `[lat, lon, week(1-48)]`, output occurrence scores for ~12K species. FP16 ONNX is ~7.5 MB. The repo states MIT for code and Apache-2.0 for model artifacts, and a HF mirror states CC BY-SA 4.0, so the licence is inconsistent and recorded as such. | **Selected** for biogeographic evidence, offline. Note: this is the same data family as eBird/GBIF, so GBIF occurrence counts are **not** independent evidence and EchoMap never adds both. |
| Google Perch v2 | Bird/bioacoustics embedding plus classifier, Apache-2.0, TF SavedModel | Needs TensorFlow. Future candidate if an ONNX version appears. |
| GBIF API ([api.gbif.org](https://techdocs.gbif.org/en/openapi/)) | Free, no key, occurrence data under CC0 / CC BY. Country facets per species. | **Optional online** display of where a species has been recorded. Not fused into scores, because it is correlated with the geomodel. |
| eBird API | Requires an API key and eBird's terms restrict redistribution | Not used by default. Can be added with the user's own key. |

**Geographic caveat implemented in the engine**: a species can be native, migrating, seasonal, captive (zoo or aviary), escaped, transported or vagrant. The geomodel occurrence is floored at a non-zero "out-of-range" likelihood, so a single non-native species can never eliminate a region (see `docs/GEO_ENGINE.md`).

---

## 3. Language identification

| Candidate | Facts | Verdict |
|---|---|---|
| **OpenAI Whisper `base`**, language-ID head only. [Repo](https://github.com/openai/whisper), [paper](https://arxiv.org/abs/2212.04356) | MIT. 99 languages. Language ID comes from the decoder's first-token distribution over language tokens. The paper reports LID accuracy that varies strongly by language and drops sharply for low-resource languages and short clips. | **Selected**. EchoMap exports only *encoder + one decoder step + the 99 language-token rows* to ONNX (int8, ~50 MB), so **the app physically cannot transcribe**. That matches the privacy requirement. |
| SpeechBrain ECAPA VoxLingua107 | Apache-2.0, 107 languages, reported ~93% on the VoxLingua107 dev set | Good accuracy, but needs PyTorch for export with a custom frontend. Future candidate. |
| Meta MMS-LID | 4,000+ languages | CC BY-NC and 1 GB+. Rejected. |

Speech regions come from the AudioSet *Speech* family of scores. No speaker identification is done or attempted.

---

## 4. Signal processing and audio I/O

| Candidate | Facts | Verdict |
|---|---|---|
| **numpy / scipy** (already installed) | STFT, mel and log-frequency spectrograms, MFCC (DCT), spectral features, onset detection, autocorrelation, filtering | **Selected**. All signal processing in EchoMap is original code on numpy, so it is auditable. |
| librosa | ISC licence, excellent, but pulls in numba and llvmlite (~150 MB) | Rejected (storage). EchoMap re-implements the needed parts. |
| Essentia | AGPL-3.0, no Windows Python wheels | Rejected. |
| **PyAV 19.0.2** (`av`). [Repo](https://github.com/PyAV-Org/PyAV) | BSD-3 bindings with **statically bundled FFmpeg 8.x**. The bundled FFmpeg build is reported as GPL in upstream wheels, which is fine for private use but matters if EchoMap is ever distributed publicly. Wheel 28 MB, cp312-abi3 (works on 3.14). Decodes every required format (MP3, WAV, FLAC, OGG, M4A, AAC, MP4, MOV, MKV, WEBM), reads container and stream metadata, can **stream-copy** an audio track without re-encoding, and resamples via libswresample. No known vulnerabilities on OSV for 19.0.2. | **Selected**. One package replaces ffmpeg.exe plus ffprobe.exe (~180 MB). |
| FFmpeg CLI static builds (gyan.dev / BtbN) | GPL, ~90 MB per executable | Rejected (size, and an extra executable). |

Forensic techniques researched:
- **Electric Network Frequency (ENF)**: mains hum at 50 or 60 Hz picked up by recorders. The nominal value separates 50 Hz grids from 60 Hz grids, a coarse but independent geographic clue. Matching the fine ENF fluctuation against grid reference data can verify time and grid, but needs grid reference databases that are not openly available ([Grigoras 2005](https://journal.equinoxpub.com/IJSLL/article/view/9977); [Hajj-Ahmad et al., UMD ICASSP 2013](https://www.mast.umd.edu/files/paper_ENF/conf_paper6_ICASSP13.pdf); [JRC 2024 on noise limits](https://publications.jrc.ec.europa.eu/repository/handle/JRC138933)). EchoMap measures nominal 50/60 Hz presence and records the ENF trace, but does **not** claim time verification. Per-country nominal frequency comes from the [Wikipedia "Mains electricity by country"](https://en.wikipedia.org/wiki/Mains_electricity_by_country) table, cross-checked with IEC listings. Japan is split (50 Hz east, 60 Hz west).
- **Reverberation (blind RT60)**: estimated from free decays after impulsive events (Schroeder-style slope fit). This is a coarse indicator (small room, hall, outdoor) and not a calibrated room-acoustic measurement.
- **SNR**: percentile noise-floor estimate. Reported as an estimate, not a measurement against a known reference.
- **Wind**: microphone wind noise appears as intermittent, broadband energy below ~200 Hz. EchoMap reports relative gust intensity over time, but **cannot** infer wind speed in m/s or temperature from ordinary audio, and says so.

---

## 5. Geography and geolocation

| Item | Facts | Verdict |
|---|---|---|
| Acoustic geolocation research | [Chasmai et al. 2025, "Audio Geolocation: A Natural Sounds Benchmark"](https://arxiv.org/abs/2505.18726) shows species vocalisations plus species range models are the most useful natural-sound cue. [AGL1K, 2026](https://arxiv.org/abs/2601.03227) finds language is often the dominant cue for audio-language models. | This confirms EchoMap's design: species ranges, language, and infrastructure as independent evidence groups. |
| **CLDR territoryInfo** (Unicode, [UTS #35](https://unicode.org/reports/tr35/tr35-info.html)) | Per-territory `languagePopulation` with `populationPercent` and `officialStatus`. Unicode License v3 (permissive). | **Selected** for the language → country likelihood. |
| **Natural Earth** via [world-atlas](https://github.com/topojson/world-atlas) | Public-domain country boundaries, 50m TopoJSON ~750 KB. ISC wrapper. | **Selected** for the globe, the land mask and point-in-country lookup. |
| **GeoNames cities15000** | CC BY 4.0, ~33K cities with population ≥ 15,000 | **Selected** (compressed to ~1.5 MB) for offline lookup of user-supplied cities and naming candidates. Nothing is sent online. |
| OpenStreetMap Overpass API | ODbL data. Fair-use public endpoint, which needs a User-Agent and small bounding boxes. Tags such as `railway=tram`, `railway=rail`, `natural=coastline`, `waterway=waterfall`, `amenity=place_of_worship`, and [`traffic_signals:sound=yes/locate/walk`](https://wiki.openstreetmap.org/wiki/Key:traffic_signals:sound) | **Optional online refinement** only when a candidate area is ≤ 100 km (for example, the user knows the city). It sends a bounding box only, never audio. |
| Open-Meteo archive API ([docs](https://open-meteo.com/en/docs/historical-weather-api)) | ERA5-based history back to 1940. Data under CC BY 4.0, free API for non-commercial use, no key | **Optional online** weather consistency check when a date and a candidate area exist. |
| Public transport GTFS | Per-agency feeds, often hundreds of MB worldwide | Not downloaded (storage). Documented as a future expansion. |
| **CesiumJS** | Apache-2.0, but the default imagery and terrain need a Cesium ion token and network. The build is 30-100 MB. | Rejected for the default. |
| **d3-geo** ([d3](https://github.com/d3/d3), ISC) | Orthographic projection (3D globe) and equirectangular (2D), rotation, zoom, and `geoCircle`, which draws **true geodesic radii**. ~280 KB, fully offline. | **Selected** for the interactive globe. Uncertainty circles are true great-circle radii in km. |
| MapLibre GL JS | BSD-3, has a globe view, but needs a tile source (online) | Future optional layer. |

Distinctive infrastructure sounds:
- **US MUTCD accessible pedestrian signals**: locator tone ≤ 0.15 s, repeating at 1 s intervals ([walkinginfo.org APS guide](https://www.walkinginfo.org/aps/appendixa)). Similar locator tones are used in other countries, so the pattern is reported as "compatible", not as a location.
- **Japan**: crossings use bird-call imitations ("piyo-piyo" chirp, "kakkō" cuckoo) and have moved away from melodies ([kuruma-news.jp](https://kuruma-news.jp/post/604477/2); [INTER-NOISE 2019](https://www.sea-acustica.es/INTERNOISE_2019/Fchrs/Proceedings/1868.pdf)). Canada also uses chirp/cuckoo. These can be confused with real birds, which is recorded as a limitation.
- **Netherlands**: the "ticker" type was selected by TNO in 1977 ([SWOV](https://swov.nl/nl/publicatie/akoestische-signalering-ten-behoeve-van-blinden-bij-voetgangeroversteekplaatsen)). No current technical spec was found, so no country weight is assigned.
- **Change ringing (campanology)**: full-circle English-style change ringing is concentrated in England, as Dove's Guide lists most towers there. This is used as moderate regional evidence with a floor for other countries.
- **Sirens**: EchoMap classifies the pitch pattern (two-tone alternation vs continuous wail/yelp sweep). The country interpretation is weak and labelled heuristic.

---

## 6. Desktop shell and report

| Candidate | Facts | Verdict |
|---|---|---|
| **pywebview 6.2.1** ([repo](https://github.com/r0x0r/pywebview), BSD-3) + **pythonnet 3.2.1** (MIT, supports Python 3.11-3.15) + clr_loader, cffi, pycparser, proxy_tools, bottle | Native Win32 window hosting the installed Edge WebView2. The full stack is ~1.3 MB. No OSV vulnerabilities for these versions. | **Selected**. |
| PySide6 / Qt | LGPL, ~400 MB installed | Rejected (storage). |
| Electron / Tauri | Electron bundles Chromium (~150 MB). Tauri needs a Rust toolchain (~1 GB+). | Rejected. |
| **PyMuPDF** (already installed, AGPL-3.0) | `Story` renders HTML/CSS to PDF | **Selected** for PDF export. The AGPL licence is fine for private use and must be reconsidered before any public distribution. |
| **PyInstaller** (already installed, GPL with bootloader exception) | Builds EchoMap.exe in GitHub Actions | **Selected**. No local build is needed, which saves disk space. |

---

## 7. Model conversion strategy (why GitHub Actions)

EfficientAT and Whisper weights are published as PyTorch checkpoints. Converting them needs PyTorch (~1 GB installed), which does not fit on the PC. EchoMap therefore runs a **GitHub Actions workflow** (`.github/workflows/convert-models.yml`) that:
1. downloads the official weights from their official URLs,
2. exports ONNX (EfficientAT includes no frontend; Whisper exports a LID-only graph),
3. runs a **parity test** comparing PyTorch output against ONNX Runtime output on fixed test signals, and fails if they differ beyond tolerance,
4. publishes the ONNX files plus SHA-256 sums as release assets in the private repository.

The mel frontends are re-implemented in numpy inside EchoMap, and CI checks them against the reference PyTorch frontends.

---

## 8. Privacy summary

- All audio analysis is **offline**. Audio never leaves the PC.
- Optional online lookups are off by default and each one is logged in the processing history:
  - GBIF (sends a species name),
  - Overpass (sends a bounding box),
  - Open-Meteo (sends coordinates and a date).
- No transcription. The language model cannot output words.
- No speaker identification.
- A user-supplied city is resolved offline (GeoNames), never sent to a geocoding service.
