# EchoMap Capability Registry

This document is maintained in the repository. Every capability lists what does the work, how it was tested, and its known limits.
Status values:
- **Tested**: verified on real or synthetic recordings in CI; results are in `docs/TEST_RESULTS.md`.
- **Implemented**: the code exists, but there is no labelled real-audio test yet.
- **Researched**: an investigation and plan exist, but nothing is built.
- **Not feasible (yet)**: an investigation found no credible route.

Last updated: 2026-11-12.

## 1. Implemented capabilities

| Capability | Status | Responsible model / algorithm | Test evidence | Known limitations |
|---|---|---|---|---|
| Import of MP3, WAV, FLAC, OGG, M4A, AAC, MP4, MOV, MKV, WEBM; SHA-256; metadata | Implemented (WAV tested) | PyAV / FFmpeg 8 | CI: every case hashes and probes its file | Other formats rely on FFmpeg and are not yet covered by a labelled test file |
| Lossless audio extraction from video | Implemented | PyAV stream copy to Matroska | Untested on video (no test video yet) | |
| Waveform, STFT, mel, log-frequency spectrogram, interactive zoom | Tested (synthetic) | numpy (`dsp/spectral.py`) | `test_spectrogram_*` | |
| Frame features: RMS, peak, ZCR, centroid, bandwidth, roll-off, flatness, MFCC, LTAS | Tested (synthetic) | numpy | `test_features_centroid_tracks_tone` | |
| Transients, periodicity, SNR, energy segmentation | Tested (synthetic) | Spectral flux, autocorrelation, percentiles | `test_onsets_and_periodicity`, `test_snr_estimate_orders` | SNR is a percentile estimate |
| Blind reverberation (RT60) | Tested (synthetic) | Schroeder backward integration on free decays | `test_rt60_synthetic_decay` (0.8 s ±0.25) | Needs ≥ 5 impulsive decays; coarse indicator only |
| Mains hum 50/60 Hz + ENF trace | Tested (synthetic); real-clip behaviour under review | Welch PSD harmonic SNR | `test_mains_hum_*` | Short clips (< 10 s) cannot be "confirmed"; motors can mimic hum |
| 527-class sound events (AudioSet) | Tested (15 ESC-50 classes, top-10) | EfficientAT mn10_as (ONNX) | `test_sed_topk_contains_truth` | Saturated output for windows ≤ 3 s (EchoMap uses 5 s); low scores on quiet clips |
| Bird / insect / amphibian / mammal species | Tested | BirdNET+ V3.0-preview3.1 | Soundscape: ≥ 3 bird species ≥ 0.5; crickets → *Tettigonia viridissima* 0.85 | Preview model with no published accuracy; spurious low-score species occur on non-animal sounds (e.g. 0.19 "Humpback Whale" on bells) |
| Species range and seasonality | Implemented | BirdNET+ Geomodel V3.0.4 | Used in the geo tests on the soundscape | Biased towards well-surveyed regions |
| Spoken language identification (99 languages) | Tested (CI parity on TTS speech) | Whisper-base, LID-only ONNX | Parity: int8 top-1 equals FP32 top-1. Misidentifies robotic TTS French as Thai and Dutch as German | No real-speech labelled test yet |
| Engine firing frequency, acceleration, gear-change heuristics | Tested (synthetic) | Harmonic-sum f0 tracking | `test_engine_f0_rising` | Doppler shift also raises f0; events limited to road-engine labels with continuous f0 |
| Siren pitch pattern (two-tone, wail, yelp) | Tested (synthetic) | Pitch-contour statistics | `test_two_tone_siren`, `test_wail_siren` | Real-clip test pending (ESC-50 siren scored low on the AudioSet "Siren" class) |
| Periodic beeps, ticks, locator tones | Tested (synthetic) | Tone tracking plus regularity | `test_periodic_beeps_1hz_locator_pattern`, `test_click_train` | The source type is an interpretation |
| Wind gusts (microphone wind noise) | Implemented | Low-frequency envelope with LF-dominance check | Real clips reviewed | No wind speed in m/s |
| Scene inference (beach, forest, urban, railway, hall, …) | Implemented | Hand-weighted rules over AudioSet scores | Train clip → "Railway environment" | Weights are heuristics, not trained |
| Geographic inference with uncertainty radii | Tested (unit) + soundscape | Bounded log-likelihood fusion over a 0.5° land grid | `test_geo_*`; soundscape → US/Canada | Without strong evidence it reports "no reliable estimate" |
| Temporal analysis (time sources, species seasonality, clock strikes) | Implemented | `temporal/timeinfo.py` | | No external time verification |
| Report: HTML and PDF | Tested (HTML in end-to-end test) | Self-contained HTML; PDF via WebView2 PrintToPdf | `test_end_to_end_pipeline_and_report` | PDF is only tested on Windows builds |

## 2. Missing or partial capabilities: investigations

For each gap: what must be detected, prior work, a candidate approach, resources needed, licence, how to test it, a realistic accuracy expectation, and the decision.

### 2.1 Vehicle manufacturer, model, generation
- **What must be detected:** the make and model of a vehicle from its engine or exhaust sound.
- **Prior work:**
  - The closest published result is fuel type, not make/model: petrol vs diesel at **91.7%** on 80 dealership recordings ([Electronics 12(9) 2012, 2023](https://mdpi.com/2079-9292/12/9/2012)).
  - Engine attributes (fuel type, configuration, cylinder count, aspiration) reached **> 92%** validation accuracy, but on a *synthesized/augmented* 40-hour dataset ([arXiv 2205.09667](https://arxiv.org/pdf/2205.09667)).
  - A make/model figure of 98% is cited only second-hand in a peer review, and the original paper could not be found.
- **Decision:** make/model is **not feasible** with public data. No open dataset covers more than a handful of models under real traffic conditions.
- **Engineered alternative (planned):** an attribute classifier (fuel type, cylinder-count band) trained on EchoMap's own harmonic features plus public engine clips.
  - Needs a dataset build. Candidate sources are Freesound CC0/CC-BY engine recordings and the IDMT-Traffic dataset (Fraunhofer, check licence).
  - Training would run in GitHub Actions or another cloud runner. The model would be < 1 MB (gradient boosting over harmonic features).
  - **Requires your approval** for dataset downloads into the cloud.

### 2.2 Electric vehicles (AVAS)
- **What must be detected:** EVs emitting their mandatory warning sound below 20 km/h.
- **Prior work:** UN R138 and EU Regulation 2017/1576 require:
  - one-third-octave content from 160 Hz to 5 kHz, with at least one band at or below 1.6 kHz;
  - a tone whose frequency shifts with speed, on average ≥ 0.8% per km/h between 5 and 20 km/h ([Acoustics Today, Winter 2020](https://acousticstoday.org/issues/2020AT/Winter2020/files/basic-html/page24.html); [EU 2017/1576](https://legislation.gov.uk/eur/2017/1576/data.pdf)).
- **Approach:** pure DSP, no new dependency.
  - Look for tonal components (from the existing tone tracker) in 160 Hz–5 kHz whose frequency changes slowly and steadily over seconds, without a low-frequency combustion harmonic series.
  - Report "AVAS-like tonal pattern (possible EV at low speed)".
- **Test:** synthetic AVAS-style sweeps (rule-derived), then real EV pass-by clips from Freesound (needs approval).
- **Expected:** useful for low-speed pass-bys only. Above ~20–30 km/h tyre noise dominates and AVAS may switch off. **Status: Researched, next to build.**

### 2.3 Church bells: identification and geolocation
- **What must be detected:** which tower (or kind of ring) a bell recording comes from.
- **Prior work:**
  - Bell acoustics: a strike note dominated by the *nominal* partial, with hum, prime, tierce and quint partials in characteristic ratios.
  - **Dove's Guide** lists full-circle rings worldwide with **tenor weight, tenor note/frequency and location**, and offers CSV downloads under **CC BY-SA 4.0** ([dove.cccbr.org.uk/downloads](https://dove.cccbr.org.uk/downloads)).
- **Approach:**
  1. Estimate the ring size from distinct strike pitches in change-ringing segments.
  2. Estimate each bell's nominal partial with high-resolution spectral peaks; the lowest is the tenor.
  3. Match (number of bells, tenor nominal ± tolerance) against Dove's tower list, giving candidate towers with coordinates.
  4. Feed those candidates as a sharp geographic evidence layer, with a floor for unlisted towers.
- **Resources:** Dove CSV is a few MB, stored in the repo.
- **Test:** Wikimedia Commons recordings of named towers (needs approval).
- **Expected:** strong narrowing in England when ≥ 6 bells are clearly audible. Tuning and recording-speed errors limit precision. **Status: Researched, proposed. Needs approval for the Dove's Guide download.**

### 2.4 Country-specific pedestrian crossing signals
- **What must be detected:** which country's crossing-signal design is audible.
- **Prior work:**
  - Patterns found so far: US MUTCD locator tone (≤ 0.15 s tone every 1 s); Japanese bird-call imitations ("piyo", "kakkō"); Dutch tickers (TNO 1977). Other countries are not documented.
  - OSM tags `traffic_signals:sound=*` mark crossings with audible signals.
- **Approach:**
  1. Build a small reference library of documented signal recordings, each with a source and a country.
  2. Match by period, pulse duration, pitch and spectral envelope (DTW on the pitch track).
  3. Use OSM audible-signal density inside candidate areas as a compatibility check.
- **Resources:** a few MB of reference clips (Freesound or Wikimedia, with licences recorded).
- **Expected:** discriminates *signal families* (bird-call vs ticker vs tone), not exact cities. **Status: Researched. Needs approval for the reference clip downloads.**

### 2.5 Open-vocabulary detection (sounds outside AudioSet)
- **What must be detected:** classes such as "tram bell", "ticket machine" or "pedestrian signal chirp" that AudioSet does not cover.
- **Prior work:**
  - CLAP-style zero-shot models. TinyCLAP keeps < 5% zero-shot loss at 6% of Microsoft CLAP's size ([arXiv 2311.14517](https://arxiv.org/abs/2311.14517)).
  - Whisper-CLAP (LAION) pairs a Whisper-small encoder with GTE text embeddings. No ready ONNX export of either was found.
- **Approach:** export TinyCLAP to ONNX in GitHub Actions (same pattern as `convert-models.yml`), with a parity test, and use it for second-opinion labels on unclassified segments.
- **Resources:** an estimated 30–60 MB.
- **Status: Researched. Needs approval** (new model).

### 2.6 Better language identification
- **What must be improved:** Whisper-base LID confuses related languages and misread synthetic speech.
- **Candidate:** SpeechBrain ECAPA VoxLingua107 (Apache-2.0, ~85 MB), as a second model in an ensemble.
- **Test:** real-speech labelled set (e.g. FLEURS, CC BY 4.0), streamed in CI only.
- **Status: Researched. Needs approval.**

### 2.7 ENF time verification
- **What it would do:** match EchoMap's existing ENF trace against grid frequency logs to verify recording time.
- **Requirement:** grid frequency logs. Some operators publish them (e.g. UK National Grid ESO, 1-second historic frequency data under an open licence).
- **Approach:** an optional online lookup per user-supplied date and grid; correlate the traces.
- **Status: Researched. Needs a data-source review.**

### 2.8 Public-transport schedules
- **What it would do:** use station announcements plus a time to narrow location.
- **Requirement:** GTFS feeds (often hundreds of MB worldwide; per-city feeds via Mobility Database).
- **Approach:** user-selected city feeds fetched on demand, in the cloud or on request.
- **Status: Not started.** Depends on reliable announcement detection.

### 2.9 Source separation for overlapping sounds
- **Candidates:** Demucs and AudioSep (hundreds of MB, GPU-oriented).
- **Decision:** too heavy for a lightweight exe. Possible as a cloud-side optional pass.
- **Status: Researched, deferred.**

## 3. Resource summary

| Item | Size | Location |
|---|---|---|
| Core models (EfficientAT, BirdNET+, Geomodel, Whisper LID) | ~150 MB | `EchoMap-models-v1.zip` release asset, unpacked next to the exe |
| App (exe and libraries, no models) | target < 150 MB unpacked | `EchoMap-v*-win64.zip` |
| Derived datasets | ~2 MB | inside the app |
