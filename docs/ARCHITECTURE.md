# EchoMap: Architecture and Storage Plan (Phase 2)

## 1. Process model

```
EchoMap.exe (PyInstaller, Python 3.14)
 ├─ local API server   127.0.0.1:<random port>, ThreadingHTTPServer (stdlib)
 │    every request needs a per-launch random token, and the Host header must be 127.0.0.1
 │    serves: UI static files, JSON API, audio with HTTP Range (seeking), spectrogram tiles
 ├─ analysis worker     one background thread per case, with progress events
 └─ pywebview window    a native Win32 window hosting the Edge WebView2 already on Windows
                        native file dialogs (open recording, save report)
```

No other process is spawned and nothing listens on a public interface.

## 2. Package layout

```
echomap/
  app.py              entry point: config, server, window
  config.py           settings (models dir, cases dir, online toggles), stored in %LOCALAPPDATA%\EchoMap\config.json
  server.py           HTTP API and static files
  pipeline.py         orchestrates a case: ingest → dsp → detectors → fusion → geo → time → report
  provenance.py       processing history (timestamped steps, parameters, versions, hashes)
  ingest/
    media.py          PyAV probe, metadata, SHA-256, stream-copy extraction, decode/resample
  dsp/
    spectral.py       STFT, mel, log-frequency, MFCC, spectral features, LTAS
    events.py         onsets/transients, periodicity, energy segmentation, SNR
    tonal.py          narrowband tracking: mains hum/ENF, beeps, sirens, bells, harmonic engine f0
    reverb.py         blind RT60 from free decays
    wind.py           low-frequency wind-noise and gust analysis
  models/
    registry.py       model catalogue (file, source URL, sha256, size, licence, RAM/CPU notes)
    runtime.py        lazy ONNX Runtime sessions (load on demand, release after the case)
    frontends.py      numpy mel frontends matching EfficientAT and Whisper (parity-tested in CI)
    sed.py            EfficientAT AudioSet tagging over sliding windows
    birdnet.py        BirdNET+ V3 species detection
    geomodel.py       BirdNET geomodel (lat, lon, week) → occurrence
    langid.py         Whisper LID-only graph
  detect/
    taxonomy.py       AudioSet ontology → 18 EchoMap categories (expandable registry)
    inventory.py      event association, de-duplication, tiering (Confirmed/Probable/Possible/Unclassified)
    vehicles.py       engine harmonic analysis, acceleration/idle/gear-change heuristics
    infrastructure.py periodic tone patterns (pedestrian signals), sirens, bells, announcements
    environment.py    scene inference (beach, forest, urban, ...) from rules over detections and measurements
  geo/
    grid.py           0.5° land grid, country lookup, coastal flag
    evidence.py       evidence layers per independence group
    fusion.py         bounded log-score fusion, candidate extraction, uncertainty radii
    refine_osm.py     optional Overpass refinement (online, opt-in)
    gazetteer.py      offline GeoNames lookup
  temporal/
    timeinfo.py       metadata vs filesystem vs user vs estimated times, seasonal inference
    weather.py        acoustic weather observations, optional Open-Meteo comparison
  report/
    html.py           self-contained HTML report
    pdf.py            PDF via PyMuPDF Story
    staticmap.py      Pillow-rendered map for the report
  web/                index.html, app.css, app.js and vendor/ (d3, topojson-client)
  data/               compact derived datasets (CLDR languages, mains frequency, land grid, gazetteer, ontology)
tools/                data-build scripts (run once and their output committed); model conversion scripts (CI)
tests/                pytest suite and synthetic signal generators
```

## 3. Analysis pipeline

| Step | Output | Notes |
|---|---|---|
| 1 Ingest | SHA-256 of the original, metadata, analysis copy | The original is opened read-only and never modified. Video audio is stream-copied to `.mka` (no re-encode), and the settings are logged. |
| 2 Decode | float32 mono at the native rate (≤ 48 kHz) plus 32 kHz and 16 kHz model copies | Resampled with libswresample through PyAV. Multichannel audio is down-mixed to mono and each channel's level is logged. |
| 3 DSP | waveform envelope, spectrogram pyramid, features, onsets, periodicity, SNR, RT60, wind, tonal tracks, ENF | Pure numpy. |
| 4 SED | 527-class scores every 0.5 s (3 s windows) | EfficientAT mn10. |
| 5 Bio | species scores every 1.5 s (3 s windows) | BirdNET+ V3. |
| 6 Language | top-3 languages per speech region | Whisper LID. Runs only where Speech scores are high. |
| 7 Specialised | engine f0 tracks, siren patterns, periodic signals, bell strike counts | Run inside the relevant event windows. |
| 8 Inventory | one record per individual event | Association: overlapping events whose labels are parent/child or map to the same entity merge into one record. All raw model outputs are kept under `evidence.model_outputs`. |
| 9 Context | questionnaire answers (date, timezone, country, region, city, or "unknown") | Never required. |
| 10 Geo | per-cell scores, candidates, radii, evidence for and against | See `GEO_ENGINE.md`. |
| 11 Time | separate time fields and seasonal compatibility | |
| 12 Report | HTML, PDF and `case.json` (full raw scores) | |

### Confidence tiers (shown everywhere; never silently dropped)
- **Confirmed**: a deterministic measurement (for example a mains hum peak with SNR ≥ 12 dB over 3 harmonics), or a model score ≥ 0.70 corroborated by an *independent* detector (for example a BirdNET species and an AudioSet bird class in the same window).
- **Probable**: model score ≥ 0.50 with no corroboration.
- **Possible**: model score from 0.15 to 0.50.
- **Unclassified**: an energy event above the noise floor that no model explains.

Scores below 0.15 are not listed as events, but they are **kept** in `case.json` (the full score matrix) and summarised in the report. Thresholds are configurable. Scores are **model scores, not calibrated probabilities**, and every screen says so.

## 4. Storage plan

| Item | Location | Size |
|---|---|---|
| New Python packages (PyAV, pywebview stack) | existing Python site-packages | ~75 MB |
| Default models: EfficientAT mn10 (20 MB), BirdNET+ V3 FP16-pruned (71.5 MB), Geomodel FP16 (7.9 MB), labels (1.5 MB) | `%LOCALAPPDATA%\EchoMap\models` (can be moved to an external drive in Settings) | ~101 MB |
| Optional language pack: Whisper-base LID int8 | same | ~50 MB |
| Derived datasets committed to the repo | `echomap/data` | ~3 MB |
| Per case | `%LOCALAPPDATA%\EchoMap\cases\<id>` (can be moved) | ~1-3 MB plus an optional analysis WAV (≤ 29 MB for 5 min). The default does not keep a WAV for audio files, because the original is the source. |
| Test recordings | `tests/recordings` (git-ignored) | ≤ 25 MB |
| Windows build | GitHub Actions only (no local build) | 0 locally |

Total new local footprint for the default configuration: **about 180 MB**, or about 230 MB with the language pack.

Models load only when the step that needs them runs, and they are released when the case finishes. Peak RAM for a 5-minute file is estimated below 1.5 GB.
