# EchoMap Audit (v1.0.1 → v1.0.2)

Audit date: 2026-11-12. Scope: the whole repository at commit 47ccbd7 (v1.0.1), before the v1.0.2 work.

## Architecture found
- **Language and frameworks:**
  - Python 3.14 backend: numpy DSP, ONNX Runtime models, PyAV/FFmpeg I/O.
  - Local HTTP API: stdlib `ThreadingHTTPServer`, token-protected, loopback only.
  - Vanilla JS/HTML/CSS frontend: d3 + topojson.
  - Desktop window: pywebview (WebView2), with an Edge app-window fallback.
  - Packaging: PyInstaller one-folder build in GitHub Actions.
- **Pipeline (`echomap/pipeline.py`):**
  - Stage A: ingest → DSP → models → inventory → scene.
  - Stage B: context → geo → time → weather → report.
  - Case data goes to `case.json`, plus a self-contained HTML report.
- **Models:** EfficientAT mn10_as (AudioSet 527), BirdNET+ V3.0-preview3.1 (~11.5K taxa), BirdNET+ Geomodel V3.0.4, Whisper-base LID-only graph.
  - All are ONNX files, loaded lazily, and released after each case.
  - Registry: `echomap/models/registry.py`.
- **Geography (`echomap/geo/engine.py`):** a bounded log-likelihood-ratio fusion over a 0.5° land grid.
  - Evidence groups: biogeography, language, electrical grid, cultural, environment, user context.
  - Correlation handling: a √n discount within each group and ±6-nat clipping.
  - Rules: species gating and an information-ratio reliability threshold.
  - Output: candidates with geodesic radii, plus local refinement.
- **Data structures:**
  - Detection records (`detect/inventory.py`) carry an ID, category, label, candidates, times, confidence, tier, frequency, evidence and limitations.
  - There is no database: each case is a JSON document. This is a deliberate choice for portability and forensic transparency.
- **Tests:** 55 at v1.0.1 (synthetic DSP, geo, taxonomy, and real-audio ESC-50 / BirdNET / end-to-end) in CI on Ubuntu and Windows.
- **Performance:** a 5 s clip analyses in about 1 s and a 120 s soundscape in about 13 s, plus about 9 s for geo on a 4-vCPU runner. Two pathological hot spots were found and fixed earlier.

## 1. Features that already worked (v1.0.1)
- Import of 10 formats, SHA-256, metadata, video stream-copy extraction, processing history.
- STFT / mel / log-frequency interactive spectrogram, waveform, frame features, MFCC, LTAS, onsets, periodicity, SNR, RT60, wind, mains hum / ENF trace.
- AudioSet events with association and de-duplication, BirdNET species, language ID, engine f0, siren patterns plus the DSP siren scanner, periodic signals, AVAS pattern, unclassified energy.
- Detection inventory (18 categories, tiers), timeline, scene rules.
- Geo inference with candidates and radii; orthographic globe; HTML report; PDF via WebView2.
- Windows exe built, tested and released by CI; startup-crash fix in v1.0.1.

## 2. Partially working
- **PDF export:** WebView2 PrintToPdf, with a print-dialog fallback. Not verified on every Windows setup.
- **AudioSet detection:** misses quiet clips (in ESC-50 testing: wind, helicopter, footsteps below threshold).
- **Language ID:** never evaluated on real speech before v1.0.2 (FLEURS evaluation added).
- **Geolocation:** never evaluated on recordings with known coordinates before v1.0.2 (iNaturalist/Commons evaluation added).
- **Event playback:** interval play existed, but context, speed and replay were missing (added).

## 3. Missing at v1.0.1 (all added in v1.0.2 unless noted)
- 2D clue map with per-clue regions.
- 3D terrain/building map.
- Unknown-sound discovery.
- Reverse sound search / signature database.
- Event relationship graph.
- Dove's Guide bell matching.
- Integrity checks (lossy history, clipping runs, dropouts, metadata consistency).
- Stereo direction cues.
- Analysis modes and capability status.
- Text search.
- Natural-language CLAP search: **not added** (see §8 of the capability registry).
- Neural source separation: **not added**.

## 4. Implemented incorrectly (found and fixed)
- O(n³) click-train scan (hang). Geomodel cost (141 s down to 9 s). Short-clip mains-hum false positives (statistics). Wind-gust baseline on digital silence. RT60 integration across the next impulse. BirdNET FP16 NaN.
- **Dove statistics:** previous "7,263 / 64,139" included the CSV header rows. Verified: **7,262 rings, 64,138 bells**.
- **Lossy-history detector:** first version measured the edge against the empty stop-band. Fixed in v1.0.2.

## 5. Need more accurate models
- **AudioSet tagging:** mn10 mAP 0.471 (larger EfficientAT/CED models would add ~2 mAP points at 5-15x size).
- **Bird species:** BirdNET+ V3 is a preview with no published accuracy.
- **Language ID:** Whisper-base (larger Whisper or ECAPA-VoxLingua107 candidates).

## 6. Need better signal processing
- Event boundaries are about 0.5 s (window pooling). Frame-level SED would be finer.
- Bell nominal extraction is sensitive to overlapping bells and distance.

## 7. Need additional datasets
- Vehicle make/model (no open dataset).
- Country-specific crossing-signal reference library (sparse on Commons).
- ENF grid-frequency logs for time verification.
- GTFS schedules for announcement timing.

## 8. Technically impossible with the available input
- RF / microwave emitter detection from a microphone.
- Source altitude or 3-D position from mono/stereo.
- Distance without a known source level.
- Speaker identity (deliberately excluded).
- Exact private location from generic sounds.
- Specific room identification from reverberation alone.

## 9. Require external hardware
- SDR receivers (RTL-SDR, HackRF, SDRplay, Airspy with GNU Radio) for RF.
- Calibrated microphone arrays for direction of arrival and elevation.

## 10. Require external APIs or cloud services (all opt-in, off by default)
- GBIF occurrence counts.
- OpenStreetMap Overpass infrastructure checks.
- Open-Meteo historical weather.
- **Online 3D map tiles:** OpenFreeMap (basemap and 3D buildings), AWS Terrain Tiles (elevation), EOX Sentinel-2 cloudless imagery (CC BY-NC-SA).
- **Not used:** Google Photorealistic 3D Tiles and Cesium ion would need your own API key or account, and billing.
  - Google applies separate terms and functionality limits to projects with an EEA billing address (effective 8 July 2025). For example, Street View tiles from the Map Tiles API are not allowed in the EEA, and some tile types return 403 errors.
  - Search did not confirm what the EEA terms say about Photorealistic 3D Tiles specifically. Check your own Google Maps Platform Agreement before enabling it.
