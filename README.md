# EchoMap: acoustic intelligence, audio forensics and probabilistic geolocation

EchoMap is a Windows desktop application. You give it an audio or video recording of up to 5 minutes, and it:

1. **Preserves and fingerprints** the original. The file is opened read-only, its SHA-256 is recorded, and the audio track of a video is extracted by stream copy without re-encoding.
2. **Analyses the signal**: waveform, STFT, mel and log-frequency spectrograms, spectral features, MFCC, transients, periodicity, SNR, reverberation, wind noise, tonal tracks, and mains hum / ENF.
3. **Detects every supported sound**, each as an individual, timestamped record:
   - 527 AudioSet classes (EfficientAT);
   - about 11.5K bird, insect, amphibian and mammal species (BirdNET+ V3);
   - 99 spoken languages, identified without transcription (Whisper LID-only graph);
   - engine harmonics, siren patterns, bell strikes, and periodic signals such as crossing signals and beepers.
4. **Builds a complete detection inventory** in 18 expandable categories, with confidence tiers (confirmed / probable / possible / unclassified). Raw model outputs are preserved and duplicates are merged.
5. **Estimates where the recording may have been made**, combining independent evidence groups (species ranges and seasonality, language-population data, the 50/60 Hz grid, cultural infrastructure, coastal environment, and optional user context) into candidate areas with **geodesic uncertainty radii** on an interactive 3D globe. When the evidence is too weak, it says "no reliable estimate".
6. **Produces a forensic report** in HTML and PDF covering metadata, every detection, the evidence for and against each candidate area, model information, limitations, and the full processing history.

Nothing is invented: species, vehicles and languages are reported as model predictions with scores, and anything EchoMap cannot determine is labelled as such.

## Repository map

| Path | Contents |
|---|---|
| `echomap/` | Application source: ingest, DSP, models, detection, geo, temporal, report, server, web UI |
| `data-sources/` | Raw open datasets (CLDR, AudioSet ontology, GeoNames). Derived data lives in `echomap/data/` |
| `tools/` | Data build, model fetch, and model conversion (`tools/convert/`, runs in CI) |
| `tests/` | Synthetic-signal, geo, and real-audio tests (test audio lives in the `test-audio-v1` release) |
| `.github/workflows/` | `convert-models`, `test`, `analyze` (cloud case runner), `build-data`, `build-windows` (exe and release) |
| `docs/` | Architecture, capability registry, test results, install guide |
| `RESEARCH.md`, `DEPENDENCY_APPROVAL.md` | Phase 1 research and the vetted dependency list |

## Cloud-first development
All building, testing, model conversion and packaging runs on GitHub Actions. Large binaries live in private releases:
- `models-v1`: ONNX models plus checksums;
- `test-audio-v1`: labelled clips;
- `build-deps-v1`: pinned wheels.

The Windows app is published as a release ZIP, with the models as a separate ZIP so the app download stays small.

## Getting the app
See [docs/INSTALL.md](docs/INSTALL.md).
