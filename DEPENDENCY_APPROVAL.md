# EchoMap: Dependency and Security Approval Report (Phase 3)

Prepared 2026-11-12. **Nothing in this list has been installed or downloaded yet.**
Metadata was read from PyPI, the npm registry, the GitHub API, Zenodo and OSV.dev (metadata queries only, no package files).

> A security review lowers risk but cannot prove software is safe. The checks below are what was done. They are not a guarantee.

## A. Already on the PC (reused, nothing to install)
numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1, Pillow 12.3, PyMuPDF 2.0 (AGPL-3.0, used for PDF export), shapely 2.2, requests 2.34, psutil 7.2, PyInstaller 6.22, plus the Python standard library. Edge WebView2 157 and .NET Framework 4.8.1 are already present.

## B. Python packages (install into the existing Python 3.14)

| # | Package | Publisher / repo | Purpose | Exact file (PyPI) | Size | SHA-256 (PyPI) | Licence | OSV vulns (this version) |
|---|---|---|---|---|---|---|---|---|
| B1 | **av 19.0.2** | PyAV-Org, maintainers WyattBlue and Jeremy Lainé. [github.com/PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | Decode all 10 formats, metadata, stream-copy video audio, resample | av-19.0.2-cp312-abi3-win_amd64.whl | 28.2 MB (~70 MB installed) | 3ce5a1d7b916d13c16ac58f1756f0fee4af2d53f02bc1403e18550b50d375a22 | BSD-3 bindings. Bundled FFmpeg 8.x is reported as GPL in upstream wheels. | none |
| B2 | **pywebview 6.2.1** | Roman Sirokov. [github.com/r0x0r/pywebview](https://github.com/r0x0r/pywebview) | Native desktop window with WebView2 | pywebview-6.2.1-py3-none-any.whl | 0.53 MB | 9d07275f53894ab4d5e2e0e996227193e7187dec276d9b624dccbce029216b46 | BSD-3 | none |
| B3 | pythonnet 3.2.1 (required by B2 on Windows) | Python.NET project. [github.com/pythonnet/pythonnet](https://github.com/pythonnet/pythonnet) | Bridge to the WinForms/WebView2 control | pythonnet-3.2.1-cp311…cp315-none-any.whl | 0.22 MB | dd5668b74b0c903d9a24e1ba86f728fc490f8e919d56b3c24bf06739b9ee2b7f | MIT | none |
| B4 | clr_loader 0.3.1 (required by B3) | Benedikt Reinartz (pythonnet org). [github.com/pythonnet/clr-loader](https://github.com/pythonnet/clr-loader) | Loads the .NET runtime | clr_loader-0.3.1-py3-none-any.whl | 0.06 MB | cbad189de20d202a7d621956b0fc38049e13c9bf7ca2923441eff725cd121aa1 | MIT | none |
| B5 | cffi 2.1.1 (required by B4) | python-cffi. [github.com/python-cffi/cffi](https://github.com/python-cffi/cffi) | C foreign-function interface | cffi-2.1.1-cp314-cp314-win_amd64.whl | 0.19 MB | 3222ba5d678f80a030e6afbcc33dc1ae5cb45facabb61cee2c7016b8432fde48 | MIT-0 | none |
| B6 | pycparser 3.12 (required by B5) | Eli Bendersky. [github.com/eliben/pycparser](https://github.com/eliben/pycparser) | C parser used by cffi | pycparser-3.12-py3-none-any.whl | 0.06 MB | a076f3f489b39b43ed13563a1dafe42447445bec2e294154ec5c57fcdda4c85e | BSD-3 | none |
| B7 | proxy_tools 0.1.0 (required by B2) | Jonathan Tushman. [github.com/jtushman/proxy_tools](https://github.com/jtushman/proxy_tools) | Small proxy helper used by pywebview | proxy_tools-0.1.0.tar.gz (**source only**, 3 KB) | 3 KB | ccb3751f529c047e2d8a58440d86b205303cf0fe8146f784d1cbcd94f0a28010 | MIT | none |
| B8 | bottle 0.13.4 (required by B2) | Marcel Hellkamp. [github.com/bottlepy/bottle](https://github.com/bottlepy/bottle) | Micro web server pywebview imports. EchoMap does not use it to serve files. | bottle-0.13.4-py2.py3-none-any.whl | 0.10 MB | 045684fbd2764eac9cdeb824861d1551d113e8b683d8d26e296898d3dd99a12e | MIT | none |

`typing_extensions` (also required by pywebview) is already installed.

**Findings**
- **B7 is source-only**, so pip runs its `setup.py`. I read it: a plain `setuptools.setup()` with no imports beyond `os` and `setuptools`, no network access and no file writes. The package itself is ~200 lines adapted from Werkzeug's `LocalProxy`, with no network or filesystem access.
- B3/B4 load the .NET Framework runtime in-process. That is their purpose. They have no network code.
- B1 (FFmpeg) has network protocol support compiled in, but EchoMap only opens local file paths. FFmpeg's parsers are a historical source of CVEs, which is why the original file is only ever opened read-only and the latest FFmpeg 8.x is used.
- None of these packages accesses credentials.

**Install procedure (if approved):**
```
pip download --no-deps --only-binary=:all: <wheels B1-B6,B8> -d %TEMP%\echomap-quarantine
pip download --no-deps --no-binary=:all: proxy_tools==0.1.0 -d %TEMP%\echomap-quarantine
# verify each SHA-256 against the table above, and abort on mismatch
"C:\Program Files\Windows Defender\MpCmdRun.exe" -Scan -ScanType 3 -File %TEMP%\echomap-quarantine -DisableRemediation
# list the archive contents and look for unexpected .exe/.dll/.ps1/.bat files
pip install --no-index --find-links %TEMP%\echomap-quarantine --require-hashes -r requirements-lock.txt
# then delete the quarantine folder
```

## C. Models (downloaded to `%LOCALAPPDATA%\EchoMap\models`, can be moved to an external drive)

| # | Model | Official source | File | Size | Integrity | Licence | RAM / speed (estimate, measured later) |
|---|---|---|---|---|---|---|---|
| C1 | **BirdNET+ V3.0-preview3.1** audio model, FP16 pruned | Cornell Lab / TU Chemnitz, [Zenodo 20703646](https://zenodo.org/records/20703646) | BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx | 71.5 MB | md5 8278be6e7d1a9fc1717cf953e269b2e3 (Zenodo) | CC BY-SA 4.0 plus Terms of Use (no poaching or military use) | ~400 MB RAM, CPU only. Expected under 1 min for 5 min of audio. |
| C2 | BirdNET+ V3 labels | same | BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv | 0.8 MB | md5 21ccf9e984a955bfc2d5c52faab6a6ea | CC BY-SA 4.0 | - |
| C3 | **BirdNET+ Geomodel V3.0.4**, FP16 | [github.com/birdnet-team/geomodel v3.0.4](https://github.com/birdnet-team/geomodel/releases/tag/v3.0.4) | BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx | 7.9 MB | sha256 80e410f811bf4455e6cb1a2d48e394737d3828fe943a444646778495b67e7bab | Repo says Apache-2.0 for model artifacts (LICENSE-MODELS.md). A third-party mirror says CC BY-SA. | ~100 MB RAM, about a second for the global grid |
| C4 | Geomodel labels + licence | same | …_Global_14K_Labels.txt, LICENSE-MODELS.md, ACCEPTABLE_USE.md | 0.7 MB | sha256 8250b457… (labels) | same | - |
| C5 | **EfficientAT mn10_as** → ONNX | Weights from [fschmid56/EfficientAT v0.0.1](https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1) `mn10_as_mAP_471.pt` (19.7 MB), **converted in GitHub Actions**, not on your PC | mn10_as.onnx | ~20 MB | sha256 recorded by CI, plus a parity test against PyTorch | MIT | ~200 MB RAM, a few seconds per 5 min |
| C6 | **Whisper base**, language-ID-only graph → ONNX int8 (optional language pack) | Weights from OpenAI (`openai/whisper-base` on Hugging Face, MIT), **converted in GitHub Actions** | whisper_base_lid_int8.onnx | ~50 MB | sha256 recorded by CI, plus a parity test | MIT | ~300 MB RAM, ~1 s per speech region |

## D. Data files (one-time download; compact derived versions committed to the repo)

| # | Data | Source | Size | Licence |
|---|---|---|---|---|
| D1 | CLDR territoryInfo (language share per country) | [unicode-org/cldr-json](https://github.com/unicode-org/cldr-json) | 15 KB (compressed transfer) | Unicode License v3 |
| D2 | AudioSet ontology | [audioset/ontology](https://github.com/audioset/ontology) ontology.json | 86 KB (compressed) | CC BY-SA 4.0 |
| D3 | AudioSet class index (527) | EfficientAT metadata/class_labels_indices.csv | 7 KB | MIT |
| D4 | Natural Earth countries 50m (TopoJSON) | npm `world-atlas@2.0.2` (countries-50m.json only) | ~750 KB | Public domain (NE), ISC wrapper |
| D5 | GeoNames cities15000 + admin1 codes | [download.geonames.org](https://download.geonames.org/export/dump/) | 3.4 MB zip + 0.15 MB (reduced to ~1.5 MB) | CC BY 4.0 |
| D6 | Mains frequency per country | Wikipedia "Mains electricity by country" table, transcribed to JSON | ~5 KB | Facts (CC BY-SA source) |

## E. JavaScript (vendored into `echomap/web/vendor`, no npm install, no build step)

| # | Library | Source | File | Size | Licence |
|---|---|---|---|---|---|
| E1 | d3 7.9.2 | npm `d3` (Mike Bostock / Observable) | d3.min.js | ~280 KB | ISC |
| E2 | topojson-client 3.1.0 | npm `topojson-client` | topojson-client.min.js | ~7 KB | ISC |

Files are fetched as single files from the npm registry tarball, checked against the npm `integrity` (sha512) value, and Defender-scanned.

## F. Test recordings (git-ignored, ≤ 25 MB, deleted on request)

Up to 12 short clips with known content from freely licensed sources:
- BirdNET V3 repo `example/soundscape.wav` (birds),
- selected ESC-50 clips (CC BY-NC; for example church bells, siren, train, rain, crickets, engine),
- Wikimedia Commons recordings (tram, station announcement, change ringing).

Deterministic synthetic signals (mains hum, 1 Hz locator ticks, two-tone sirens, engine harmonics) are generated in code and need no download.

## G. GitHub actions taken through your existing `gh` login (account shown by `gh auth status`)

No credentials are requested. A GitHub MCP server is not connected in this session, so the authenticated `gh` CLI is used instead.

1. Create the **private** repository `echomap-acoustic-intelligence`. This is requested **now** (moved forward from Phase 9) because model conversion (C5, C6) must run in GitHub Actions to avoid installing PyTorch on your PC.
2. Workflows:
   - `convert-models.yml`: Ubuntu runner, installs torch-cpu, transformers and onnx **on GitHub's machine only**.
   - `build-windows.yml`: Windows runner, PyInstaller.
   - `test.yml`
3. Releases containing `EchoMap-vX-win64.zip` and a separate `EchoMap-models-vX.zip`.

## H. Total local footprint if everything is approved
Packages ~75 MB, models ~101 MB (plus ~50 MB with the language pack), data ~3 MB, test clips ≤ 25 MB. **About 205 MB, or 255 MB with the language pack.** Free space now: ~3 GB.
