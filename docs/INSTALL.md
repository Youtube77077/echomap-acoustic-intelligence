# Installing EchoMap (Windows 10/11, 64-bit)

1. From the GitHub Release, download:
   - `EchoMap-v<version>-win64.zip` (the application)
   - `EchoMap-models-v1.zip` (the models, about 150 MB)
2. Extract `EchoMap-v<version>-win64.zip` to a folder of your choice. You can use an external drive.
3. Extract `EchoMap-models-v1.zip` into the `EchoMap\models` folder inside it.
   - The models can live elsewhere, such as an external drive. If you put them somewhere else, choose that folder in **⚙ Models & settings**.
4. Run `EchoMap.exe`.

## Requirements
- **Microsoft Edge WebView2 Runtime.** Windows 11 includes it.
- **CPU.** No GPU is needed. A 5-minute recording typically analyses in under a minute on a modern laptop CPU.
- **RAM.** About 1.5 GB at peak.

## Where EchoMap stores things
- **Settings:** `%LOCALAPPDATA%\EchoMap\config.json`
- **Cases:** `%LOCALAPPDATA%\EchoMap\cases\<case>`. Each case holds `case.json`, `report.html`, and the extracted audio for videos.
  - The cases folder can be changed in the settings file.
- **Your original recordings are never modified.**

## Privacy
- All analysis runs offline.
- Optional online lookups (GBIF, OpenStreetMap, Open-Meteo) are off by default. Each one sends only:
  - species names (GBIF),
  - a bounding box (OpenStreetMap),
  - or coordinates and a date (Open-Meteo).
- Audio is never sent anywhere, and no speech is transcribed.

## Uninstall
Delete the EchoMap folder and `%LOCALAPPDATA%\EchoMap`.
