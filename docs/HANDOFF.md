# EchoMap handoff (for the next session)

Updated 2026-11-12 (v1.0.8 work, branch `claude/sleepy-hopper-e169di`).

## Where things stand
- v1.0.8 fixes the v1.0.7 complaints: slow/never-finishing analysis, no transcription/language, sounds not described,
  no country shown, laggy 2D map. See `docs/RELEASE_NOTES.md` for the full list and measured timings.
- The location step now runs automatically after every analysis (`Case.analyse(auto_geo=True)`); the context dialog is
  "Refine location…". Tools that pass their own context call `analyse(auto_geo=False)` then `contextualise(ctx)`.
- `result["findings"]` (`echomap/detect/findings.py`) is the plain-language summary shown first in the UI and report.

## How to test (GitHub only, nothing on the user's PC)
- `realaudio.yml` runs on every push to `claude/**` that touches `echomap/**`: 3 modes x 3 files (Dutch FLEURS speech,
  street mix with that speech, 5-minute mix), with cProfile output and `tests/test_real_pipeline.py` assertions
  (finish time, Dutch transcribed + `nl` identified, Netherlands/Belgium located, vehicle type attached).
- `tools/profile_case.py <file> <mode> out.json` prints step timings, slowest functions and what the user would see.
- `build-windows.yml -f release=true` builds the exe, runs the Windows tests and the UI screenshot test, and publishes
  release `v<APP_VERSION>` with `docs/RELEASE_NOTES.md`.

## Performance notes (5-minute file, CI runner)
- Biggest costs now: Whisper transcription (budgeted per mode in `capabilities.ASR_BUDGET`), BirdNET (~22 s),
  geomodel occurrence (~7 s), report maps (~6-7 s).
- Whisper decoder has no KV cache (cross-attention recomputed per token). Re-exporting it with a cache in
  `tools/convert/export_whisper_asr.py` would make transcription several times faster.

## Open items
1. Mislocated transit clips (~15) in `transit_refs.json.gz`: add a blocklist in `tools/build_transit_refs.py` and rebuild.
2. `tram_networks.json.gz` still not built (Overpass unreliable); use a Geofabrik/osmium extract.
3. `eval.yml -f which=misc` never run; also re-run `which=geo` to check the new population prior does not hurt the
   iNaturalist geolocation numbers (it only applies when speech / place names / transit clues exist).
4. Whisper decoder with KV cache (speed).
