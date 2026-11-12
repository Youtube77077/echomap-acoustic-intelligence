# EchoMap v1.0.8

Downloads:
- `EchoMap-v1.0.8-win64.zip`: the application.
- `EchoMap-models-v2.zip`: same model pack as before (no change). If you already have it, keep it.

This release fixes the problems reported with v1.0.7. Every change was checked on GitHub's machines with real audio
(Dutch read speech from FLEURS, a street mix with that speech under traffic, and a 5-minute clip) in Quick, Standard
and Deep mode. That end-to-end test now runs on every change (`realaudio` workflow).

## Fixed
- **Analysis never finishing / far too slow.**
  - Deep mode: the reverse sound search (fingerprints) took 108 s of a 217 s run on a 5-minute file. It is now
    vectorised and takes about 2 s.
  - Speech transcription had no time limit and no progress. It now skips segments that Whisper itself marks as
    "no speech" (most false candidates), caps the text length to the segment length, transcribes the clearest speech
    first within a time budget (Quick 20 s + 0.5x length, Standard 30 s + 1x, Deep 45 s + 2x), and shows progress.
  - Quick mode transcribes clear speech only; Deep also tries faint background speech.
- **No location shown.** The location step now runs automatically at the end of every analysis. "Recording context"
  became "Refine location…" (optional: add a date, country or city and re-run).
- **Dutch (any language) not identified.** Language identification now runs in every mode, and transcribed speech
  also yields a language record (from Whisper's language token), so speech under traffic noise still counts.
- **Wrong country for a clear language.** Speech, place names and transit sounds now carry a population prior (people
  make these sounds where people live). Dutch speech gave "Suriname" first; it now gives "The Netherlands, Belgium".
- **Sounds not described.**
  - New **Findings** panel at the top: Where (most likely area, likely places, alternatives, which clues),
    Language, What was said (transcript lines), What the sounds are (plain descriptions with time and confidence).
    The same summary opens the report.
  - Vehicle type (car / bus / truck / tram / motorcycle) now runs in every mode and is shown in the inventory. When
    the vehicle model is unsure (near-equal scores) it says "uncertain" instead of naming a type.
  - Clear speech (score 0.74) was shown as "Unclassified sound" and labelled "Speech synthesizer"; fixed.
- **Laggy 2D clue map.** While dragging or zooming, a cached image of the map is moved instead of redrawing every
  border, label and city; the full redraw follows when you stop. 1:10m borders only load when zoomed in.

## Measured (GitHub 4-core runner)
| Audio | Quick | Standard | Deep |
|---|---|---|---|
| 42 s Dutch speech | 13 s | see Actions | see Actions |
| 60 s street + Dutch | 14 s | 39 s | see Actions |
| 300 s mix | 90 s | 62 s | 217 s before the fingerprint fix |

A home PC can be 2-4x slower than this runner.

## Known limits (unchanged)
- Location from a language alone is country-level (about 300 km). City-level needs strong clues (a spoken place
  name, matched bells, an exact announcement match).
- The vehicle-type model was trained in Melbourne only (67% balanced accuracy there).

# EchoMap v1.0.7

- Fixed: 'Settings' object has no attribute 'save' when starting an analysis (regression in v1.0.6).

# EchoMap v1.0.6

- Fixed: models were reported missing when the model zip was extracted next to EchoMap.exe. EchoMap now finds the model pack automatically (next to the exe at any nesting, in %LOCALAPPDATA%\EchoMap\models, or in Downloads/Desktop).

# EchoMap v1.0.5

- Fixed: Import failed with "not a valid file filter".
- Fixed: clue map lag (one redraw per frame, cached overlays).

# EchoMap v1.0.4

Downloads:
- `EchoMap-v1.0.7-win64.zip`: the application.
- `EchoMap-models-v2.zip`: same model pack as v1.0.2 (no change). Extract into `EchoMap\models`.

## New
- Bundles the worldwide tram-network dataset, the city transit sound library, the larger reference library and the trained vehicle-type head (car / bus / truck / tram / motorcycle; 67% balanced accuracy on held-out Melbourne sites), where their builds succeeded.
- (v1.0.3 was not published separately; its features are listed below.)
- **Circles and likely places.**
  - Point clues are now ranked as concrete **likely places**, each drawn as a circle sized to the place: tram networks, cities named in speech, bell towers. Every place is scored with *all* evidence evaluated at that exact point.
  - Each place shows a relative share. That share is relative among the places considered, not a calibrated probability.
  - The estimate areas are drawn as filled circles labelled with their radius.
- **Tram / streetcar clue (worldwide).**
  - New dataset of every tram network on OpenStreetMap (railway=tram track), built on GitHub.
  - The sound model has no "tram" class, so a tram is hypothesised from a close match to a tram reference recording, or (weakly) from rail sound together with a bell/ding or wheel squeal.
- **Spoken place names.**
  - Transcribed speech is searched for city names: 31,821 GeoNames cities over 15,000 inhabitants, Latin script.
  - Announcement phrases ("next stop", "příští zastávka", "nächster Halt", …) raise confidence.
  - Common words that are also city names ("Reading", "Mobile", "Nice", …) need such a phrase.
  - Mentioning a place is not proof of being there, so this evidence is moderate and floored.
- **Sharper, zoomable map.**
  - Country-level clues (language, mains hum) are drawn with exact country borders instead of 55 km squares.
  - Sharper 1:10m borders load when zoomed in.
  - + / − / "most likely place" / reset zoom buttons.
  - Zoom up to 2000×.
  - The view auto-zooms to the likely places.
- **Filters and transparency.**
  - Filter chips show or hide every clue of one kind (Language, Electrical hum, Wildlife, Bells, Trams, Place names, …).
  - Click a clue's name to make it see-through, use its opacity slider, or ◉ to show only that clue.
  - A global fill slider controls all clue regions.
- **3D map:**
  - "Flat street map" mode (Mercator, no tilt) for street-level zoom.
  - Likely-place circles.
  - "Most likely place" button.
- **Reports:** world view plus a zoomed view of the most likely area, drawn as vectors. They also include a likely-places table and a list of place names heard in speech.
- **Worked examples** (`tools/example_geo.py`): hand-specified clue sets (tram + Czech + 50 Hz, the same with a spoken stop name, tram + English + 50 Hz, Spanish + 60 Hz) run through the real engine. Screenshots come from the real UI.

## Limits (stated in the app)
- A tram sound alone narrows to tram cities worldwide, not to one city. Language, mains hum or a spoken name are needed to separate cities.
- English is a weak clue: CLDR counts second-language speakers, so many countries share English. Accent identification is not included.
- Island / small-grid identification from hum fluctuations is **not** included in this version. No verified reference statistics are bundled yet.

# EchoMap v1.0.2

Downloads:
- `EchoMap-v1.0.2-win64.zip`: the application.
- `EchoMap-models-v2.zip`: models, now including the speech-to-text pack. Extract into `EchoMap\models`.

## New
- **Clue map (2D).**
  - Every locatable clue gets its own coloured region with a smooth white outline and a label: bird and wildlife ranges, languages, mains grid, bells / Dove's Guide towers, coast, sirens, reference recordings, your own context.
  - The **"most clues agree" zone** is highlighted.
  - Low-confidence clues can be toggled on (dashed).
  - Clicking any point lists the clues that cover it and the used clues that contradict it.
  - **Unlocated evidence** (sounds that occur everywhere) is listed separately.
- **3D map (opt-in, online).**
  - MapLibre globe with terrain (AWS Terrain Tiles), 3D buildings (OpenFreeMap) and optional Sentinel-2 imagery (EOX).
  - Clue regions, combined probability and candidate circles are draped on the terrain.
  - Off by default because tile requests reveal which areas you view.
- **Speech transcription.** Local Whisper-base transcription of all speech, including faint background speech, which is filtered, noise-reduced and level-normalised first. Every line is labelled clear / uncertain / likely spurious.
- **Intelligence tab:**
  - text search over the recording;
  - unknown-sound discovery (novelty, model disagreement, acoustic families, comparison with earlier unknowns);
  - event relationships (overlap, sequences, repetition, level-crossing / pass-by / fly-over / announcement hypotheses);
  - reverse sound search against a reference library of openly licensed recordings (landmark fingerprints for near-exact matches, embeddings for similarity tiers);
  - bell partial analysis matched against Dove's Guide (7,262 rings, 64,138 bells);
  - stereo direction cues;
  - forensic integrity checks;
  - capability status (what ran, what did not, and why).
- **Analysis modes:** Quick, Standard, Deep (default), Experimental.
- **Playback:** context before and after events, speed, replay, return to full recording, and listening to only an event's frequency band.
- **Report:** clue-map image and sections for transcripts, integrity, bells, references, relationships, unknown sounds, direction and capabilities.
- **Documentation:** `docs/AUDIT.md`, `docs/MODEL_REGISTRY.md`, `docs/DATASET_REGISTRY.md`, updated capability registry, and cloud evaluation results in `docs/eval/`.

## Fixed
- The lossy-history detector now measures the cut-off against the passband (brick-wall test).
- Data workflows rebase before pushing.

## Not included (see docs/CAPABILITY_REGISTRY.md)
- TinyCLAP natural-language model (verified; conversion pending).
- Neural source separation.
- Vehicle make/model.
- RF detection (needs SDR hardware).
- Google Photorealistic 3D Tiles (needs your own billed API key; EEA terms apply).
