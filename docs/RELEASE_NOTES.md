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
