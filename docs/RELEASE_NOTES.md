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
