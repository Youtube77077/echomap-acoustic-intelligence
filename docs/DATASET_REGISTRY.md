# Dataset Registry

Last verification: 2026-11-12. Counts below were measured from the actual files where marked ✓.

| Dataset | Source | Licence | Access | Coverage | Samples | Used for | Attribution / restrictions | Known biases |
|---|---|---|---|---|---|---|---|---|
| Dove's Guide (dove.csv, bells.csv) | dove.cccbr.org.uk | CC BY-SA 4.0 | public CSV download (CI) | full-circle rings worldwide, mostly UK | ✓ 7,262 rings (7,141 with coordinates, 6,400 with tenor Hz, 6,200 with per-bell nominals); ✓ 64,138 bell rows | bell-tower matching | Attribution, share-alike | Continental swinging bells and carillons are largely absent |
| Unicode CLDR territoryInfo | unicode-org/cldr-json | Unicode License v3 | git | 258 territories | ✓ 252 countries mapped | language-to-country likelihood | Notice | Speaker percentages include L2 speakers (e.g. English) |
| AudioSet ontology | audioset/ontology | CC BY-SA 4.0 | git | 632 nodes | ✓ 527 model classes mapped | taxonomy, de-duplication | Attribution | - |
| Natural Earth 1:50m (world-atlas 2.0.2) | npm world-atlas | public domain (NE), ISC | npm | world countries | ✓ 59,903 land cells at 0.5° | land grid, maps | none required | Small islands may be missed at 0.5° |
| GeoNames cities15000 / admin1 / countryInfo | download.geonames.org | CC BY 4.0 | download | cities ≥ 15,000 inhabitants | ✓ 34,215 cities (without PPLX districts after rebuild) | offline city lookup, candidate naming | Attribution | - |
| Mains frequency per country | Wikipedia "Mains electricity by country" | facts (CC BY-SA source) | transcribed | ~230 territories | ~230 | electrical-grid evidence | Source cited | Mixed-grid countries simplified (Japan split modelled) |
| ESC-50 (15 clips) | karolpiczak/ESC-50 | CC BY-NC 3.0 | GitHub raw | 50 classes | ✓ 15 used | tests | Non-commercial | Freesound origins unknown (no coordinates) |
| BirdNET V3 example soundscape | birdnet-team/birdnet-V3.0-dev | see repository | GitHub | North American birds | 1 | tests | - | - |
| Wikimedia Commons reference recordings | commons.wikimedia.org (API) | per file: CC0 / PD / CC BY / CC BY-SA | official API | crossing signals, level crossings, bells, sirens, announcements, trams, trains | see `references.json.gz` meta | reverse sound search, geo reference matches, evaluations | Per-file author and licence kept | Uneven geography (Europe/US heavy); coordinates are uploader supplied |
| iNaturalist observations with sounds | api.inaturalist.org v1 | per sound: CC0 / CC BY / CC BY-SA | official API (low volume) | research-grade bird sounds | sampled per evaluation run | geolocation evaluation only | Per-sound attribution kept | Possible overlap with geomodel training data (leakage noted) |
| FLEURS (test split) | huggingface google/fleurs | CC BY 4.0 | HF download in CI | ~102 languages | 15 utterances × 20 languages per run | language-ID evaluation | Attribution | Read speech (cleaner than field audio) |
| OpenFreeMap (optional, online) | tiles.openfreemap.org | OSM ODbL data | public tiles, no key | world basemap and 3D buildings | - | 3D map | © OpenStreetMap contributors | Building heights incomplete outside cities |
| AWS Terrain Tiles (optional, online) | s3.amazonaws.com/elevation-tiles-prod | open data (Mapzen/Joerd attribution) | public tiles | global DEM (SRTM, 3DEP, …) | - | 3D terrain | Attribution required | Resolution varies by region |
| EOX Sentinel-2 cloudless (optional, online) | tiles.maps.eox.at | CC BY-NC-SA 4.0 | public tiles | global imagery | - | satellite toggle | Non-commercial | Seasonal composite, not current |
| GBIF / Overpass / Open-Meteo (optional, online) | public APIs | CC0 / CC BY, ODbL, CC BY 4.0 | APIs | - | - | display / compatibility / weather checks | per service | GBIF effort bias; OSM gaps |
