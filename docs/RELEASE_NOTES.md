# EchoMap v1.0.9

Downloads:
- `EchoMap-v1.0.9-win64.zip`: the application.
- `EchoMap-images-v1.zip` (optional): pictures of animals, vehicles and sound sources. Put it next to the models.
- `EchoMap-transit-rail-v1.sqlite.gz` (optional): railway / tram / metro lines and stations worldwide (OpenStreetMap
  + GTFS), used to name the line or network when a station announcement or tram is heard.
- The model pack is unchanged (`EchoMap-models-v2.zip`); keep the one you have.

## New
- **Animal species: 13,786 species** (birds, insects, frogs, mammals, reptiles), trained on iNatSounds 2024 plus about
  110,000 Xeno-canto recordings. Measured on 56,630 held-out recordings: right species first 63% of the time, in the
  top 5 77% (insects 66%, birds 63%, frogs 60%, mammals 56%). A species is only shown above the confidence where it was
  right at least 70% of the time; when it is 70% sure it was right 88% of the time.
- **Pictures** for every identified animal, vehicle and sound source (not for speech, language or electrical hum).
- **"Where found" circle on the map** for everything identified, including languages.
- **Where and when for animals:** Xeno-canto recording sites (1-degree cells) and GBIF country ranges, plus a season
  check ("unusual in January").
- **Transit lines:** when a station announcement or tram is heard, the line / network / operator is matched (worldwide).
- **Weather:** rain intensity estimate; wind gusts. **Sirens:** American-style wail/yelp points to the Americas/Oceania.

## Fixed / improved
- Fake detections during church bells, trams and engines (bells were read as American birds, tram squeal as whales)
  no longer move the location. Whales and seals only count with a clear animal sound.
- Broad estimates now say "Somewhere in Europe (broad estimate…)" instead of naming one country.
- Fake languages from bells and trams are no longer used for location.

## Honest accuracy (10 real recordings with known places, Deep mode)
- Right country (or within ~60 km): Czech church bells (58 km). Right country/region: UK garden, US forest, Thailand,
  India, Germany (300-600 km from the exact spot).
- Recordings with only church bells or one common bird often get "No reliable location estimate" rather than a guess.
- Best results come from recordings with speech, station announcements, trams or sirens.
