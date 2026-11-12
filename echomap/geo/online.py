"""Optional online lookups. All are OFF by default and each call is logged in the processing history.

- GBIF: sends a species name, returns occurrence counts per country (display only, never fused).
- Overpass (OpenStreetMap): sends a bounding box, returns counts of infrastructure features matching detections.
- Open-Meteo archive: sends coordinates and a date, returns hourly weather for a consistency check.
No audio and no user-supplied text is ever sent.
"""
from __future__ import annotations

import requests

UA = {"User-Agent": "EchoMap/1.0 (desktop acoustic analysis; low volume)"}
TIMEOUT = 25


def gbif_country_counts(scientific_name: str, history=None) -> dict:
    m = requests.get("https://api.gbif.org/v1/species/match", params={"name": scientific_name}, headers=UA, timeout=TIMEOUT).json()
    key = m.get("usageKey")
    if history:
        history.log("online.gbif", "species match", name=scientific_name, usageKey=key)
    if not key:
        return {"matched": False}
    r = requests.get("https://api.gbif.org/v1/occurrence/search",
                     params={"taxonKey": key, "facet": "country", "facetLimit": 15, "limit": 0}, headers=UA, timeout=TIMEOUT).json()
    facets = (r.get("facets") or [{}])[0].get("counts", [])
    return {"matched": True, "usageKey": key, "total_occurrences": r.get("count"),
            "top_countries": [{"iso2": f["name"], "count": f["count"]} for f in facets],
            "licence_note": "GBIF-mediated data, CC0 / CC BY / CC BY-NC per dataset",
            "note": "Recording effort is uneven, so counts reflect observers as well as birds. Display only, not fused."}


OSM_QUERIES = {
    "railway": ('nwr["railway"~"^(rail|light_rail|subway|tram)$"]', {"railway"}),
    "tram": ('nwr["railway"="tram"]', set()),
    "coastline_beach": ('nwr["natural"~"^(coastline|beach)$"]', {"environment_coast"}),
    "waterfall": ('nwr["waterway"="waterfall"]', {"waterfall"}),
    "place_of_worship_christian": ('nwr["amenity"="place_of_worship"]["religion"="christian"]', {"church_bell"}),
    "audible_traffic_signals": ('nwr["traffic_signals:sound"~"^(yes|locate|walk)$"]', {"ped_signal"}),
    "level_crossing": ('nwr["railway"~"^(level_crossing|crossing)$"]', {"railway"}),
}


def overpass_feature_counts(lat: float, lon: float, radius_km: float, wanted: list[str], history=None) -> dict:
    radius_m = int(min(radius_km, 100) * 1000)
    parts = []
    for key in wanted:
        q = OSM_QUERIES.get(key)
        if q:
            parts.append(f'{q[0]}(around:{radius_m},{lat:.4f},{lon:.4f});out count;')
    if not parts:
        return {}
    query = "[out:json][timeout:25];" + "".join(parts)
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, headers=UA, timeout=60)
    r.raise_for_status()
    els = r.json().get("elements", [])
    counts = {}
    for key, el in zip([k for k in wanted if k in OSM_QUERIES], els):
        counts[key] = int(el.get("tags", {}).get("total", 0))
    if history:
        history.log("online.overpass", "feature counts", centre=[round(lat, 3), round(lon, 3)], radius_km=radius_km, wanted=wanted)
    return {"radius_km": radius_km, "counts": counts, "source": "OpenStreetMap contributors (ODbL), Overpass API",
            "note": "Presence makes a candidate compatible. Absence may be a gap in OSM data."}


def open_meteo_day(lat: float, lon: float, date: str, history=None) -> dict:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": date, "end_date": date,
        "hourly": "precipitation,wind_speed_10m,wind_gusts_10m,temperature_2m,weather_code", "timezone": "auto"},
        headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if history:
        history.log("online.open_meteo", "historical weather", centre=[round(lat, 3), round(lon, 3)], date=date)
    h = j.get("hourly", {})
    return {"hourly": h, "timezone": j.get("timezone"), "source": "Open-Meteo.com (ERA5 reanalysis), CC BY 4.0",
            "note": "Reanalysis grid (~25 km) and hourly resolution, so local showers or gusts may not appear."}
