"""Global 0.5° land grid with country, coastal flag, mains frequency and language shares."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
R_EARTH_KM = 6371.0088


@lru_cache(maxsize=1)
def countries() -> dict:
    return json.loads((DATA / "countries.json").read_text(encoding="utf-8"))


class LandGrid:
    def __init__(self):
        meta = json.loads((DATA / "grid_meta.json").read_text(encoding="utf-8"))
        g = np.load(DATA / "land_grid.npy")
        coast = np.load(DATA / "coastal_grid.npy")
        self.res = meta["resolution_deg"]
        rows, cols = np.nonzero(g)
        self.lat = (meta["lat0"] - rows * self.res).astype(np.float32)
        self.lon = (meta["lon0"] + cols * self.res).astype(np.float32)
        iso = np.array(meta["iso_list"], dtype=object)
        self.iso = iso[g[rows, cols]]
        self.coastal = coast[rows, cols].astype(bool)
        self.n = len(self.lat)
        self.area_km2 = (111.32 * self.res) ** 2 * np.cos(np.radians(self.lat))
        c = countries()
        self.mains = np.array([str(c.get(i, {}).get("mains_hz", 50)) for i in self.iso], dtype=object)
        # Japan split (east 50 Hz / west 60 Hz) approximated by the Fuji/Itoigawa line
        jp = self.iso == "JP"
        east = jp & ((self.lon > 138.6) | ((self.lon > 137.5) & (self.lat > 36.8)))
        self.mains[jp] = "60"
        self.mains[east] = "50"
        self.mains = self.mains.astype(str)

    def language_share(self, code: str) -> np.ndarray:
        c = countries()
        lut = {k: v.get("languages", {}).get(code, 0.0) for k, v in c.items()}
        return np.array([lut.get(i, 0.0) for i in self.iso], np.float32)

    def in_country(self, iso2: str) -> np.ndarray:
        return self.iso == iso2


@lru_cache(maxsize=1)
def grid() -> LandGrid:
    return LandGrid()


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_grid(lat: float, lon: float, radius_km: float, step_km: float = 2.0):
    """Regular grid of points (lat, lon) within radius_km of a centre."""
    dlat = step_km / 111.32
    dlon = step_km / (111.32 * max(0.05, np.cos(np.radians(lat))))
    n_lat = int(radius_km / step_km)
    lats = lat + np.arange(-n_lat, n_lat + 1) * dlat
    lons = lon + np.arange(-n_lat, n_lat + 1) * dlon
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    LA, LO = LA.ravel(), LO.ravel()
    m = haversine_km(lat, lon, LA, LO) <= radius_km
    return LA[m].astype(np.float32), LO[m].astype(np.float32)
