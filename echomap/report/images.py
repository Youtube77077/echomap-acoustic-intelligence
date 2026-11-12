"""Static images for the report: spectrogram and map (Pillow)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw

from ..geo.topo import world_countries

# purple -> blue -> yellow -> orange -> red  (same stops as the interactive UI)
STOPS = [(0.0, (10, 4, 30)), (0.18, (68, 20, 120)), (0.38, (40, 80, 200)), (0.6, (245, 225, 60)),
         (0.8, (245, 140, 40)), (1.0, (215, 25, 28))]


def palette_lut() -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255
        for (t0, c0), (t1, c1) in zip(STOPS[:-1], STOPS[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                lut[i] = [round(c0[k] + f * (c1[k] - c0[k])) for k in range(3)]
                break
    return lut


def png_b64(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def spectrogram_png(view: dict) -> str:
    rgb = palette_lut()[view["data"]]
    return png_b64(Image.fromarray(rgb, "RGB"))


def _destination(lat, lon, bearing, dist_km):
    R = 6371.0088
    la1, lo1, b = math.radians(lat), math.radians(lon), math.radians(bearing)
    d = dist_km / R
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540) % 360 - 180


def map_png(geo: dict, width: int = 1400, bbox=None) -> str:
    """Equirectangular map with countries, plausibility cells, and geodesic uncertainty circles."""
    cands = geo.get("candidates", [])
    if bbox is None:
        if cands:
            c = cands[0]
            r_deg = max(3.0, c["uncertainty_radius_km"] / 111.0 * 3)
            lats = [k["centre"]["lat"] for k in cands]
            lons = [k["centre"]["lon"] for k in cands]
            bbox = (max(-180, min(lons) - r_deg * 1.5), max(-85, min(lats) - r_deg),
                    min(180, max(lons) + r_deg * 1.5), min(85, max(lats) + r_deg))
            if bbox[2] - bbox[0] < 8:
                cx = (bbox[0] + bbox[2]) / 2
                bbox = (cx - 4, bbox[1], cx + 4, bbox[3])
        else:
            bbox = (-180, -60, 180, 85)
    x0, y0, x1, y1 = bbox
    height = int(width * (y1 - y0) / (x1 - x0) / max(0.35, math.cos(math.radians((y0 + y1) / 2))) * 1.0)
    height = int(np.clip(height, 300, 1000))
    img = Image.new("RGB", (width, height), (14, 22, 36))
    d = ImageDraw.Draw(img, "RGBA")

    def px(lon, lat):
        return ((lon - x0) / (x1 - x0) * width, (y1 - lat) / (y1 - y0) * height)

    for f in world_countries():
        for poly in f["polygons"]:
            ring = poly[0]
            if len(ring) < 3:
                continue
            pts = [px(lo, la) for lo, la in ring]
            d.polygon(pts, fill=(30, 42, 60, 255), outline=(90, 110, 140, 255))
    cell = geo.get("heat_cell_deg", 0.5)
    for la, lo, v in geo.get("heat", []):
        if x0 - cell <= lo <= x1 + cell and y0 - cell <= la <= y1 + cell:
            a, b = px(lo - cell / 2, la + cell / 2)
            c2, e = px(lo + cell / 2, la - cell / 2)
            col = palette_lut()[int(80 + 175 * v)]
            d.rectangle([a, b, c2, e], fill=(int(col[0]), int(col[1]), int(col[2]), int(60 + 150 * v)))
    for k in reversed(cands):
        ring = [_destination(k["centre"]["lat"], k["centre"]["lon"], br, k["uncertainty_radius_km"]) for br in range(0, 361, 4)]
        pts = [px(lo, la) for la, lo in ring]
        col = (255, 90, 60) if k["rank"] == 1 else (255, 200, 80) if k["strength"] != "weaker" else (170, 180, 200)
        d.line(pts, fill=col + (255,), width=3 if k["rank"] == 1 else 2)
        cx, cy = px(k["centre"]["lon"], k["centre"]["lat"])
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col + (255,))
        d.text((cx + 8, cy - 8), f"#{k['rank']}", fill=(255, 255, 255, 255))
    d.text((8, height - 18), "Natural Earth boundaries; circles are geodesic uncertainty radii; heuristic relative plausibility",
           fill=(200, 210, 225, 255))
    return png_b64(img)


CLUE_RGB = [(79, 140, 255), (255, 179, 64), (46, 211, 183), (255, 107, 154), (199, 125, 255), (123, 216, 143), (245, 225, 60),
            (255, 138, 76), (77, 208, 225), (229, 115, 115)]


def clue_map_png(geo: dict, width: int = 1400) -> str | None:
    """Equirectangular clue map: each locatable clue's footprint as a coloured region with a white edge."""
    import base64 as b64
    fps = [f for f in geo.get("footprints", []) if f.get("used")]
    if not fps:
        return None
    H = width // 2
    img = Image.new("RGB", (width, H), (14, 20, 32))
    d = ImageDraw.Draw(img, "RGBA")

    def px(lon, lat):
        return ((lon + 180) / 360 * width, (90 - lat) / 180 * H)
    for f in world_countries():
        for poly in f["polygons"]:
            if len(poly[0]) >= 3:
                d.polygon([px(lo, la) for lo, la in poly[0]], fill=(38, 48, 63, 255), outline=(70, 84, 104, 255))
    cw = width / 720
    count = np.zeros(720 * 360, np.uint8)
    for i, f in enumerate(fps):
        bits = np.unpackbits(np.frombuffer(b64.b64decode(f["bits"]), np.uint8))[:720 * 360]
        count += bits
        col = CLUE_RGB[i % len(CLUE_RGB)]
        grid = bits.reshape(360, 720)
        rows, cols = np.nonzero(grid)
        for r, c in zip(rows, cols):
            d.rectangle([c * cw, r * cw, (c + 1) * cw, (r + 1) * cw], fill=col + (70,))
        edge = grid & ~(np.roll(grid, 1, 0) & np.roll(grid, -1, 0) & np.roll(grid, 1, 1) & np.roll(grid, -1, 1))
        for r, c in zip(*np.nonzero(edge)):
            d.rectangle([c * cw, r * cw, (c + 1) * cw, (r + 1) * cw], fill=(255, 255, 255, 140))
    mx = int(count.max())
    if mx >= 2:
        for r, c in zip(*np.nonzero(count.reshape(360, 720) >= mx)):
            d.rectangle([c * cw, r * cw, (c + 1) * cw, (r + 1) * cw], outline=(255, 90, 60, 255))
    y = 6
    for i, f in enumerate(fps[:12]):
        col = CLUE_RGB[i % len(CLUE_RGB)]
        d.rectangle([8, y, 20, y + 12], fill=col + (255,), outline=(255, 255, 255, 255))
        d.text((26, y), f["label"][:70], fill=(230, 235, 245, 255))
        y += 16
    if mx >= 2:
        d.text((8, H - 18), f"Red cells: area where the most clues overlap ({mx} of {len(fps)}). Footprints are approximate 0.5° cells.",
               fill=(255, 200, 190, 255))
    return png_b64(img)
