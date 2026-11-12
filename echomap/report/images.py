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


CLUE_RGB = [(232, 137, 214), (240, 147, 124), (92, 195, 199), (169, 156, 240), (111, 177, 240), (195, 216, 106), (242, 196, 90),
            (229, 115, 115), (127, 200, 169), (179, 157, 219)]   # pastel, as in the app's light map


def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _countries_hi():
    """Country polygons: Natural Earth 1:10m when bundled (v1.0.3+), else 1:50m."""
    import json
    from pathlib import Path
    from ..geo.topo import features
    p = Path(__file__).resolve().parents[1] / "web" / "vendor" / "countries-10m.json"
    if p.exists():
        if not hasattr(_countries_hi, "_c"):
            _countries_hi._c = features(json.loads(p.read_text(encoding="utf-8")), "countries")
        return _countries_hi._c
    return world_countries()


def _rings(ring):
    """Yield a ring once, or (if it crosses the antimeridian) unwrapped and shifted copies, so no stray horizontal spans."""
    if not any(abs(b[0] - a[0]) > 180 for a, b in zip(ring, ring[1:])):
        yield ring
        return
    out, off = [ring[0]], 0.0
    for a, b in zip(ring, ring[1:]):
        if b[0] - a[0] > 180:
            off -= 360
        elif a[0] - b[0] > 180:
            off += 360
        out.append((b[0] + off, b[1]))
    for shift in (0, 360, -360):
        yield [(x + shift, y) for x, y in out]


def default_view(geo: dict) -> tuple:
    """Bounding box (lon0, lat0, lon1, lat1) around the strongest places, else the best estimate circle, else the world."""
    P = [q for q in (geo.get("likely_places") or {}).get("places", []) if q["delta_to_best_nats"] <= 1.0 or q["share"] >= 0.03][:8]
    if P:
        pts = [(q["lon"], q["lat"], max(q["radius_km"] * 3, 25)) for q in P]
    elif geo.get("candidates"):
        c = geo["candidates"][0]
        pts = [(c["centre"]["lon"], c["centre"]["lat"], c["uncertainty_radius_km"] * 1.3)]
    else:
        return (-180, -60, 180, 85)
    xs, ys = [], []
    for lon, lat, r in pts:
        for b in (0, 90, 180, 270):
            la, lo = _destination(lat, lon, b, r)
            xs.append(lo)
            ys.append(la)
    x0, x1, y0, y1 = min(xs), max(xs), max(-85, min(ys)), min(85, max(ys))
    k = max(0.2, math.cos(math.radians((y0 + y1) / 2)))
    w, h = (x1 - x0) * k, y1 - y0
    if w < h * 1.6:  # landscape framing
        cx, half = (x0 + x1) / 2, h * 1.6 / k / 2
        x0, x1 = cx - half, cx + half
    else:
        cy, half = (y0 + y1) / 2, w / 1.6 / 2
        y0, y1 = max(-85, cy - half), min(85, cy + half)
    pad, padx = (y1 - y0) * 0.12, (x1 - x0) * 0.06
    return (max(-180, x0 - padx), max(-85, y0 - pad * 0.5), min(180, x1 + padx), min(85, y1 + pad * 1.5))


def clue_map_png(geo: dict, width: int = 1400, bbox=None, title: str | None = None) -> str | None:
    """Clue map drawn as vectors: country clues with exact borders, other areas smoothed, point clues, estimate
    circles and likely places as geodesic circles. Rendered at 2x and downsampled for clean edges."""
    import base64 as b64
    from PIL import ImageFilter
    fps = [f for f in geo.get("footprints", []) if f.get("used")]
    places = (geo.get("likely_places") or {}).get("places", [])
    if not fps and not places:
        return None
    x0, y0, x1, y1 = bbox or (-180, -60, 180, 85)
    kx = max(0.2, math.cos(math.radians((y0 + y1) / 2))) if bbox else 1.0
    S = 2
    W = width * S
    H = int(np.clip(W * (y1 - y0) / ((x1 - x0) * kx), 300 * S, 1100 * S))
    img = Image.new("RGB", (W, H), (223, 232, 241))
    d = ImageDraw.Draw(img, "RGBA")   # RGB base + RGBA drawing = alpha blending

    def px(lon, lat):
        return ((lon - x0) / (x1 - x0) * W, (y1 - lat) / (y1 - y0) * H)
    cs = _countries_hi() if bbox and (x1 - x0) < 90 else world_countries()
    for f in cs:
        for poly in f["polygons"]:
            if len(poly[0]) >= 3:
                for r in _rings(poly[0]):
                    d.polygon([px(lo, la) for lo, la in r], fill=(247, 248, 250, 255), outline=(195, 204, 216, 255))
    by_num = {}
    for f in cs:
        if f["id"] is not None:
            by_num.setdefault(str(f["id"]).zfill(3), []).append(f)
    deg_px = W / (x1 - x0)

    def overlay(draw_fn, alpha_fill, color, outline=True):
        dark = tuple(int(c * 0.7) for c in color)
        mask = Image.new("L", (W, H), 0)
        draw_fn(ImageDraw.Draw(mask))
        cover = float(np.asarray(mask, np.float32).mean() / 255.0)
        if bbox and cover > 0.55:   # a clue that fills most of a zoomed view would only tint everything: lighter fill
            alpha_fill *= 0.35
        img.paste(Image.new("RGB", (W, H), color), (0, 0), mask.point(lambda v: int(v * alpha_fill)))
        if outline:
            edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
            img.paste(Image.new("RGB", (W, H), dark), (0, 0), edge.point(lambda v: 200 if v > 40 else 0))

    def circle_pts(lat, lon, r_km, n=96):
        out = []
        for b in np.linspace(0, 360, n, endpoint=False):
            la, lo = _destination(lat, lon, float(b), r_km)
            out.append(px(lo, la))
        return out

    legend = []
    for i, f in enumerate(fps):
        col = CLUE_RGB[i % len(CLUE_RGB)]
        shape = f.get("shape") or {"kind": "raster"}
        legend.append((col, f["label"]))
        if shape["kind"] == "countries" and by_num:
            def fn(md, shape=shape):
                for n in shape.get("numeric") or []:
                    for ft in by_num.get(str(n).zfill(3), []):
                        for poly in ft["polygons"]:
                            if len(poly[0]) >= 3:
                                for r in _rings(poly[0]):
                                    md.polygon([px(lo, la) for lo, la in r], fill=255)
                                for hole in poly[1:]:
                                    if len(hole) >= 3:
                                        for r in _rings(hole):
                                            md.polygon([px(lo, la) for lo, la in r], fill=0)
            overlay(fn, 0.42, col)
        elif shape["kind"] == "points":
            def fn(md, shape=shape):
                for q in shape["points"]:
                    if x0 - 2 <= q["lon"] <= x1 + 2 and y0 - 2 <= q["lat"] <= y1 + 2:
                        pts = circle_pts(q["lat"], q["lon"], max(q.get("r", 3), 2))
                        xs = [p[0] for p in pts]
                        if max(xs) - min(xs) < 5 * S:
                            cx, cy = px(q["lon"], q["lat"])
                            md.ellipse([cx - 3 * S, cy - 3 * S, cx + 3 * S, cy + 3 * S], fill=255)
                        else:
                            md.polygon(pts, fill=255)
            overlay(fn, 0.55, col)
        else:
            bits = np.unpackbits(np.frombuffer(b64.b64decode(f["bits"]), np.uint8))[:720 * 360].reshape(360, 720) * 255
            src = Image.fromarray(bits.astype(np.uint8), "L")
            box = ((x0 + 180) * 2, (90 - y1) * 2, (x1 + 180) * 2, (90 - y0) * 2)
            m = src.transform((W, H), Image.Transform.EXTENT, box, Image.Resampling.BILINEAR)
            m = m.filter(ImageFilter.GaussianBlur(max(1.0, deg_px * 0.25))).point(lambda v: 255 if v > 110 else 0)

            def fn(md, m=m):
                md.bitmap((0, 0), m, fill=255)
            overlay(fn, 0.42, col)
    f11, f13, f15 = _font(11 * S), _font(13 * S), _font(15 * S)
    if bbox:
        from ..geo.gazetteer import major_cities
        span = x1 - x0
        min_pop = 3e6 if span > 60 else 8e5 if span > 25 else 2.5e5 if span > 8 else 6e4
        fc = _font(11 * S)
        for nm, la, lo, pop in major_cities(int(min_pop)):
            if x0 <= lo <= x1 and y0 <= la <= y1 and not any(
                    abs(q["lat"] - la) < 0.25 and abs(q["lon"] - lo) < 0.4 for q in places[:8]):
                cx, cy = px(lo, la)
                d.ellipse([cx - 2.5 * S, cy - 2.5 * S, cx + 2.5 * S, cy + 2.5 * S], fill=(107, 119, 140, 255))
                d.text((cx + 5 * S, cy - 9 * S), nm, fill=(75, 87, 107, 255), font=fc, stroke_width=S, stroke_fill=(255, 255, 255, 220))

    def label(xy, text, fill, font):
        d.text(xy, text, fill=fill, font=font, stroke_width=S + 1, stroke_fill=(255, 255, 255, 235))
    for c in geo.get("candidates", [])[:3]:
        if not places or c["rank"] == 1:
            pts = circle_pts(c["centre"]["lat"], c["centre"]["lon"], c["uncertainty_radius_km"], 180)
            col = (31, 102, 229) if c["rank"] == 1 else (125, 147, 184)
            for j in range(0, len(pts), 2):   # dashed ring
                d.line([pts[j], pts[(j + 1) % len(pts)]], fill=col + (255,), width=2 * S)
            cx, cy = px(c["centre"]["lon"], c["centre"]["lat"])
            label((cx + 8 * S, cy - 18 * S), f"Estimate #{c['rank']}: {c['name']} (±{c['uncertainty_radius_km']:g} km)", (22, 41, 77, 255), f11)
    for q in reversed([q for q in places[:8] if q["share"] >= 0.005 or q["rank"] <= 3]):
        pts = circle_pts(q["lat"], q["lon"], max(q["radius_km"], 3), 120)
        xs = [p[0] for p in pts]
        cx, cy = px(q["lon"], q["lat"])
        top = q["rank"] == 1
        col = (15, 61, 153) if top else (31, 102, 229)
        if max(xs) - min(xs) < 12 * S:
            r = (9 if top else 6) * S
            pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 48)]
            xs = [p[0] for p in pts]
        d.polygon(pts, fill=col + (120 if top else 70,))
        d.line(pts + [pts[0]], fill=col + (255,), width=(3 if top else 2) * S)
        if not bbox and not top:
            continue
        label((max(xs) + 6 * S, cy - 10 * S), f"{q['rank']}. {q['name']}  {q['share'] * 100:.0f}%",
              (15, 61, 153, 255) if top else (22, 41, 77, 255), f15 if top else f13)
    out = img.resize((width, H // S), Image.Resampling.LANCZOS)
    d2 = ImageDraw.Draw(out, "RGBA")
    y = 8
    rows = [((15, 61, 153), "Likely places (circle = place size, % = relative share)")] if places else []
    rows += legend[:10]
    if title:
        d2.rectangle([0, 0, width, 26], fill=(255, 255, 255, 230))
        d2.text((10, 4), title, fill=(22, 41, 77, 255), font=_font(15))
        y = 34
    d2.rectangle([6, y - 4, 470, y + 17 * len(rows) + 2], fill=(255, 255, 255, 220), outline=(216, 225, 238, 255))
    for col, text in rows:
        d2.ellipse([12, y, 24, y + 12], fill=col + (255,), outline=(120, 130, 150, 255))
        d2.text((30, y - 2), text[:66], fill=(24, 40, 63, 255), font=_font(12))
        y += 17
    d2.text((8, out.height - 18), "Inferred regions, not measured locations. Country clues use exact borders; other areas are smoothed "
            "0.5° cells; shares are relative, not calibrated.", fill=(70, 85, 110, 255), font=_font(11))
    return png_b64(out)


def clue_map_views(geo: dict, width: int = 1400) -> list[tuple[str, str]]:
    """World view plus a zoomed view of the strongest places or estimate (for reports and examples)."""
    out = []
    w = clue_map_png(geo, width, None, "World view")
    if w:
        out.append(("world", w))
    if (geo.get("likely_places") or {}).get("places") or geo.get("candidates"):
        z = clue_map_png(geo, width, default_view(geo), "Zoomed to the most likely area")
        if z:
            out.append(("zoom", z))
    return out
