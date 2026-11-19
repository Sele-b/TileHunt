"""buildmap: the self-contained interactive HTML. The page itself lives in map.html, map.css and map.js beside this file."""
from __future__ import annotations

from pathlib import Path

import argparse
import base64
import json
import math
import os
from typing import Sequence

import numpy as np

from ..layout import default_input
from ..tiles import TileGrid, largest_square, tile_bounds


# --------------------------------------------------------------------------- #
# command: map                                                                 #
# --------------------------------------------------------------------------- #

def douglas_peucker(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        den = math.hypot(dx, dy)
        best, bi = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            d = (abs(dy * (px - ax) - dx * (py - ay)) / den) if den else math.hypot(px - ax, py - ay)
            if d > best:
                best, bi = d, k
        if best > tol:
            keep[bi] = True
            stack.append((i, bi))
            stack.append((bi, j))
    return [p for p, k in zip(pts, keep) if k]


def delta_ints(pts: Sequence[tuple[float, float]]) -> list[int]:
    """Flat delta-encoded lon/lat at 1e-4 degree (~11 m) precision."""
    flat: list[int] = []
    lx = ly = 0
    for x, y in pts:
        ix, iy = int(round(x * 1e4)), int(round(y * 1e4))
        flat.append(ix - lx)
        flat.append(iy - ly)
        lx, ly = ix, iy
    return flat


def build_admin(path: str, tol: float, min_area: float = 0.0008) -> dict:
    """
    Outlines + label points for an administrative GeoJSON (regions,
    departements, ...). Expects `nom` / `code` properties, as published by
    gregoiredavid/france-geojson; falls back to name/NAME and code/CODE.
    """
    from shapely.geometry import shape

    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]

    rings: list[list[int]] = []
    labels: list[list] = []
    for ft in feats:
        props = ft.get("properties") or {}
        name = (props.get("nom") or props.get("name") or props.get("NAME") or "")
        code = str(props.get("code") or props.get("CODE") or "")
        try:
            geom = shape(ft["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        parts.sort(key=lambda p: p.area, reverse=True)
        for poly in parts:
            if poly.area < min_area:
                continue                      # drop islets and slivers
            pts = douglas_peucker([(round(x, 4), round(y, 4))
                                   for x, y in poly.exterior.coords], tol)
            if len(pts) >= 4:
                rings.append(delta_ints(pts))
        rp = parts[0].representative_point()  # guaranteed inside the polygon
        labels.append([name, code, round(rp.x, 4), round(rp.y, 4)])
    return {"rings": rings, "labels": labels}


def build_basemap(path: str, clip: tuple[float, float, float, float],
                  tol: float, highlight_iso: str | None) -> dict:
    """Simplified country outlines from a GeoJSON, as delta-encoded ints."""
    from shapely.geometry import shape, box
    clip_geom = box(*clip)
    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]

    rings: list[dict] = []
    for ft in feats:
        props = ft.get("properties") or {}
        iso = str(props.get("ISO_A3") or props.get("ADM0_A3") or "").upper()
        try:
            geom = shape(ft["geometry"])
        except Exception:
            continue
        if not geom.intersects(clip_geom):
            continue
        geom = geom.intersection(clip_geom)
        if geom.is_empty:
            continue
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.area < 0.002:
                continue
            pts = douglas_peucker([(round(x, 4), round(y, 4))
                                   for x, y in poly.exterior.coords], tol)
            if len(pts) < 4:
                continue
            rings.append({"h": 1 if (highlight_iso and iso == highlight_iso.upper()) else 0,
                          "d": delta_ints(pts)})
    return {"rings": rings}


FRENCH_CITIES = [
    # (name, lon, lat, tier). Tier 1 is labelled at every zoom; tier 2
    # appears once tiles are big enough to read. Prefecture positions come
    # from the commune polygons of gregoiredavid/france-geojson; the major
    # cities use their own centre coordinates.
    ("Ajaccio", 8.7369, 41.9192, 1),                # 2A
    ("Amiens", 2.2759, 49.8998, 1),                 # 80
    ("Angers", -0.5614, 47.4741, 1),                # 49
    ("Bayonne", -1.4748, 43.4929, 1),               # 64
    ("Besançon", 6.0241, 47.238, 1),                # 25
    ("Bordeaux", -0.5792, 44.8378, 1),              # 33
    ("Brest", -4.4861, 48.3904, 1),                 # 29
    ("Caen", -0.3708, 49.1829, 1),                  # 14
    ("Clermont-Ferrand", 3.087, 45.7772, 1),        # 63
    ("Dijon", 5.0415, 47.322, 1),                   # 21
    ("Grenoble", 5.7245, 45.1885, 1),               # 38
    ("Le Havre", 0.1079, 49.4944, 1),               # 76
    ("Lille", 3.0573, 50.6292, 1),                  # 59
    ("Limoges", 1.2611, 45.8336, 1),                # 87
    ("Lyon", 4.8357, 45.764, 1),                    # 69
    ("Marseille", 5.3698, 43.2965, 1),              # 13
    ("Metz", 6.2069, 49.1032, 1),                   # 57
    ("Montpellier", 3.8767, 43.6108, 1),            # 34
    ("Nancy", 6.1844, 48.6921, 1),                  # 54
    ("Nantes", -1.5536, 47.2184, 1),                # 44
    ("Nice", 7.262, 43.7102, 1),                    # 06
    ("Orléans", 1.9093, 47.9029, 1),                # 45
    ("Paris", 2.3522, 48.8566, 1),                  # 75
    ("Perpignan", 2.8954, 42.6887, 1),              # 66
    ("Poitiers", 0.366, 46.5812, 1),                # 86
    ("Reims", 4.0317, 49.2583, 1),                  # 51
    ("Rennes", -1.6778, 48.1173, 1),                # 35
    ("Rouen", 1.0908, 49.4445, 1),                  # 76
    ("Strasbourg", 7.7521, 48.5734, 1),             # 67
    ("Toulon", 5.9302, 43.1429, 1),                 # 83
    ("Toulouse", 1.4442, 43.6047, 1),               # 31
    ("Tours", 0.6848, 47.3941, 1),                  # 37
    ("Agen", 0.6271, 44.1965, 2),                   # 47
    ("Albi", 2.146, 43.9283, 2),                    # 81
    ("Alençon", 0.0864, 48.4284, 2),                # 61
    ("Angoulême", 0.1487, 45.6434, 2),              # 16
    ("Annecy", 6.1302, 45.9041, 2),                 # 74
    ("Arras", 2.7636, 50.2917, 2),                  # 62
    ("Auch", 0.6072, 43.6681, 2),                   # 32
    ("Aurillac", 2.4404, 44.9258, 2),               # 15
    ("Auxerre", 3.5803, 47.8004, 2),                # 89
    ("Avignon", 4.8374, 43.9451, 2),                # 84
    ("Bar-le-Duc", 5.1527, 48.7612, 2),             # 55
    ("Bastia", 9.4217, 42.6821, 2),                 # 2B
    ("Beauvais", 2.0737, 49.4435, 2),               # 60
    ("Belfort", 6.8526, 47.6464, 2),                # 90
    ("Biarritz", -1.553, 43.4743, 2),               # 64
    ("Blois", 1.3045, 47.5781, 2),                  # 41
    ("Bobigny", 2.4414, 48.9083, 2),                # 93
    ("Bourg-en-Bresse", 5.2413, 46.201, 2),         # 01
    ("Bourges", 2.4008, 47.0793, 2),                # 18
    ("Cahors", 1.4475, 44.4548, 2),                 # 46
    ("Calais", 1.8685, 50.9498, 2),                 # 62
    ("Carcassonne", 2.3423, 43.2043, 2),            # 11
    ("Chambéry", 5.9037, 45.5762, 2),               # 73
    ("Charleville-Mézières", 4.712, 49.7801, 2),    # 08
    ("Chartres", 1.5079, 48.4489, 2),               # 28
    ("Chaumont", 5.1198, 48.0843, 2),               # 52
    ("Châlons-en-Champagne", 4.3805, 48.9674, 2),   # 51
    ("Châteauroux", 1.7009, 46.8034, 2),            # 36
    ("Colmar", 7.3645, 48.1103, 2),                 # 68
    ("Créteil", 2.4525, 48.7814, 2),                # 94
    ("Digne-les-Bains", 6.2152, 44.1075, 2),        # 04
    ("Dunkerque", 2.3488, 51.0311, 2),              # 59
    ("Foix", 1.6022, 42.9694, 2),                   # 09
    ("Gap", 6.0775, 44.5799, 2),                    # 05
    ("Guéret", 1.8681, 46.1616, 2),                 # 23
    ("La Roche-sur-Yon", -1.4153, 46.6635, 2),      # 85
    ("La Rochelle", -1.1788, 46.1625, 2),           # 17
    ("Laon", 3.6136, 49.5745, 2),                   # 02
    ("Laval", -0.7662, 48.0635, 2),                 # 53
    ("Le Mans", 0.2036, 47.9821, 2),                # 72
    ("Le Puy-en-Velay", 3.902, 45.0259, 2),         # 43
    ("Lons-le-Saunier", 5.5556, 46.6711, 2),        # 39
    ("Melun", 2.6641, 48.5437, 2),                  # 77
    ("Mende", 3.4604, 44.5354, 2),                  # 48
    ("Mont-de-Marsan", -0.4868, 43.8928, 2),        # 40
    ("Montauban", 1.3677, 44.0217, 2),              # 82
    ("Moulins", 3.3262, 46.5616, 2),                # 03
    ("Mâcon", 4.8241, 46.3035, 2),                  # 71
    ("Nanterre", 2.2024, 48.8974, 2),               # 92
    ("Nevers", 3.1584, 46.9859, 2),                 # 58
    ("Niort", -0.463, 46.3337, 2),                  # 79
    ("Nîmes", 4.3516, 43.8326, 2),                  # 30
    ("Pau", -0.3406, 43.3231, 2),                   # 64
    ("Pontoise", 2.0947, 49.0534, 2),               # 95
    ("Privas", 4.5977, 44.7216, 2),                 # 07
    ("Périgueux", 0.7179, 45.1935, 2),              # 24
    ("Quimper", -4.1115, 47.9988, 2),               # 29
    ("Roanne", 4.083, 46.0467, 2),                  # 42
    ("Rodez", 2.5696, 44.3592, 2),                  # 12
    ("Saint-Brieuc", -2.7625, 48.51, 2),            # 22
    ("Saint-Lô", -1.0757, 49.1104, 2),              # 50
    ("Saint-Nazaire", -2.232, 47.279, 2),           # 44
    ("Saint-Étienne", 4.4158, 45.4222, 2),          # 42
    ("Tarbes", 0.0639, 43.2397, 2),                 # 65
    ("Troyes", 4.0812, 48.2932, 2),                 # 10
    ("Tulle", 1.7643, 45.2668, 2),                  # 19
    ("Valence", 4.9127, 44.9285, 2),                # 26
    ("Vannes", -2.744, 47.6611, 2),                 # 56
    ("Versailles", 2.1199, 48.8018, 2),             # 78
    ("Vesoul", 6.1524, 47.6329, 2),                 # 70
    ("Épinal", 6.4934, 48.1654, 2),                 # 88
    ("Évreux", 1.1413, 49.0191, 2),                 # 27
    ("Évry", 2.4451, 48.6282, 2),                   # 91
]


OSM_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap</a> contributors')
ESRI_ATTR = ('Tiles &copy; <a href="https://www.esri.com">Esri</a>, '
             'sources: Esri, HERE, Garmin, &copy; OpenStreetMap contributors')

# Key-free raster sources. Carto's light_all / dark_all used to be open and
# now answer with an "API KEY REQUIRED" watermark, so they are not listed:
# use --basemap-url if you have a Carto key.
BASEMAPS = {
    "osm": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "sub": "abc", "budget": 120, "light": True, "attr": OSM_ATTR,
    },
    # Esri's grey canvases. Note the {z}/{y}/{x} order, which the template
    # substitution handles on its own.
    "esri-gray": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
               "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "sub": "", "budget": 400, "light": True, "attr": ESRI_ATTR,
    },
    "esri-dark-gray": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
               "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "sub": "", "budget": 400, "light": False, "attr": ESRI_ATTR,
    },
}


def basemap_spec(a: argparse.Namespace) -> dict | None:
    """Resolve --basemap / --basemap-url into the viewer's basemap record."""
    if a.basemap_url:
        spec = {"url": a.basemap_url, "sub": a.basemap_sub or "",
                "budget": a.basemap_budget, "light": not a.basemap_dark,
                "attr": a.basemap_attr or OSM_ATTR}
    elif a.basemap in BASEMAPS:
        spec = dict(BASEMAPS[a.basemap])
        if a.basemap_dark:
            spec["light"] = False
        if a.basemap_attr:
            spec["attr"] = a.basemap_attr
    else:
        return None
    spec["grey"] = bool(a.basemap_grey)
    spec["opacity"] = float(a.basemap_opacity)
    spec["detail"] = float(a.basemap_detail)
    spec["clip"] = not a.basemap_everywhere
    return spec


def cmd_map(a: argparse.Namespace) -> None:
    if not a.out:
        raise SystemExit("give --out FILE, or --runname NAME to write "
                         "Runs/Square/NAME/NAME.html")
    # fall back to the conventional filenames sitting in the Data folder
    picked = []
    if not a.mask:
        a.mask = default_input("france.mask.npz", "mask.npz")
        if a.mask:
            picked.append("mask")
    if not a.boundary:
        a.boundary = default_input("countries.geojson",
                                   "ne_10m_admin_0_countries.geojson")
        if a.boundary:
            picked.append("boundary")
    if not a.regions:
        a.regions = default_input("regions.geojson")
        if a.regions:
            picked.append("regions")
    if not a.departments:
        a.departments = default_input("departements.geojson",
                                      "departments.geojson")
        if a.departments:
            picked.append("departments")
    if picked:
        print("auto from Data/: " + ", ".join(picked))

    roads = TileGrid.load(a.tiles)
    mask = TileGrid.load(a.mask) if a.mask else None

    if mask is not None:
        x0 = min(roads.x0, mask.x0)
        y0 = min(roads.y0, mask.y0)
        x1 = max(roads.x0 + roads.w, mask.x0 + mask.w)
        y1 = max(roads.y0 + roads.h, mask.y0 + mask.h)
        roads = roads.reframe(x0, y0, x1 - x0, y1 - y0)
        mask = mask.reframe(x0, y0, x1 - x0, y1 - y0)
        if a.clip_to_mask:
            # drop roads outside the area of interest (e.g. across the border)
            roads.bits &= mask.bits
            # and shrink the window to the mask's own extent
            rows = np.where(mask.bits.any(axis=1))[0]
            cols = np.where(mask.bits.any(axis=0))[0]
            if len(rows) and len(cols):
                nx0, ny0 = mask.x0 + int(cols[0]), mask.y0 + int(rows[0])
                nw = int(cols[-1] - cols[0]) + 1
                nh = int(rows[-1] - rows[0]) + 1
                roads = roads.reframe(nx0, ny0, nw, nh)
                mask = mask.reframe(nx0, ny0, nw, nh)
    else:
        x0, y0 = roads.x0, roads.y0

    base = {}
    if a.boundary:
        west, _, _, north = tile_bounds(roads.x0, roads.y0)
        _, south, east, _ = tile_bounds(roads.x0 + roads.w - 1, roads.y0 + roads.h - 1)
        pad = 2.0
        base = build_basemap(a.boundary, (west - pad, south - pad, east + pad, north + pad),
                             a.simplify, a.iso)

    admin = {}
    if a.regions:
        admin["regions"] = build_admin(a.regions, a.admin_simplify)
    if a.departments:
        admin["departments"] = build_admin(a.departments, a.admin_simplify)

    # road length per tile, quantised to 10 m and clamped to uint16
    km_b64 = None
    km_stats = None
    if roads.metres is not None:
        q = np.clip(np.rint(roads.metres / 10.0), 0, 65535).astype('<u2')
        km_b64 = base64.b64encode(q.tobytes()).decode()
        vals = roads.metres[roads.bits] / 1000.0
        if vals.size:
            km_stats = {
                "total": round(float(vals.sum()), 1),
                "max": round(float(vals.max()), 2),
                "p50": round(float(np.percentile(vals, 50)), 2),
                "p95": round(float(np.percentile(vals, 95)), 2),
                "p99": round(float(np.percentile(vals, 99)), 2),
            }
            # where the busiest tiles actually are, so the figure in the
            # panel is something you can jump to rather than hunt for
            flat = np.where(roads.bits, roads.metres, -1.0).ravel()
            top = np.argsort(flat)[::-1][:10]
            km_stats["top"] = [
                [int(i % roads.w) + roads.x0, int(i // roads.w) + roads.y0,
                 round(float(flat[i]) / 1000.0, 2)]
                for i in top if flat[i] > 0]

    route = None
    if a.route:
        with open(a.route, "r", encoding="utf-8") as fh:
            rj = json.load(fh)
        cs = rj.get("coords") or []
        flat, lx, ly = [], 0, 0
        for lo, la in cs:                 # delta ints at 1e-5 deg (about 1 m)
            ix, iy = int(round(lo * 1e5)), int(round(la * 1e5))
            flat.append(ix - lx); flat.append(iy - ly)
            lx, ly = ix, iy
        route = {"d": flat, "stats": rj.get("stats") or {},
                 "start": rj.get("start"), "end": rj.get("end"),
                 "rect": rj.get("rect"), "profile": rj.get("profile"),
                 "visits": rj.get("visits") or [], "mode": rj.get("mode", ""),
                 "pinned": rj.get("pinned"),
                 "skipped": rj.get("skipped"),
                 "wanted": rj.get("wanted")}

    green = roads.count
    # the tiles already ridden, reframed onto the same window as the roads so
    # the page can index all three grids with one set of coordinates
    visited = None
    vis_stats = None
    if getattr(a, "visited", None):
        v = TileGrid.load(a.visited)
        visited = v.reframe(roads.x0, roads.y0, roads.w, roads.h)
        have = int(visited.bits.sum())
        need = int((roads.bits & ~visited.bits).sum())
        if mask is not None:
            need = int((roads.bits & mask.bits & ~visited.bits).sum())
        if v.meta.get("max_square"):
            side, sqx, sqy = v.meta["max_square"][:3]
        else:
            side, bx_, by_ = largest_square(v.bits)
            sqx, sqy = v.x0 + bx_, v.y0 + by_
        vis_stats = {"have": have, "need": need,
                     "square": [int(side), int(sqx), int(sqy)],
                     "total": int(v.bits.sum()),
                     "meta": v.meta}
        print(f"visited     : {int(v.bits.sum()):,} tile(s) ridden, "
              f"{have:,} of them in this window, {need:,} still to get")
        print(f"max square  : {side} x {side} at {sqx},{sqy}")

    inscope = mask.count if mask is not None else green
    red = int((mask.bits & ~roads.bits).sum()) if mask is not None else 0

    data = {
        "z": roads.z, "x0": roads.x0, "y0": roads.y0, "w": roads.w, "h": roads.h,
        "road": roads.b64(),
        "km": km_b64,
        "kmScale": 0.01,          # stored unit: hundredths of a km (10 m)
        "kmStats": km_stats,
        "land": mask.b64() if mask is not None else None,
        "visited": visited.b64() if visited is not None else None,
        "visitedStats": vis_stats,
        "cities": [[n, lo, la, t] for n, lo, la, t in FRENCH_CITIES],
        "base": base,
        "admin": admin,
        "route": route,
        "cmd": {"tiles": os.path.basename(a.tiles),
                "db": os.path.basename(a.db_name or "Database_OSM"),
                "py": a.python_cmd,
                "args": a.route_args,
                "dem": a.dem_name},
        "stats": {"green": green, "red": red, "inscope": inscope,
                  "pct": round(100.0 * green / inscope, 2) if inscope else 0.0},
        "meta": roads.meta,
        "title": a.title,
        "note": a.note or "",
        "basemap": basemap_spec(a),
        "tileAlpha": float(a.tile_opacity) if a.tile_opacity is not None
                     else (0.55 if (a.basemap in BASEMAPS or a.basemap_url)
                           else 0.8),
    }

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    html = (VIEWER_HTML
            .replace("__TITLE__", esc(a.title))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))
                     .replace("</", "<\\/")))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    size = os.path.getsize(a.out)
    print(f"green tiles : {green:,}")
    if mask is not None:
        print(f"red tiles   : {red:,}   (in scope: {inscope:,}, "
              f"{100.0*green/max(inscope,1):.1f}% covered)")
    if route is not None:
        st = route["stats"]
        print(f"route       : {st.get('km','?')} km, "
              f"{st.get('covered','?')}/{st.get('tiles','?')} tiles")
    print(f"written     : {a.out}  ({size/1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# the viewer                                                                   #
# --------------------------------------------------------------------------- #

# The page lives beside this file as map.html, map.css and map.js, where an
# editor can colour and check them. They are inlined back together here
# because the map must stay one self-contained file that opens from disk.
_VIEWER = Path(__file__).resolve().parent
_CSS_MARK = "/*@inline map.css*/"
_JS_MARK = "/*@inline map.js*/"


def _load_viewer() -> str:
    def read(name: str) -> str:
        return (_VIEWER / name).read_text(encoding="utf-8")
    html = read("map.html")
    if html.count(_CSS_MARK) != 1 or html.count(_JS_MARK) != 1:
        raise SystemExit("viewer/map.html has lost its inline markers")
    head, rest = html.split(_CSS_MARK)
    mid, tail = rest.split(_JS_MARK)
    return head + read("map.css") + mid + read("map.js") + tail


VIEWER_HTML = _load_viewer()
