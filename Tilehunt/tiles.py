"""Explorer-tile maths: lon/lat <-> z14 tiles, exact segment traversal, the TileGrid bitmap, and the grid commands."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# tile math                                                                    #
# --------------------------------------------------------------------------- #

ZOOM = 14                      # VeloViewer / Statshunters explorer tiles
NTILES = 1 << ZOOM             # 16384
MAX_LAT = 85.0511287798066     # web-mercator limit


def lonlat_to_tilef(lon: float, lat: float, z: int = ZOOM) -> tuple[float, float]:
    """Longitude/latitude -> fractional tile coordinates (x right, y down)."""
    n = 1 << z
    if lat > MAX_LAT:
        lat = MAX_LAT
    elif lat < -MAX_LAT:
        lat = -MAX_LAT
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1.0 + s) / (1.0 - s)) / (4.0 * math.pi)) * n
    return x, y


def tilef_to_lonlat(x: float, y: float, z: int = ZOOM) -> tuple[float, float]:
    """Fractional tile coordinates -> longitude/latitude."""
    n = 1 << z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def tile_bounds(x: int, y: int, z: int = ZOOM) -> tuple[float, float, float, float]:
    """(west, south, east, north) of one tile, in degrees."""
    w, n = tilef_to_lonlat(x, y, z)
    e, s = tilef_to_lonlat(x + 1, y + 1, z)
    return w, s, e, n


def tile_size_m(y: int, z: int = ZOOM) -> tuple[float, float]:
    """Approximate width/height of a tile in metres, at tile row y."""
    w, s, e, n = tile_bounds(0, y, z)
    midlat = math.radians((s + n) * 0.5)
    width = (e - w) * 111_320.0 * math.cos(midlat)
    height = (n - s) * 110_574.0
    return width, height


# --------------------------------------------------------------------------- #
# exact segment -> tile traversal                                              #
# --------------------------------------------------------------------------- #

def walk_segment(x0: float, y0: float, x1: float, y1: float) -> Iterator[tuple[int, int]]:
    """
    Yield every integer cell the segment (x0,y0)->(x1,y1) touches, in order.

    Amanatides-Woo style grid traversal ("supercover"): a cell is yielded if the
    open segment intersects the cell's interior, so corner-clipping cells are
    included. Coordinates are fractional tile units.
    """
    cx, cy = math.floor(x0), math.floor(y0)
    yield cx, cy

    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        return

    ex, ey = math.floor(x1), math.floor(y1)
    if cx == ex and cy == ey:
        return

    if dx > 0.0:
        step_x, t_max_x, t_delta_x = 1, (cx + 1.0 - x0) / dx, 1.0 / dx
    elif dx < 0.0:
        step_x, t_max_x, t_delta_x = -1, (cx - x0) / dx, -1.0 / dx
    else:
        step_x, t_max_x, t_delta_x = 0, math.inf, math.inf

    if dy > 0.0:
        step_y, t_max_y, t_delta_y = 1, (cy + 1.0 - y0) / dy, 1.0 / dy
    elif dy < 0.0:
        step_y, t_max_y, t_delta_y = -1, (cy - y0) / dy, -1.0 / dy
    else:
        step_y, t_max_y, t_delta_y = 0, math.inf, math.inf

    # guard against pathological float input; a segment can never cross more
    # cells than its extent in cells, plus slack
    guard = 4 + int(abs(dx)) + int(abs(dy)) + abs(ex - cx) + abs(ey - cy)
    while guard > 0:
        guard -= 1
        if t_max_x <= t_max_y:
            if t_max_x > 1.0:
                break
            cx += step_x
            t_max_x += t_delta_x
        else:
            if t_max_y > 1.0:
                break
            cy += step_y
            t_max_y += t_delta_y
        yield cx, cy
        if cx == ex and cy == ey:
            break


EARTH_CIRCUMFERENCE_M = 40075016.686


def walk_segment_frac(x0: float, y0: float, x1: float, y1: float):
    """
    Like walk_segment, but yields (cx, cy, frac) where frac is the share of the
    segment's length lying inside that cell. The fracs sum to 1.
    """
    cx, cy = math.floor(x0), math.floor(y0)
    dx, dy = x1 - x0, y1 - y0
    if dx == 0.0 and dy == 0.0:
        yield cx, cy, 1.0
        return
    ex, ey = math.floor(x1), math.floor(y1)
    if cx == ex and cy == ey:
        yield cx, cy, 1.0
        return

    if dx > 0.0:
        step_x, t_max_x, t_delta_x = 1, (cx + 1.0 - x0) / dx, 1.0 / dx
    elif dx < 0.0:
        step_x, t_max_x, t_delta_x = -1, (cx - x0) / dx, -1.0 / dx
    else:
        step_x, t_max_x, t_delta_x = 0, math.inf, math.inf
    if dy > 0.0:
        step_y, t_max_y, t_delta_y = 1, (cy + 1.0 - y0) / dy, 1.0 / dy
    elif dy < 0.0:
        step_y, t_max_y, t_delta_y = -1, (cy - y0) / dy, -1.0 / dy
    else:
        step_y, t_max_y, t_delta_y = 0, math.inf, math.inf

    t_prev = 0.0
    guard = 4 + int(abs(dx)) + int(abs(dy)) + abs(ex - cx) + abs(ey - cy)
    while guard > 0:
        guard -= 1
        t_cross = t_max_x if t_max_x <= t_max_y else t_max_y
        if t_cross > 1.0:
            break
        if t_cross > t_prev:
            yield cx, cy, t_cross - t_prev
            t_prev = t_cross
        if t_max_x <= t_max_y:
            cx += step_x
            t_max_x += t_delta_x
        else:
            cy += step_y
            t_max_y += t_delta_y
        if cx == ex and cy == ey:
            break
    if t_prev < 1.0:
        yield cx, cy, 1.0 - t_prev


def walk_polyline(coords: Sequence[tuple[float, float]], out: set[tuple[int, int]],
                  acc: dict[tuple[int, int], float] | None = None) -> None:
    """Add every tile touched by a lon/lat polyline to `out`."""
    if not coords:
        return
    plon, plat = coords[0]
    px, py = lonlat_to_tilef(plon, plat)
    out.add((math.floor(px), math.floor(py)))
    for lon, lat in coords[1:]:
        qx, qy = lonlat_to_tilef(lon, lat)
        same = (math.floor(px) == math.floor(qx)
                and math.floor(py) == math.floor(qy))
        if acc is None:
            # fast path: both endpoints in the same tile (true for the vast
            # majority of OSM segments, which are far shorter than 1.5 km)
            if not same:
                for cell in walk_segment(px, py, qx, qy):
                    out.add(cell)
        else:
            # Web Mercator is conformal, so one tile unit is the same ground
            # distance in x and y; it just shrinks with latitude.
            mid_lat = (plat + lat) * 0.5
            m_per_unit = (EARTH_CIRCUMFERENCE_M
                          * math.cos(math.radians(mid_lat)) / NTILES)
            seg_m = math.hypot(qx - px, qy - py) * m_per_unit
            if same:
                cell = (math.floor(px), math.floor(py))
                acc[cell] = acc.get(cell, 0.0) + seg_m
            else:
                for cx, cy, frac in walk_segment_frac(px, py, qx, qy):
                    out.add((cx, cy))
                    acc[(cx, cy)] = acc.get((cx, cy), 0.0) + seg_m * frac
        px, py, plon, plat = qx, qy, lon, lat


# --------------------------------------------------------------------------- #
# tile sets / bitmaps                                                          #
# --------------------------------------------------------------------------- #

class TileGrid:
    """A rectangular window of the z14 grid, with one bit per tile."""

    def __init__(self, x0: int, y0: int, w: int, h: int, z: int = ZOOM,
                 bits: np.ndarray | None = None, meta: dict | None = None,
                 metres: np.ndarray | None = None):
        self.x0, self.y0, self.w, self.h, self.z = int(x0), int(y0), int(w), int(h), int(z)
        self.bits = np.zeros((self.h, self.w), dtype=bool) if bits is None else bits
        self.meta = meta or {}
        # metres of accepted road inside each tile, or None if not measured
        self.metres = metres

    @classmethod
    def from_tiles(cls, tiles: Iterable[tuple[int, int]], pad: int = 0,
                   z: int = ZOOM, meta: dict | None = None,
                   lengths: dict[tuple[int, int], float] | None = None) -> "TileGrid":
        tiles = list(tiles)
        if not tiles:
            raise SystemExit("no tiles to build a grid from")
        xs = np.fromiter((t[0] for t in tiles), dtype=np.int64, count=len(tiles))
        ys = np.fromiter((t[1] for t in tiles), dtype=np.int64, count=len(tiles))
        x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
        y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
        g = cls(x0, y0, x1 - x0 + 1, y1 - y0 + 1, z, meta=meta)
        g.bits[ys - y0, xs - x0] = True
        if lengths:
            g.metres = np.zeros((g.h, g.w), dtype=np.float32)
            for (tx, ty), m in lengths.items():
                if x0 <= tx <= x1 and y0 <= ty <= y1:
                    g.metres[ty - y0, tx - x0] = m
        return g

    def reframe(self, x0: int, y0: int, w: int, h: int) -> "TileGrid":
        """Return this grid resampled onto another window (clipped, zero-filled)."""
        out = TileGrid(x0, y0, w, h, self.z, meta=dict(self.meta))
        if self.metres is not None:
            out.metres = np.zeros((h, w), dtype=np.float32)
        sx0, sy0 = max(x0, self.x0), max(y0, self.y0)
        sx1, sy1 = min(x0 + w, self.x0 + self.w), min(y0 + h, self.y0 + self.h)
        if sx1 > sx0 and sy1 > sy0:
            dst = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0))
            src = (slice(sy0 - self.y0, sy1 - self.y0),
                   slice(sx0 - self.x0, sx1 - self.x0))
            out.bits[dst] = self.bits[src]
            if self.metres is not None:
                out.metres[dst] = self.metres[src]
        return out

    @property
    def count(self) -> int:
        return int(self.bits.sum())

    def save(self, path: str) -> None:
        arrays = dict(
            packed=np.packbits(self.bits.reshape(-1)),
            frame=np.array([self.x0, self.y0, self.w, self.h, self.z], dtype=np.int64),
            meta=np.array(json.dumps(self.meta)),
        )
        if self.metres is not None:
            arrays["metres"] = self.metres.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "TileGrid":
        with np.load(path, allow_pickle=False) as z:
            x0, y0, w, h, zz = (int(v) for v in z["frame"])
            n = w * h
            bits = np.unpackbits(z["packed"])[:n].astype(bool).reshape(h, w)
            meta = json.loads(str(z["meta"]))
            metres = z["metres"].reshape(h, w) if "metres" in z.files else None
        return cls(x0, y0, w, h, zz, bits, meta, metres)

    def b64(self) -> str:
        return base64.b64encode(np.packbits(self.bits.reshape(-1)).tobytes()).decode()


# --------------------------------------------------------------------------- #
# command: mask                                                                #
# --------------------------------------------------------------------------- #

def cmd_mask(a: argparse.Namespace) -> None:
    from .layout import default_input
    if not a.boundary:
        a.boundary = default_input("countries.geojson",
                                   "ne_10m_admin_0_countries.geojson")
        if a.boundary:
            print("auto from Data/: " + os.path.basename(a.boundary))
        else:
            raise SystemExit("give --boundary, or put countries.geojson in Data/")
    try:
        from shapely.geometry import shape, box
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        raise SystemExit("shapely is required for `mask`:  pip install shapely")

    with open(a.boundary, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]

    picked = []
    for ft in feats:
        props = ft.get("properties") or {}
        if a.iso:
            vals = {str(props.get(k, "")).upper()
                    for k in ("ISO_A3", "ADM0_A3", "iso_a3", "adm0_a3",
                              "SOV_A3", "ISO_A3_EH", "GU_A3")}
            if a.iso.upper() not in vals:
                continue
        geom = shape(ft["geometry"])
        picked.append(geom)
    if not picked:
        raise SystemExit(f"no feature matched iso={a.iso!r} in {a.boundary}")

    geom = unary_union(picked)

    # keep only the parts inside the area of interest (drops overseas France)
    if a.clip:
        w, s, e, n = a.clip
        geom = geom.intersection(box(w, s, e, n))
    if geom.is_empty:
        raise SystemExit("boundary is empty after --clip")

    # minimum part area, in square degrees, to drop islets
    if a.min_part_area > 0 and geom.geom_type == "MultiPolygon":
        parts = [g for g in geom.geoms if g.area >= a.min_part_area]
        if parts:
            geom = unary_union(parts)

    w, s, e, n = geom.bounds
    tx0, ty0 = lonlat_to_tilef(w, n)
    tx1, ty1 = lonlat_to_tilef(e, s)
    x0, y0 = math.floor(tx0), math.floor(ty0)
    x1, y1 = math.floor(tx1), math.floor(ty1)
    grid = TileGrid(x0, y0, x1 - x0 + 1, y1 - y0 + 1,
                    meta={"boundary": os.path.basename(a.boundary), "iso": a.iso})

    pg = prep(geom)
    total = grid.w * grid.h
    done = 0
    for ty in range(y0, y1 + 1):
        # a whole tile row shares the same latitude band
        _, lat_n = tilef_to_lonlat(0, ty)
        _, lat_s = tilef_to_lonlat(0, ty + 1)
        row = grid.bits[ty - y0]
        for tx in range(x0, x1 + 1):
            lon_w, _ = tilef_to_lonlat(tx, 0)
            lon_e, _ = tilef_to_lonlat(tx + 1, 0)
            if pg.intersects(box(lon_w, lat_s, lon_e, lat_n)):
                row[tx - x0] = True
        done += grid.w
        if a.verbose and (ty - y0) % 50 == 0:
            print(f"[mask] {done:,}/{total:,}", file=sys.stderr, flush=True)

    grid.save(a.out)
    print(f"frame       : x {x0}..{x1}  y {y0}..{y1}  ({grid.w} x {grid.h})")
    print(f"tiles in    : {grid.count:,} of {total:,}")
    print(f"written     : {a.out}")


# --------------------------------------------------------------------------- #
# command: zone                                                                #
# --------------------------------------------------------------------------- #

def parse_tile(s: str) -> tuple[int, int]:
    parts = s.replace("/", ",").replace(" ", ",").split(",")
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected x,y - got {s!r}")
    return int(parts[0]), int(parts[1])


def cmd_zone(a: argparse.Namespace) -> None:
    (ax, ay), (bx, by) = a.corners
    x0, x1 = min(ax, bx), max(ax, bx)
    y0, y1 = min(ay, by), max(ay, by)
    w, h = x1 - x0 + 1, y1 - y0 + 1

    west, _, _, north = tile_bounds(x0, y0)
    _, south, east, _ = tile_bounds(x1, y1)
    tw, th = tile_size_m((y0 + y1) // 2)

    print(f"zone        : x {x0}..{x1}  y {y0}..{y1}   ({w} x {h} = {w*h} tiles)")
    print(f"bbox        : {west:.5f},{south:.5f},{east:.5f},{north:.5f}  (W,S,E,N)")
    print(f"tile size   : {tw:.0f} m x {th:.0f} m")
    print(f"overpass    : [bbox:{south:.5f},{west:.5f},{north:.5f},{east:.5f}]")

    if a.tiles:
        grid = TileGrid.load(a.tiles).reframe(x0, y0, w, h)
        have = grid.count
        print(f"green       : {have} / {w*h}   ({100.0*have/(w*h):.1f}%)")
        missing = [(x0 + int(c), y0 + int(r)) for r, c in zip(*np.where(~grid.bits))]
        if missing:
            print(f"red tiles   : {len(missing)}")
            for tx, ty in missing[:a.list_limit]:
                bw, bs, be, bn = tile_bounds(tx, ty)
                print(f"   {tx},{ty}   centre {(bs+bn)/2:.5f},{(bw+be)/2:.5f}")
            if len(missing) > a.list_limit:
                print(f"   ... {len(missing) - a.list_limit} more")
            print("\n=> this square is NOT completable: the red tiles contain no "
                  "paved cyclable road.")
        else:
            print("\n=> every tile in this square has a paved cyclable road. "
                  "It is completable.")


# --------------------------------------------------------------------------- #
# command: info                                                                #
# --------------------------------------------------------------------------- #

def cmd_info(a: argparse.Namespace) -> None:
    g = TileGrid.load(a.tiles)
    west, _, _, north = tile_bounds(g.x0, g.y0)
    _, south, east, _ = tile_bounds(g.x0 + g.w - 1, g.y0 + g.h - 1)
    print(f"zoom        : {g.z}")
    print(f"frame       : x {g.x0}..{g.x0+g.w-1}  y {g.y0}..{g.y0+g.h-1}  ({g.w} x {g.h})")
    print(f"bbox        : {west:.4f},{south:.4f},{east:.4f},{north:.4f}")
    print(f"bits set    : {g.count:,} / {g.w*g.h:,}")

    if g.metres is not None:
        vals = g.metres[g.bits] / 1000.0
        if vals.size:
            print(f"road length : {vals.sum():,.0f} km   "
                  f"(median {np.median(vals):.2f}, p95 "
                  f"{np.percentile(vals, 95):.2f}, max {vals.max():.2f} "
                  f"km/tile)")
            n = max(1, a.top)
            flat = np.where(g.bits, g.metres, -1.0).ravel()
            order = np.argsort(flat)[::-1][:n]
            print(f"\nbusiest {n} tile(s):")
            print(f"  {'tile':>12}  {'km':>7}   centre            "
                  f"openstreetmap")
            for i in order:
                if flat[i] <= 0:
                    break
                tx = int(i % g.w) + g.x0
                ty = int(i // g.w) + g.y0
                w_, s_, e_, n_ = tile_bounds(tx, ty)
                lon, lat = (w_ + e_) / 2, (s_ + n_) / 2
                print(f"  {tx:>5},{ty:<6} {flat[i]/1000.0:>7.2f}   "
                      f"{lat:.4f},{lon:.4f}  "
                      f"https://www.openstreetmap.org/#map=15/"
                      f"{lat:.4f}/{lon:.4f}")

    print(f"\nmeta        : {json.dumps(g.meta, indent=2)}")


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat = r2 - r1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat * 0.5) ** 2
         + math.cos(r1) * math.cos(r2) * math.sin(dlon * 0.5) ** 2)
    return 2.0 * 6371008.8 * math.asin(math.sqrt(a))


def _tile_to_lonlat(xt, yt):
    """Vectorised fractional-tile -> lon/lat, at ZOOM."""
    n = float(NTILES)
    lon = np.asarray(xt, dtype=np.float64) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(
        np.pi * (1.0 - 2.0 * np.asarray(yt, dtype=np.float64) / n))))
    return lon, lat


def _lonlat_to_tile(lon, lat):
    """Vectorised lon/lat -> fractional tile coordinates, at ZOOM."""
    n = float(NTILES)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MAX_LAT, MAX_LAT)
    x = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0 * n
    s = np.sin(np.radians(lat))
    y = (0.5 - np.log((1.0 + s) / (1.0 - s)) / (4.0 * np.pi)) * n
    return x, y


def largest_square(bits: np.ndarray) -> tuple[int, int, int]:
    """
    The biggest solid square of set tiles: side, and its top-left corner.

    The classic dynamic program: a cell's score is how big a square can end
    there, which is one more than the smallest of its three neighbours. This
    is the number the whole explorer-tile game is played for.
    """
    h, w = bits.shape
    dp = np.zeros((h + 1, w + 1), dtype=np.int32)
    best, by, bx = 0, 0, 0
    for y in range(h):
        row = bits[y]
        for x in range(w):
            if not row[x]:
                continue
            v = 1 + int(min(dp[y, x], dp[y, x + 1], dp[y + 1, x]))
            dp[y + 1, x + 1] = v
            if v > best:
                best, by, bx = v, y - v + 1, x - v + 1
    return int(best), int(bx), int(by)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def bbox_arg(s: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in s.replace(" ", ",").split(",") if p]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be W,S,E,N")
    return parts[0], parts[1], parts[2], parts[3]
