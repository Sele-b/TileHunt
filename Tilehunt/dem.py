"""Elevation: 1-degree tiles (SRTM .hgt, Copernicus GeoTIFF), the per-block elevation database, and reading Copernicus by range."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import layout
from .tiles import NTILES, ZOOM, _lonlat_to_tile, _tile_to_lonlat
from .db import BLOCK_ZOOM


# --------------------------------------------------------------------------- #
# elevation                                                                    #
# --------------------------------------------------------------------------- #
#
# The profile is sampled from SRTM .hgt tiles, which are about as simple as a
# raster gets: a square grid of big-endian int16 metres, 3601 x 3601 for the
# 1 arc-second product and 1201 x 1201 for 3 arc-second, row 0 along the north
# edge. No GDAL, no rasterio -- numpy reads them directly.
#
# Free tiles, no account needed, one file per degree square:
#   https://elevation-tiles-prod.s3.amazonaws.com/skadi/N48/N48E002.hgt.gz
#
# Raw DEM samples are far too noisy to sum straight into an ascent figure: at
# 30 m postings a flat road accumulates hundreds of spurious metres. The
# profile is therefore resampled at a fixed spacing and smoothed over a
# distance window before the climbs are counted, which is what every sane
# tracker does.

SRTM_VOID = -32768


def hgt_name(lat: int, lon: int) -> str:
    return (f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}.hgt")


def hgt_url(lat: int, lon: int) -> str:
    ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    return ("https://elevation-tiles-prod.s3.amazonaws.com/skadi/"
            f"{ns}/{hgt_name(lat, lon)}.gz")


def copernicus_name(lat: int, lon: int) -> str:
    return (f"Copernicus_DSM_COG_10_"
            f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")


def copernicus_url(lat: int, lon: int) -> str:
    n = copernicus_name(lat, lon)
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


class _Grid:
    """One elevation tile, with where its samples sit in lon/lat."""

    __slots__ = ("z", "lon0", "lat0", "dlon", "dlat", "ny", "nx")

    def __init__(self, z, lon0, lat0, dlon, dlat):
        self.z = z
        self.lon0, self.lat0 = lon0, lat0     # the first sample, row 0 col 0
        self.dlon, self.dlat = dlon, dlat     # dlat is negative, going south
        self.ny, self.nx = z.shape

    def covers(self, lon, lat) -> bool:
        c = (lon - self.lon0) / self.dlon
        r = (lat - self.lat0) / self.dlat
        return -0.5 <= c <= self.nx - 0.5 and -0.5 <= r <= self.ny - 0.5

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        c = (lon - self.lon0) / self.dlon
        r = (lat - self.lat0) / self.dlat
        c0 = np.clip(np.floor(c).astype(int), 0, self.nx - 2)
        r0 = np.clip(np.floor(r).astype(int), 0, self.ny - 2)
        dc, dr = c - c0, r - r0
        return (self.z[r0, c0] * (1 - dc) * (1 - dr)
                + self.z[r0, c0 + 1] * dc * (1 - dr)
                + self.z[r0 + 1, c0] * (1 - dc) * dr
                + self.z[r0 + 1, c0 + 1] * dc * dr)


class DEM:
    """
    Bilinear elevation lookup over a folder of elevation tiles.

    Two formats are read:

      * SRTM .hgt / .hgt.gz -- a bare square of big-endian int16 metres, read
        straight by numpy with no extra dependency.
      * GeoTIFF (.tif), which is what the Copernicus DEM ships as. Needs
        rasterio, but the data is far better: no voids, no striping, and clean
        above 60 degrees north where SRTM simply stops.

    A GeoTIFF is located by its own georeferencing rather than by its filename,
    so any naming works and the two formats can sit in the same folder.
    """

    def __init__(self, root: str):
        self.root = root
        self._hgt: dict[tuple[int, int], _Grid | None] = {}
        self._tif: list[_Grid] | None = None
        self.missing: set[tuple[int, int]] = set()
        self.sources: set[str] = set()

    # ---- SRTM ------------------------------------------------------------ #
    def _load_hgt(self, ilat: int, ilon: int) -> "_Grid | None":
        key = (ilat, ilon)
        if key in self._hgt:
            return self._hgt[key]
        grid = None
        name = hgt_name(ilat, ilon)
        for cand in (os.path.join(self.root, name),
                     os.path.join(self.root, name + ".gz")):
            if not os.path.exists(cand):
                continue
            if cand.endswith(".gz"):
                import gzip
                with gzip.open(cand, "rb") as fh:
                    raw = fh.read()
            else:
                with open(cand, "rb") as fh:
                    raw = fh.read()
            n = int(round(math.sqrt(len(raw) / 2)))
            if n * n * 2 != len(raw):
                raise SystemExit(f"{cand} is not a square .hgt grid")
            z = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
            z = np.where(z == SRTM_VOID, np.nan, z)
            # .hgt is pixel-is-point: the first row is the north edge exactly
            grid = _Grid(z, float(ilon), float(ilat + 1),
                         1.0 / (n - 1), -1.0 / (n - 1))
            self.sources.add("SRTM .hgt")
            break
        self._hgt[key] = grid
        return grid

    # ---- GeoTIFF --------------------------------------------------------- #
    def _load_tifs(self) -> list:
        if self._tif is not None:
            return self._tif
        self._tif = []
        if not os.path.isdir(self.root):
            return self._tif
        paths = sorted(f for f in os.listdir(self.root)
                       if f.lower().endswith((".tif", ".tiff")))
        if not paths:
            return self._tif
        try:
            import rasterio
        except ImportError:
            raise SystemExit(
                f"{self.root} holds GeoTIFF tiles but rasterio is not "
                f"installed:  pip install rasterio")
        for name in paths:
            with rasterio.open(os.path.join(self.root, name)) as ds:
                t = ds.transform
                if t.b or t.d:
                    print(f"[dem] {name} is rotated, skipped", file=sys.stderr)
                    continue
                # read as float first: filling an int16 band with NaN would
                # quietly write garbage instead of a hole
                z = ds.read(1).astype(np.float32)
                holes = [SRTM_VOID, -32767.0, -9999.0]
                if ds.nodata is not None:
                    holes.append(float(ds.nodata))
                for h in holes:
                    z[z == np.float32(h)] = np.nan
                # GeoTIFF coordinates address pixel corners; a grid tagged
                # Point stores its value at that corner, one tagged Area at
                # the pixel centre half a step in.
                point = str(ds.tags().get("AREA_OR_POINT", "Area")).lower() == "point"
                off = 0.0 if point else 0.5
                self._tif.append(_Grid(z, t.c + off * t.a, t.f + off * t.e,
                                       t.a, t.e))
            self.sources.add("GeoTIFF")
        return self._tif

    def _find(self, lon: float, lat: float) -> "_Grid | None":
        g = self._load_hgt(int(math.floor(lat)), int(math.floor(lon)))
        if g is not None:
            return g
        for t in self._load_tifs():
            if t.covers(lon, lat):
                return t
        return None

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        out = np.full(len(lon), np.nan, dtype=np.float64)
        buckets: dict[int, list[int]] = {}
        grids: dict[int, _Grid] = {}
        for i in range(len(lon)):
            g = self._find(float(lon[i]), float(lat[i]))
            if g is None:
                self.missing.add((int(math.floor(lat[i])),
                                  int(math.floor(lon[i]))))
                continue
            buckets.setdefault(id(g), []).append(i)
            grids[id(g)] = g
        for key, idx in buckets.items():
            sel = np.asarray(idx)
            out[sel] = grids[key].sample(lon[sel], lat[sel])
        return out


# --------------------------------------------------------------------------- #
# Elevation database
#
# Same idea as the road database: the heights are cut up once into small
# per-block files so a ride only ever loads the few blocks it crosses. One
# block is a BLOCK_ZOOM tile, 8 x 8 explorer tiles, about 13 km a side.
#
# Inside a block the samples sit on a regular grid in *tile* coordinates, not
# in degrees. Web Mercator is conformal, so that grid is square on the ground
# and sampling it is a plain affine transform -- no geotransform bookkeeping at
# query time. The grid carries both edges (8 * res + 1 samples a side), so
# neighbouring blocks share their boundary samples and a bilinear lookup never
# has to reach into the block next door.
#
# Heights are int16 metres, which is 1 m precision and half the size of float32.
# -32768 marks a hole.
# --------------------------------------------------------------------------- #

DEM_RES = 64                    # samples per explorer tile edge (~25 m at 48N)
DEM_INDEX = "index.json"
DEM_VOID16 = -32768


def dem_block_of(x: int, y: int) -> tuple[int, int]:
    step = ZOOM - BLOCK_ZOOM
    return (int(x) >> step, int(y) >> step)


def dem_blocks_for_tiles(xs, ys) -> set:
    step = ZOOM - BLOCK_ZOOM
    bx = np.asarray(xs, dtype=np.int64) >> step
    by = np.asarray(ys, dtype=np.int64) >> step
    return set(zip(bx.tolist(), by.tolist()))


def dem_block_name(bx: int, by: int) -> str:
    return f"h{bx}_{by}.npz"


def dem_block_bbox(bx: int, by: int, pad_tiles: float = 0.0):
    """(west, south, east, north) of a block, optionally padded."""
    step = ZOOM - BLOCK_ZOOM
    k = 1 << step
    x0, x1 = bx * k - pad_tiles, (bx + 1) * k + pad_tiles
    y0, y1 = by * k - pad_tiles, (by + 1) * k + pad_tiles
    lon, lat = _tile_to_lonlat(np.array([x0, x1]), np.array([y0, y1]))
    return float(lon[0]), float(lat[1]), float(lon[1]), float(lat[0])


# ---- where the heights come from ------------------------------------------ #

class LocalSource:
    """1-degree elevation tiles already on disk (GeoTIFF or .hgt)."""

    kind = "local"

    def __init__(self, root: str):
        self.dem = DEM(root)
        self.root = root

    def cell(self, la: int, lo: int):
        g = self.dem._load_hgt(la, lo)
        if g is not None:
            return g
        for t in self.dem._load_tifs():
            if t.covers(lo + 0.5, la + 0.5):
                return t
        return None

    def label(self) -> str:
        return f"local tiles in {self.root} ({', '.join(sorted(self.dem.sources)) or 'none read'})"


class RemoteCopernicus:
    """
    Copernicus DEM GLO-30, read straight out of the public S3 bucket.

    The tiles are cloud-optimised GeoTIFFs, so GDAL fetches only the byte
    ranges covering the window asked for. Building a square pulls a couple of
    megabytes instead of the 30-odd MB of a whole 1-degree tile, and nothing is
    kept on disk but the block files.
    """

    kind = "remote"

    def __init__(self, base: str | None = None):
        # base exists so the range-read path can be pointed at a local server
        # in the tests; in normal use it is the public bucket
        self.base = base
        try:
            import rasterio
        except ImportError:
            raise SystemExit("reading Copernicus over the network needs "
                             "rasterio:  pip install rasterio")
        self.rio = rasterio
        self._bbox = None
        self.bytes_hint = 0

    def aim(self, bbox):
        """The window to read on the next cell() call."""
        self._bbox = bbox

    def cell(self, la: int, lo: int):
        from rasterio.windows import from_bounds
        url = "/vsicurl/" + self.url_for(la, lo)
        env = self.rio.Env(AWS_NO_SIGN_REQUEST="YES",
                           GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                           GDAL_HTTP_MAX_RETRY="4",
                           GDAL_HTTP_RETRY_DELAY="2")
        try:
            with env, self.rio.open(url) as ds:
                t = ds.transform
                w, s, e, n = self._bbox
                # a couple of pixels of slack so the bilinear stencil at the
                # edge of the window still has its four corners
                pad = 3
                win = from_bounds(w, s, e, n, t).round_offsets().round_lengths()
                win = win.round_offsets()
                col0 = max(0, int(win.col_off) - pad)
                row0 = max(0, int(win.row_off) - pad)
                col1 = min(ds.width, int(win.col_off + win.width) + pad)
                row1 = min(ds.height, int(win.row_off + win.height) + pad)
                if col1 <= col0 or row1 <= row0:
                    return None
                from rasterio.windows import Window
                sub = Window(col0, row0, col1 - col0, row1 - row0)
                z = ds.read(1, window=sub).astype(np.float32)
                self.bytes_hint += z.size * 4
                holes = [SRTM_VOID, -32767.0, -9999.0]
                if ds.nodata is not None:
                    holes.append(float(ds.nodata))
                for h in holes:
                    z[z == np.float32(h)] = np.nan
                point = str(ds.tags().get("AREA_OR_POINT", "Area")
                            ).lower() == "point"
                off = 0.0 if point else 0.5
                st = ds.window_transform(sub)
                return _Grid(z, st.c + off * st.a, st.f + off * st.e,
                             st.a, st.e)
        except Exception as exc:                        # noqa: BLE001
            msg = str(exc).split("\n")[0]
            if ("404" in msg or "Not Found" in msg
                    or "does not exist" in msg or "no such file" in msg.lower()):
                return None                             # pure sea, no tile
            raise SystemExit(f"could not read {self.url_for(la, lo)}: {msg}")

    def url_for(self, la: int, lo: int) -> str:
        if self.base:
            return f"{self.base.rstrip('/')}/{copernicus_name(la, lo)}.tif"
        return copernicus_url(la, lo)

    def label(self) -> str:
        if self.base:
            return f"Copernicus-layout GeoTIFF over HTTP ({self.base})"
        return "Copernicus DEM GLO-30 (S3, range reads)"


# ---- building ------------------------------------------------------------- #

def _dem_cells_of_block(bx: int, by: int) -> list:
    """The 1-degree cells a block needs data from, in build order."""
    w, s_, e, n = dem_block_bbox(bx, by)
    out = []
    for la in range(math.floor(s_), math.floor(n) + 1):
        for lo in range(math.floor(w), math.floor(e) + 1):
            out.append((la, lo))
    return out


def _dem_block_cells(path: str) -> set:
    """The cells a block file has already been fed from, without unpacking
    the heights: a completeness check must stay cheap, because it runs over
    every block a route crosses."""
    if not os.path.exists(path):
        return set()
    try:
        with np.load(path) as f:
            if "cells" not in f:
                return set()
            return set(map(tuple, f["cells"].tolist()))
    except (OSError, ValueError, KeyError):
        return set()


def _dem_complete(root: str, b) -> bool:
    return set(_dem_cells_of_block(*b)) <= _dem_block_cells(
        os.path.join(root, dem_block_name(*b))) and bool(
        _dem_block_cells(os.path.join(root, dem_block_name(*b))))


def _dem_read_block(path: str, side: int):
    """Returns (heights as float with NaN holes, set of cells already tried)."""
    if not os.path.exists(path):
        return np.full((side, side), np.nan, dtype=np.float32), set()
    with np.load(path) as f:
        z = f["z"].astype(np.float32)
        cells = set(map(tuple, f["cells"].tolist())) if "cells" in f else set()
    z[z == DEM_VOID16] = np.nan
    return z, cells


def dem_build(root: str, blocks, src, res: int = DEM_RES,
              verbose: bool = False, force: bool = False) -> dict:
    """
    Cut elevation into one npz per block.

    The outer loop is the 1-degree source cell, not the block, so every source
    cell is opened exactly once however many blocks it feeds -- which is what
    makes a whole-country build affordable. A block straddling a cell boundary
    is filled in several passes: each pass reads the block file, fills only the
    samples still void, and writes it back, recording which cells it has seen.
    That record is what makes the build resumable: a block is skipped only once
    every cell it needs has actually been tried.

    Memory stays at one block at a time, so the size of the zone does not
    change the footprint.
    """
    os.makedirs(root, exist_ok=True)
    step = ZOOM - BLOCK_ZOOM
    k = 1 << step
    side = k * res + 1
    off = np.arange(side, dtype=np.float64) / res

    need: dict[tuple[int, int], list] = {}      # cell -> blocks it feeds
    want: dict[tuple[int, int], set] = {}       # block -> cells it needs
    for bx, by in blocks:
        cs = _dem_cells_of_block(bx, by)
        want[(bx, by)] = set(cs)
        for c in cs:
            need.setdefault(c, []).append((bx, by))

    if not force:
        skipped = 0
        for b in list(want):
            seen = _dem_block_cells(os.path.join(root, dem_block_name(*b)))
            if seen and want[b] <= seen:
                for c in want[b]:
                    if b in need.get(c, ()):
                        need[c].remove(b)
                del want[b]
                skipped += 1
        need = {c: bl for c, bl in need.items() if bl}
        if skipped and verbose:
            print(f"   {skipped} block(s) already complete, skipped",
                  flush=True)

    stats = {"written": 0, "cells": 0, "empty": [], "holes": 0}
    cells = sorted(need)
    for ci, (la, lo) in enumerate(cells, 1):
        bl = need[(la, lo)]
        ws = [dem_block_bbox(bx, by) for bx, by in bl]
        bbox = (max(min(w for w, _, _, _ in ws), lo - 0.01),
                max(min(s_ for _, s_, _, _ in ws), la - 0.01),
                min(max(e for _, _, e, _ in ws), lo + 1.01),
                min(max(n for _, _, _, n in ws), la + 1.01))
        if hasattr(src, "aim"):
            src.aim(bbox)
        grid = src.cell(la, lo)
        stats["cells"] += 1
        tag = (f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
               f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
        if grid is None:
            stats["empty"].append((la, lo))
        if verbose:
            shape = "no data" if grid is None else f"{grid.ny}x{grid.nx}"
            print(f"   [{ci}/{len(cells)}] {tag}: {shape} "
                  f"-> {len(bl)} block(s)", flush=True)

        for bx, by in bl:
            path = os.path.join(root, dem_block_name(bx, by))
            z, seen = _dem_read_block(path, side)
            if grid is not None:
                lon, _ = _tile_to_lonlat(bx * k + off, np.zeros(side))
                _, lat = _tile_to_lonlat(np.zeros(side), by * k + off)
                L, A = np.meshgrid(lon, lat)
                flon, flat_ = L.ravel(), A.ravel()
                inside = ((flon >= grid.lon0 - abs(grid.dlon))
                          & (flon <= grid.lon0 + grid.nx * abs(grid.dlon))
                          & (flat_ <= grid.lat0 + abs(grid.dlat))
                          & (flat_ >= grid.lat0 + grid.ny * grid.dlat))
                if inside.any():
                    target = z.ravel()
                    sel = np.flatnonzero(inside)
                    vals = grid.sample(flon[inside], flat_[inside])
                    fresh = np.isnan(target[sel]) & ~np.isnan(vals)
                    target[sel[fresh]] = vals[fresh]
            seen.add((la, lo))
            if np.isnan(z).all() and want[(bx, by)] <= seen:
                # nothing anywhere in this block: open sea, write nothing
                if os.path.exists(path):
                    os.remove(path)
                continue
            if np.isnan(z).all():
                continue                    # still waiting on another cell
            np.savez_compressed(
                path, z=np.where(np.isnan(z), DEM_VOID16,
                                 np.round(z)).astype(np.int16),
                cells=np.array(sorted(seen), dtype=np.int16))

    files = [f for f in os.listdir(root)
             if f.startswith("h") and f.endswith(".npz")]
    stats["written"] = len(files)
    for b in want:
        path = os.path.join(root, dem_block_name(*b))
        if os.path.exists(path):
            with np.load(path) as f:
                stats["holes"] += int((f["z"] == DEM_VOID16).sum())

    meta = {"format": "tilehunt-dem/1", "zoom": BLOCK_ZOOM, "tile_zoom": ZOOM,
            "res": int(res), "side": int(side), "source": src.label(),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.join(root, DEM_INDEX)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
            if int(old.get("res", res)) != int(res):
                raise SystemExit(
                    f"{root} was built at res {old['res']} and you asked for "
                    f"{res}; use --res {old['res']} or a new folder")
        except (OSError, ValueError):
            pass
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    stats["side"] = side
    stats["bytes"] = sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return stats


# ---- reading -------------------------------------------------------------- #

class DEMDB:
    """Elevation out of a block database. Only the blocks touched are read."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, DEM_INDEX), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.res = int(self.meta["res"])
        self.side = int(self.meta.get("side", (1 << (ZOOM - BLOCK_ZOOM))
                                      * self.res + 1))
        self._cache: dict[tuple[int, int], np.ndarray | None] = {}
        self.missing: set[tuple[int, int]] = set()
        self.sources = {self.meta.get("source", "tilehunt dem database")}

    def _block(self, bx: int, by: int):
        key = (bx, by)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.root, dem_block_name(bx, by))
        z = None
        if os.path.exists(path):
            with np.load(path) as f:
                z = f["z"].astype(np.float32)
            z[z == DEM_VOID16] = np.nan
        self._cache[key] = z
        return z

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        step = ZOOM - BLOCK_ZOOM
        k = 1 << step
        xt, yt = _lonlat_to_tile(lon, lat)
        bx = np.floor(xt / k).astype(np.int64)
        by = np.floor(yt / k).astype(np.int64)
        out = np.full(len(np.atleast_1d(lon)), np.nan, dtype=np.float64)

        # a sample exactly on a block edge belongs to either block; grouping by
        # the floor is enough because both blocks store that edge
        order = np.lexsort((bx, by))
        i = 0
        while i < len(order):
            j = i
            b = (int(bx[order[i]]), int(by[order[i]]))
            while j < len(order) and (int(bx[order[j]]),
                                      int(by[order[j]])) == b:
                j += 1
            sel = order[i:j]
            i = j
            z = self._block(*b)
            if z is None:
                self.missing.add(b)
                continue
            fc = (xt[sel] - b[0] * k) * self.res
            fr = (yt[sel] - b[1] * k) * self.res
            c0 = np.clip(np.floor(fc).astype(int), 0, self.side - 2)
            r0 = np.clip(np.floor(fr).astype(int), 0, self.side - 2)
            dc, dr = fc - c0, fr - r0
            out[sel] = (z[r0, c0] * (1 - dc) * (1 - dr)
                        + z[r0, c0 + 1] * dc * (1 - dr)
                        + z[r0 + 1, c0] * (1 - dc) * dr
                        + z[r0 + 1, c0 + 1] * dc * dr)
        return out

    def stats(self) -> dict:
        files = [f for f in os.listdir(self.root) if f.startswith("h")
                 and f.endswith(".npz")]
        return {"blocks": len(files),
                "bytes": sum(os.path.getsize(os.path.join(self.root, f))
                             for f in files),
                "res": self.res, "side": self.side,
                "source": self.meta.get("source", "?"),
                "built": self.meta.get("built", "?")}


def open_dem(path: str):
    """A block database if the folder holds one, otherwise a folder of tiles."""
    if path and os.path.exists(os.path.join(path, DEM_INDEX)):
        return DEMDB(path)
    return DEM(path)


def default_dem(given: str | None) -> str:
    """Without --dem, the standard folder -- built on demand if it is absent."""
    return given or str(layout.DATA_DIR / "DEM_DB")


def ensure_dem(root: str, lon: np.ndarray, lat: np.ndarray, res: int,
               src_dir: str | None = None, verbose: bool = False,
               no_fetch: bool = False) -> bool:
    """
    Make sure the database holds the blocks this track crosses.

    A folder of plain 1-degree tiles is left alone; only a block database can
    be extended. Blocks are derived from the track itself, not from the square,
    so a ride that wanders outside the rectangle still gets its heights and
    nothing further is fetched.

    Returns False when the heights could not be obtained, having said why: the
    route itself is the expensive part of the run and must never be lost to a
    network problem.
    """
    if os.path.isdir(root) and not os.path.exists(os.path.join(root,
                                                               DEM_INDEX)):
        return True                     # a folder of raw tiles: not ours
    xt, yt = _lonlat_to_tile(lon, lat)
    blocks = sorted(dem_blocks_for_tiles(np.floor(xt).astype(np.int64),
                                         np.floor(yt).astype(np.int64)))
    if os.path.exists(os.path.join(root, DEM_INDEX)):
        try:
            with open(os.path.join(root, DEM_INDEX), encoding="utf-8") as fh:
                res = int(json.load(fh).get("res", res))
        except (OSError, ValueError):
            pass
        todo = [b for b in blocks if not _dem_complete(root, b)]
        if not todo:
            return True
        if no_fetch:
            print(f"elevation   : {len(todo)} block(s) missing from "
                  f"{os.path.basename(root)} and --no-dem-fetch is set; "
                  f"the profile will have gaps")
            return True
        blocks = todo
    if no_fetch:
        print(f"elevation   : {os.path.basename(root)} holds nothing for this "
              f"route and --no-dem-fetch is set")
        return False
    print(f"elevation   : fetching {len(blocks)} block(s) of heights into "
          f"{os.path.basename(root)}")
    try:
        src = LocalSource(src_dir) if src_dir else RemoteCopernicus()
        r = dem_build(root, blocks, src, res, verbose=verbose)
        print(f"   {r['written']:,} block(s) held, {r['bytes']/1e6:.1f} MB, "
              f"{r['cells']} source cell(s) read")
        return True
    except (SystemExit, OSError, RuntimeError, ValueError) as exc:
        msg = str(exc).split("\n")[0] or exc.__class__.__name__
        print(f"   could not get the heights: {msg}")
        print("   the route is unaffected; the profile is simply left out")
        return False


def dem_cells_for_tiles(xs: np.ndarray, ys: np.ndarray) -> set:
    """
    The 1-degree DEM cells a set of z14 tiles falls in.

    Each z14 tile is tested by its four corners, so a tile straddling a degree
    line pulls in both cells instead of only the one its centre lands in.
    """
    n = float(NTILES)
    cells = set()
    for dx in (0, 1):
        for dy in (0, 1):
            lon = (xs + dx) / n * 360.0 - 180.0
            lat = np.degrees(np.arctan(np.sinh(
                np.pi * (1.0 - 2.0 * (ys + dy) / n))))
            cells.update(zip(np.floor(lat).astype(np.int32).tolist(),
                             np.floor(lon).astype(np.int32).tolist()))
    return {(int(a), int(b)) for a, b in cells}


def cmd_dem(a: argparse.Namespace) -> None:
    """Which elevation tiles a zone needs, and what is still missing."""
    from .layout import place_output
    from .tiles import TileGrid
    if a.corners:
        (ax, ay), (bx, by) = a.corners
        m = a.margin
        x0, x1 = min(ax, bx) - m, max(ax, bx) + m
        y0, y1 = min(ay, by) - m, max(ay, by) + m
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs, ys = gx.ravel(), gy.ravel()
        what = f"square x {x0}..{x1}  y {y0}..{y1} (margin {m})"
    elif a.tiles:
        grid = TileGrid.load(a.tiles)
        # only tiles that hold a road: everything else is sea or bare ground
        # we would never draw a profile over, and the ocean cells do not even
        # exist in the Copernicus bucket
        yy, xx = np.nonzero(grid.bits)
        xs, ys = xx + grid.x0, yy + grid.y0
        what = f"{len(xs):,} tiles with road in {os.path.basename(a.tiles)}"
    else:
        raise SystemExit("give either --corners X,Y X,Y or --tiles NPZ")

    cells = sorted(dem_cells_for_tiles(xs, ys))
    print(f"zone        : {what}")
    print(f"dem cells   : {len(cells)} tile(s) of 1 degree")

    have, need = [], []
    for la, lo in cells:
        found = None
        if a.dem and os.path.isdir(a.dem):
            names = [hgt_name(la, lo), hgt_name(la, lo) + ".gz",
                     copernicus_name(la, lo) + ".tif"]
            for n in names:
                if os.path.exists(os.path.join(a.dem, n)):
                    found = n
                    break
            if found is None:
                # a GeoTIFF under any other name still counts if it covers it
                try:
                    for t in DEM(a.dem)._load_tifs():
                        if t.covers(lo + 0.5, la + 0.5):
                            found = "a GeoTIFF"
                            break
                except SystemExit:
                    pass
        (have if found else need).append((la, lo, found))

    if a.dem:
        print(f"already have: {len(have)}   missing: {len(need)}")
        for la, lo, n in have:
            print(f"   ok   {'N' if la>=0 else 'S'}{abs(la):02d}"
                  f"{'E' if lo>=0 else 'W'}{abs(lo):03d}   {n}")
    todo = need if a.dem else [(la, lo, None) for la, lo in cells]
    if not todo:
        print("nothing to download")
        return

    srtm = a.source == "srtm"
    urls = [(hgt_url(la, lo) if srtm else copernicus_url(la, lo))
            for la, lo, _ in todo]
    names = [(hgt_name(la, lo) + ".gz" if srtm
              else copernicus_name(la, lo) + ".tif") for la, lo, _ in todo]
    if not srtm:
        print(f"estimate    : ~{len(urls)} x 25-40 MB of Copernicus GeoTIFF")

    if a.out:
        out = place_output(a.out)
        ps = out.lower().endswith(".ps1") or out.lower().endswith(".bat")
        dest = a.dem or "DEM"
        lines = []
        if ps:
            dest = dest.replace("/", "\\")
            lines.append(f'New-Item -ItemType Directory -Force -Path "{dest}" '
                         f'| Out-Null')
            for u, n in zip(urls, names):
                lines.append(f'curl.exe -L --fail -o "{dest}\\{n}" "{u}"')
        else:
            lines.append("#!/bin/sh")
            lines.append(f'mkdir -p "{dest}"')
            for u, n in zip(urls, names):
                lines.append(f'curl -L --fail -o "{dest}/{n}" "{u}"')
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote       : {out}   ({len(urls)} downloads)")
        print("             run it, then pass --dem to route or profile")
    else:
        for u in urls:
            print(f"   {u}")


def cmd_demdb(a: argparse.Namespace) -> None:
    """Cut elevation into per-block files for a zone, or for a whole country."""
    from .db import BLOCK_ZOOM
    from .tiles import TileGrid, ZOOM, tile_size_m
    if a.corners:
        (ax, ay), (bx, by) = a.corners
        m = a.margin
        x0, x1 = min(ax, bx) - m, max(ax, bx) + m
        y0, y1 = min(ay, by) - m, max(ay, by) + m
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs, ys = gx.ravel(), gy.ravel()
        what = f"square x {x0}..{x1}  y {y0}..{y1} (margin {m})"
    elif a.tiles:
        grid = TileGrid.load(a.tiles)
        yy, xx = np.nonzero(grid.bits)
        xs, ys = xx + grid.x0, yy + grid.y0
        what = (f"{len(xs):,} tiles holding a road in "
                f"{os.path.basename(a.tiles)}")
    else:
        raise SystemExit("give either --corners X,Y X,Y or --tiles NPZ")

    blocks = sorted(dem_blocks_for_tiles(xs, ys))
    root = a.out

    print(f"zone        : {what}")
    print(f"blocks      : {len(blocks)} of "
          f"{1 << (ZOOM - BLOCK_ZOOM)}x{1 << (ZOOM - BLOCK_ZOOM)} tiles")
    src = LocalSource(a.src) if a.src else RemoteCopernicus()
    mid_y = int(np.median(ys))
    tw, _ = tile_size_m(mid_y)
    print(f"source      : {src.label()}")
    print(f"resolution  : {a.res} samples per tile edge "
          f"(~{tw / a.res:.0f} m on the ground here)")
    t0 = time.time()
    r = dem_build(root, blocks, src, a.res, verbose=True, force=a.force)
    print(f"blocks held : {r['written']} file(s) in {root}")
    if r["empty"]:
        print(f"no data     : {len(r['empty'])} cell(s) with nothing to read "
              f"-- sea, normally")
    if r["holes"]:
        print(f"holes       : {r['holes']:,} void sample(s)")
    print(f"on disk     : {r['bytes'] / 1e6:.1f} MB")
    print(f"took        : {time.time() - t0:.1f} s")
    print(f"use it with : --dem {os.path.basename(root)}")
