#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tilehunt - VeloViewer / Statshunters "explorer tile" coverage from OpenStreetMap.

An explorer tile is a standard slippy-map tile at zoom 14 (16384 x 16384 tiles
covering the world). This tool answers, for every z14 tile in a region:

    "is there at least one paved road, on which cycling is allowed,
     passing through this tile?"

Green tiles are reachable. Red tiles are not (no paved cyclable road inside),
so no amount of route planning will ever let you claim them -- which is exactly
what you need to know before choosing where to attack a max-square.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

  # 0. install
  pip install numpy scipy shapely osmium rasterio

  Layout (created automatically, relative to this script):
      tilehunt.py
      Data/     .osm.pbf and .geojson downloads, plus the .npz datasets
      Maps/     the generated .html

  A bare input name is looked up in Data/. A bare output name goes to Maps/
  if it ends in .html, else to Data/. Any path containing a folder is used
  as given. Override with --data-dir / --maps-dir.

  # 1. put the downloads in Data/
  #    https://download.geofabrik.de/europe/france-latest.osm.pbf
  #    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson  -> countries.geojson
  #    https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson
  #    https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson

  # 2. the database: roads rasterised onto the z14 grid, the road graph for
  #    routing, and the heights.  (~25-30 min for France)
  python tilehunt.py genBDD --pbf france-latest.osm.pbf --out france.paved.npz \
         --db Database_OSM --dem-db DEM_DB -v

  # 3. the scope mask (countries.geojson is picked up automatically)
  python tilehunt.py mask --iso FRA --clip=-5.6,41.2,9.8,51.3 \
         --min-part-area 0.004 --out france.mask.npz

  # 4. the map. mask, boundary, regions and departements are all found in
  #    Data/ by their conventional names, so nothing else need be named.
  python tilehunt.py buildmap --tiles france.paved.npz --clip-to-mask \
         --out france.html

  # 4b. from the map: "plan a square ride" hands you the two commands
  python tilehunt.py compSquareRoute --db Database_OSM \
         --corners 8314,5512 8329,5527 --start 8329,5512 --end 8329,5512 \
         --cands 6 --seconds 120 --reopt-seconds 60 \
         --gpx z8314_5512.gpx --out z8314_5512.route.json

  # 5. inspect a candidate square, or ask why one tile came out green
  python tilehunt.py zone --tiles france.paved.npz --corners 8290,5660 8301,5671
  python tilehunt.py why --pbf france-latest.osm.pbf --tile 8301,5708 --show-rejected

Sub-extracts are faster to iterate on, e.g.
  https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf
  https://download.geofabrik.de/europe/france/centre-latest.osm.pbf
`genBDD` accepts several --pbf flags and merges them.
The former names `tiles`, `map` and `route` still work.

--------------------------------------------------------------------------------
NOTES ON CORRECTNESS
--------------------------------------------------------------------------------
* Rasterisation is exact, not sampled. Every tile a way's polyline passes
  through is recorded, including tiles where it only clips a corner, via a
  grid-traversal (supercover DDA) walk in tile space. A 5 m stub poking into a
  tile counts -- which matches how the challenge is scored.
* Straight lines in Web Mercator are rhumb lines, not great circles. Over OSM
  node spacing (metres to a few hundred metres) the difference is far below one
  tile, so segments are walked as straight lines in tile space.
* The paved / cyclable decision is tag-driven and documented in
  `classify_way()`. Defaults are tuned for France. Use --dump-rejects to audit.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import os
import random
import re
import sys
import time
import zlib
from collections import Counter
from typing import Iterable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# project layout                                                               #
# --------------------------------------------------------------------------- #
# Everything is resolved relative to the folder holding this script, so the
# commands work whatever directory you launch them from:
#
#   <script folder>/
#       tilehunt.py
#       Data/     downloads (.osm.pbf, .geojson) and datasets (.npz)
#       Maps/     the generated .html
#
# An input given as a bare name is looked up in Data/. An output given as a
# bare name goes to Maps/ if it ends in .html, otherwise to Data/. Give any
# path with a folder in it (or an absolute path) and it is used verbatim.
# Override the two folders with --data-dir / --maps-dir.

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MAPS_DIR = ROOT / "Maps"
RUNS_DIR = ROOT / "Runs"      # Runs/<kind>/<name>/, named by --runname


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def run_name(name: str) -> str:
    """
    A --runname value fit to be a folder name, on any filesystem.

    Accents are transliterated rather than struck out, so "tracé d'été" comes
    back as "trace-d-ete" and not as a row of dashes.
    """
    import unicodedata
    folded = unicodedata.normalize("NFKD", str(name).strip())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    clean = SAFE_NAME.sub("-", folded).strip("-._")
    if not clean:
        raise SystemExit(f"--runname {name!r} leaves nothing usable as a folder "
                         f"name; use letters, digits, dash or underscore")
    if len(clean) > 64:
        clean = clean[:64].rstrip("-._")
    return clean


def find_run(name: str) -> Path | None:
    """An existing Runs/<kind>/<name>/ folder, whatever kind made it."""
    n = run_name(name)
    if not RUNS_DIR.is_dir():
        return None
    for kind in sorted(RUNS_DIR.iterdir()):
        if kind.is_dir() and (kind / n).is_dir():
            return kind / n
    return None


def run_dir(name: str, kind: str = "Square") -> Path:
    """
    Runs/<kind>/<name>/, created on demand.

    The kind groups rides by how they were produced -- Square today, other
    modes later -- so the folder does not become a flat pile. A ride already
    filed under some kind is found there rather than duplicated: that is what
    lets `buildmap --runname X` work without being told which mode made X.
    """
    d = find_run(name) or (RUNS_DIR / kind / run_name(name))
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_file(name: str, suffix: str, kind: str = "Square") -> str:
    """Runs/<kind>/<name>/<name><suffix>."""
    n = run_name(name)
    return str(run_dir(n, kind) / f"{n}{suffix}")


def find_input(p: str) -> str:
    """Resolve an input path, falling back to the Data folder."""
    if not p:
        return p
    cand = Path(p).expanduser()
    if cand.exists():
        return str(cand)
    alt = DATA_DIR / cand.name
    if alt.exists():
        return str(alt)
    return str(cand)          # let the caller raise a clear error


def place_output(p: str) -> str:
    """Resolve an output path: bare .html -> Maps/, bare anything else -> Data/."""
    if not p:
        return p
    cand = Path(p).expanduser()
    if cand.parent != Path("."):
        cand.parent.mkdir(parents=True, exist_ok=True)
        return str(cand)
    target = MAPS_DIR if cand.suffix.lower() in (".html", ".htm") else DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return str(target / cand.name)


def default_input(*names: str) -> str | None:
    """First of these conventional filenames that exists in the Data folder."""
    for n in names:
        p = DATA_DIR / n
        if p.exists():
            return str(p)
    return None


def apply_layout(a: argparse.Namespace) -> None:
    """Rewrite the namespace's path fields according to the layout above."""
    global DATA_DIR, MAPS_DIR, RUNS_DIR
    if getattr(a, "data_dir", None):
        DATA_DIR = Path(a.data_dir).expanduser().resolve()
    if getattr(a, "maps_dir", None):
        MAPS_DIR = Path(a.maps_dir).expanduser().resolve()
    if getattr(a, "runs_dir", None):
        RUNS_DIR = Path(a.runs_dir).expanduser().resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # --runname gathers one ride's outputs under Runs/<name>/, all sharing
    # the name. Anything named explicitly still wins, so it is a shorthand and
    # never a constraint.
    name = getattr(a, "runname", None)
    if name:
        a.runname = run_name(name)
        kind = getattr(a, "run_kind", "Square")
        for field, suffix in (getattr(a, "run_files", ()) or ()):
            if not getattr(a, field, None):
                setattr(a, field, run_file(a.runname, suffix, kind))
        if hasattr(a, "name") and getattr(a, "name", None) is None:
            a.name = a.runname

    for field in ("pbf", "geojson"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, [find_input(x) for x in v])
    for field in ("boundary", "regions", "departments", "tiles", "mask",
                  "visited"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, find_input(v))
    for field in ("db", "dem", "src", "dem_db", "dem_src"):
        v = getattr(a, field, None)
        if v and not os.path.isabs(v) and os.sep not in v:
            setattr(a, field, str(DATA_DIR / v))
    for field in ("graph_cache", "gpx", "route"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, find_input(v) if field == "route"
                    else place_output(v))
    for field in ("out", "dump_rejects"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, place_output(v))


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
# OSM tag interpretation                                                       #
# --------------------------------------------------------------------------- #

# Surfaces that count as "paved". Includes French pavé (sett / cobblestone):
# unpleasant, but it is a hard surface and it is rideable on a road bike.
PAVED_SURFACES = {
    "paved", "asphalt", "chipseal", "concrete", "concrete:lanes",
    "concrete:plates", "paving_stones", "paving_stones:lanes",
    "bricks", "brick", "metal", "wood", "rubber", "tartan", "acrylic",
}

# Pavé: a hard surface and legally a road, but punishing on a road bike.
# Counted by default; turn off with --no-cobbles.
COBBLE_SURFACES = {
    "sett", "cobblestone", "cobblestone:flattened", "unhewn_cobblestone",
}

# `service=*` values that are not through-routes. Driveways and parking
# aisles are the single biggest source of false "there is a road here".
SERVICE_EXCLUDED = {
    "driveway", "parking_aisle", "drive-through", "emergency_access",
    "bus_stop",
}

# A road this rough is not a paved road whatever `surface` claims.
BAD_SMOOTHNESS = {
    "bad", "very_bad", "horrible", "very_horrible", "impassable",
}

UNPAVED_SURFACES = {
    "unpaved", "gravel", "fine_gravel", "compacted", "dirt", "ground",
    "earth", "grass", "grass_paver", "sand", "mud", "woodchips",
    "pebblestone", "stone", "rock", "clay", "salt", "snow", "ice",
    "metal_grid", "shells", "limestone", "crushed_limestone",
}

# Highway classes assumed paved when `surface` is missing. In France these are
# sealed essentially everywhere.
ASSUME_PAVED = {
    "trunk", "trunk_link",            # only reached if bicycle=yes (see below)
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service", "road", "cycleway", "busway",
}

# Highway classes assumed UNpaved when `surface` is missing.
ASSUME_UNPAVED = {
    "track", "path", "footway", "bridleway", "pedestrian", "steps",
}

# Never rideable, whatever the tags.
HIGHWAY_EXCLUDED = {
    "motorway", "motorway_link", "construction", "proposed", "planned",
    "abandoned", "razed", "raceway", "escape", "bus_guideway", "platform",
    "corridor", "elevator", "steps", "services", "rest_area", "emergency_bay",
    "virtual", "no",
}

# Need an explicit bicycle=yes/designated/permissive to be counted.
HIGHWAY_NEEDS_BIKE_YES = {
    "trunk", "trunk_link",            # French voies express usually ban bikes
    "footway", "pedestrian", "path", "bridleway",
}

BIKE_POSITIVE = {"yes", "designated", "permissive", "official", "destination"}
BIKE_NEGATIVE = {"no", "dismount", "use_sidepath", "private"}
ACCESS_NEGATIVE = {"no", "private", "military", "permit", "agricultural", "forestry"}


# Everything that counts as a "road" you could ride on, as opposed to a
# footway, track or path. Used by the `all-roads` preset.
ROAD_CLASSES = {
    "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service",
    "road", "cycleway", "busway",
}


def classify_all_roads(tags) -> tuple[bool, str]:
    """
    Accept every road except motorways, ignoring surface and access entirely.

    This is the widest sensible reading of "roads I could ride": it keeps
    unpaved and private roads, and keeps trunk roads without asking whether
    cycling is signed. It drops motorways, and drops footways, paths, tracks
    and steps because those are not roads -- use `any-highway` for those.
    """
    hw = tags.get("highway")
    if not hw:
        return False, "no-highway"
    if tags.get("area") == "yes":
        return False, "area"
    if hw in ("motorway", "motorway_link"):
        return False, f"highway={hw}"
    if (tags.get("motorroad") or "").lower() == "yes":
        return False, "motorroad=yes"
    if hw in ROAD_CLASSES:
        return True, f"highway={hw}"
    return False, f"highway={hw}-not-a-road"


def classify_any_highway(tags) -> tuple[bool, str]:
    """
    Accept any way that is a highway of any kind, paved or not, rideable or not.

    Useful as a "scope" layer: a tile with no highway at all is wilderness,
    water or farmland, while a tile that has highways but none of them paved
    and cyclable is the interesting kind of red tile.
    """
    hw = tags.get("highway")
    if not hw:
        return False, "no-highway"
    if tags.get("area") == "yes":
        return False, "area"
    if hw in {"construction", "proposed", "planned", "platform", "elevator",
              "corridor", "no", "virtual"}:
        return False, f"highway={hw}"
    return True, f"highway={hw}"


def classify_way(tags, cobbles: bool = True, strict_surface: bool = False,
                 trust_tracktype: bool = False) -> tuple[bool, str]:
    """
    Decide whether an OSM way counts as "paved road, cycling allowed".

    `tags` is any mapping-like object supporting `.get`. Returns
    (accepted, reason); reason names the rule that fired, for auditing.

    cobbles        -- count sett / cobblestone as paved (default yes)
    strict_surface -- require an explicit paved `surface` tag; a way with no
                      surface at all is rejected instead of being assumed
                      paved from its highway class
    trust_tracktype -- accept highway=track + tracktype=grade1 when no surface
                      is tagged. Off by default: the OSM wiki defines grade1 as
                      "solid, usually paved OR heavily compacted hardcore", so
                      roughly half of them are white forest roads rather than
                      tarmac. A genuinely surfaced track carries surface=* and
                      is accepted on that basis regardless of this switch.
    """
    hw = tags.get("highway")
    if not hw:
        return False, "no-highway"
    if hw in HIGHWAY_EXCLUDED:
        return False, f"highway={hw}"
    if tags.get("area") == "yes":
        return False, "area"

    # motorroad=yes marks a road carrying motorway-type restrictions without
    # being classed as a motorway -- the French voie express. Cycling is
    # banned on these, and they are very often tagged highway=trunk, so this
    # check has to come before anything that would accept a trunk road.
    if (tags.get("motorroad") or "").lower() == "yes":
        return False, "motorroad=yes"

    # --- lifecycle / usability -------------------------------------------- #
    for k in ("construction", "proposed"):
        if tags.get(k) and tags.get("highway") == k:
            return False, "lifecycle"
    if tags.get("impassable") == "yes":
        return False, "impassable"

    bicycle = (tags.get("bicycle") or "").lower()
    access = (tags.get("access") or "").lower()

    # --- access ------------------------------------------------------------ #
    if bicycle in BIKE_NEGATIVE:
        return False, f"bicycle={bicycle}"
    if access in ACCESS_NEGATIVE and bicycle not in BIKE_POSITIVE:
        return False, f"access={access}"
    if hw in HIGHWAY_NEEDS_BIKE_YES and bicycle not in BIKE_POSITIVE:
        # a cycleway tagged on the same way still rescues it
        if tags.get("cycleway") in ("lane", "track", "opposite_lane", "opposite_track"):
            pass
        else:
            return False, f"{hw}-without-bicycle-yes"

    # --- not-a-through-route service ways ---------------------------------- #
    if hw == "service" and (tags.get("service") or "").lower() in SERVICE_EXCLUDED:
        return False, f"service={tags.get('service')}"

    # --- vehicle access (separate from `access`) --------------------------- #
    vehicle = (tags.get("vehicle") or "").lower()
    if vehicle in ACCESS_NEGATIVE and bicycle not in BIKE_POSITIVE:
        return False, f"vehicle={vehicle}"

    # --- declared roughness ------------------------------------------------ #
    smooth = (tags.get("smoothness") or "").lower()
    if smooth in BAD_SMOOTHNESS:
        return False, f"smoothness={smooth}"

    # --- surface ----------------------------------------------------------- #
    # A multi-value surface (a;b) only counts if EVERY part is paved, and an
    # unrecognised value counts as unpaved rather than falling through to the
    # optimistic class assumption.
    surface = (tags.get("surface") or "").lower()
    if surface:
        parts = [p.strip() for p in surface.split(";") if p.strip()]
        if parts and all(p in PAVED_SURFACES or (cobbles and p in COBBLE_SURFACES)
                         for p in parts):
            return True, "surface-paved"
        return False, f"surface={parts[0] if parts else surface}"
    if strict_surface:
        return False, "no-surface-tag"
    if hw in ASSUME_PAVED:
        return True, "assumed-paved"
    if hw in ASSUME_UNPAVED:
        if trust_tracktype and tags.get("tracktype") == "grade1" and not surface:
            return True, "tracktype-grade1"
        return False, f"{hw}-assumed-unpaved"
    return False, f"highway={hw}-unhandled"


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
# command: tiles                                                               #
# --------------------------------------------------------------------------- #

def read_ways_pbf(paths: Sequence[str], idx: str, verbose: bool,
                  dump_rejects, accept=classify_way, lengths=None, db=None
                  ) -> tuple[set[tuple[int, int]], Counter, int, int]:
    """Rasterise paved+cyclable ways from one or more .osm.pbf files."""
    try:
        import osmium
    except ImportError:
        raise SystemExit("pyosmium is required for --pbf:  pip install osmium")

    tiles: set[tuple[int, int]] = set()
    reasons: Counter = Counter()
    stats = {"kept": 0, "seen": 0}
    t0 = time.time()
    rejects = open(dump_rejects, "w", encoding="utf-8") if dump_rejects else None

    def handle(way) -> None:
        """Classify one way and, if accepted, rasterise it."""
        stats["seen"] += 1
        if db is not None:
            # the database takes a wider set than any single preset, so it is
            # filled from the raw way before the preset has its say
            try:
                pts = [(n.ref, n.lon, n.lat) for n in way.nodes
                       if n.location.valid()]
            except Exception:
                pts = []
            if len(pts) >= 2:
                db.add(way.tags, pts)
        ok, reason = accept(way.tags)
        reasons[reason] += 1
        if not ok:
            if rejects is not None:
                rejects.write(f"{way.id}\t{reason}\t"
                              f"{dict(way.tags)}\n")
            return
        try:
            coords = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
        except Exception:                      # missing node locations
            reasons["no-locations"] += 1
            return
        if not coords:
            reasons["no-locations"] += 1
            return
        stats["kept"] += 1
        walk_polyline(coords, tiles, lengths)
        if verbose and stats["kept"] % 250_000 == 0:
            print(f"[tiles]   {stats['kept']:,} ways kept, {len(tiles):,} tiles, "
                  f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    for path in paths:
        if verbose:
            print(f"[tiles] reading {path}", file=sys.stderr, flush=True)

        if hasattr(osmium, "FileProcessor"):
            # Nodes must stay in the entity set so the location cache can be
            # filled; EntityFilter then forwards only ways to us, and KeyFilter
            # drops the ~90% of ways with no highway tag inside the C++ layer.
            fp = osmium.FileProcessor(path).with_locations(idx)
            fp = fp.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
            fp = fp.with_filter(osmium.filter.KeyFilter("highway"))
            for obj in fp:
                handle(obj)
        else:                                  # pyosmium < 4
            class _H(osmium.SimpleHandler):    # type: ignore[misc]
                def way(self, w):
                    handle(w)

            _H().apply_file(path, locations=True, idx=idx)

    if rejects is not None:
        rejects.close()
        if verbose:
            print(f"[tiles] rejects written to {dump_rejects}", file=sys.stderr)

    return tiles, reasons, stats["kept"], stats["seen"]


def read_ways_geojson(paths: Sequence[str], verbose: bool, filter_tags: bool,
                      accept=classify_way, lengths=None
                      ) -> tuple[set[tuple[int, int]], Counter, int, int]:
    """
    Rasterise (Multi)LineStrings from GeoJSON. Used for non-OSM line sources;
    with --no-tag-filter every line is accepted, which is the right behaviour
    for sources that are already filtered.
    """
    tiles: set[tuple[int, int]] = set()
    reasons: Counter = Counter()
    kept = seen = 0
    for path in paths:
        if verbose:
            print(f"[tiles] reading {path}", file=sys.stderr, flush=True)
        with open(path, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]
        for ft in feats:
            seen += 1
            geom = ft.get("geometry") or {}
            props = ft.get("properties") or {}
            if filter_tags:
                ok, reason = accept(props)
                reasons[reason] += 1
                if not ok:
                    continue
            else:
                reasons["accepted"] += 1
            gt = geom.get("type")
            if gt == "LineString":
                parts = [geom["coordinates"]]
            elif gt == "MultiLineString":
                parts = geom["coordinates"]
            else:
                continue
            kept += 1
            for part in parts:
                walk_polyline([(c[0], c[1]) for c in part], tiles, lengths)
    return tiles, reasons, kept, seen


def cmd_tiles(a: argparse.Namespace) -> None:
    if not a.pbf and not a.geojson:
        raise SystemExit("give --pbf and/or --geojson")
    accept = pick_classifier(a)
    db = None
    if getattr(a, "db", None):
        os.makedirs(a.db, exist_ok=True)
        db = DBWriter(a.db)
        print(f"database    : writing to {a.db}")
    lengths: dict[tuple[int, int], float] | None = (
        None if a.no_lengths else {})
    tiles: set[tuple[int, int]] = set()
    reasons: Counter = Counter()
    kept = seen = 0
    if a.pbf:
        t, r, k, s = read_ways_pbf(a.pbf, a.idx, a.verbose, a.dump_rejects,
                                   accept, lengths, db)
        tiles |= t; reasons += r; kept += k; seen += s
    if a.geojson:
        t, r, k, s = read_ways_geojson(a.geojson, a.verbose,
                                       not a.no_tag_filter, accept, lengths)
        tiles |= t; reasons += r; kept += k; seen += s

    if a.bbox:
        w, s_, e, n = a.bbox
        tx0, ty0 = lonlat_to_tilef(w, n)
        tx1, ty1 = lonlat_to_tilef(e, s_)
        bx0, by0, bx1, by1 = (math.floor(tx0), math.floor(ty0),
                              math.floor(tx1), math.floor(ty1))
        tiles = {t for t in tiles if bx0 <= t[0] <= bx1 and by0 <= t[1] <= by1}

    meta = {"source": "osm-pbf" if a.pbf else "geojson",
            "accept": a.accept,
            "inputs": [os.path.basename(p) for p in (a.pbf or []) + (a.geojson or [])],
            "ways_kept": kept, "ways_seen": seen, "built": time.strftime("%Y-%m-%d %H:%M")}
    grid = TileGrid.from_tiles(tiles, pad=a.pad, meta=meta, lengths=lengths)
    grid.save(a.out)

    if db is not None:
        print("database    : consolidating blocks ...")
        info = db.consolidate({"source": [os.path.basename(x)
                                          for x in (a.pbf or [])]}, a.verbose)
        print(f"database    : {info['blocks']:,} blocks, "
              f"{info['bytes']/1e6:,.0f} MB, {info['ways']:,} ways, "
              f"{info['node_refs']:,} node refs")

    print(f"ways seen   : {seen:,}")
    print(f"ways kept   : {kept:,}")
    print(f"tiles hit   : {len(tiles):,}")
    if grid.metres is not None:
        inside = grid.metres[grid.bits] / 1000.0
        print(f"road length : {inside.sum():,.0f} km in frame   "
              f"(median {np.median(inside):.2f} km/tile, "
              f"p95 {np.percentile(inside, 95):.2f}, max {inside.max():.1f})")
    print(f"frame       : x {grid.x0}..{grid.x0+grid.w-1}  y {grid.y0}..{grid.y0+grid.h-1}"
          f"  ({grid.w} x {grid.h})")
    print(f"written     : {a.out}")

    if getattr(a, "dem_db", None):
        # the elevation half of the database. It covers exactly the tiles that
        # came out of the PBF pass, so the two halves always agree on extent,
        # and it is built cell by cell so each 1-degree source tile is fetched
        # once. Interrupt it and run the same command again: finished blocks
        # are skipped.
        print()
        print(f"elevation   : building into {a.dem_db}")
        src = LocalSource(a.dem_src) if a.dem_src else RemoteCopernicus()
        print(f"   source   : {src.label()}")
        yy, xx = np.nonzero(grid.bits)
        blocks = sorted(dem_blocks_for_tiles(xx + grid.x0, yy + grid.y0))
        mid = int(np.median(yy)) + grid.y0
        tw, _ = tile_size_m(mid)
        print(f"   blocks   : {len(blocks):,} of 8x8 tiles, "
              f"{a.dem_res} samples per tile edge "
              f"(~{tw / a.dem_res:.0f} m on the ground)")
        t1 = time.time()
        r = dem_build(a.dem_db, blocks, src, a.dem_res,
                      verbose=True, force=a.dem_force)
        print(f"   held     : {r['written']:,} block file(s), "
              f"{r['bytes']/1e6:,.0f} MB")
        if r["empty"]:
            print(f"   no data  : {len(r['empty'])} cell(s) returned nothing "
                  f"(sea, normally)")
        if r["holes"]:
            print(f"   holes    : {r['holes']:,} void sample(s)")
        print(f"   took     : {time.time() - t1:.0f} s")

    if a.verbose:
        print("\ntop tag decisions:")
        for reason, c in reasons.most_common(25):
            print(f"  {c:>12,}  {reason}")


# --------------------------------------------------------------------------- #
# command: why                                                                 #
# --------------------------------------------------------------------------- #

def pick_classifier(a: argparse.Namespace):
    """Build the accept() used by `tiles` and `why` from the CLI flags."""
    if a.accept == "any-highway":
        return classify_any_highway
    if a.accept == "all-roads":
        return classify_all_roads
    return lambda tags: classify_way(
        tags,
        cobbles=not getattr(a, "no_cobbles", False),
        strict_surface=getattr(a, "strict_surface", False),
        trust_tracktype=getattr(a, "trust_tracktype", False),
    )


def cmd_why(a: argparse.Namespace) -> None:
    """Explain a tile: list every highway touching it, accepted or not."""
    try:
        import osmium
    except ImportError:
        raise SystemExit("pyosmium is required:  pip install osmium")

    accept = pick_classifier(a)
    targets = set(a.tile)
    hits: dict[tuple[int, int], dict[str, list]] = {
        t: {"yes": [], "no": []} for t in targets}

    # Bounding box of the tiles of interest, padded by one tile. Any way whose
    # own bounding box misses this can be dropped before doing any tile maths,
    # which is what keeps a whole-country scan bearable.
    xs = [t[0] for t in targets]
    ys = [t[1] for t in targets]
    bw, _, _, bn = tile_bounds(min(xs) - 1, min(ys) - 1)
    _, bs, be, _ = tile_bounds(max(xs) + 1, max(ys) + 1)

    stats = {"scanned": 0, "near": 0}
    t0 = time.time()

    print(f"scanning for {len(targets)} tile(s) in "
          f"{bs:.4f},{bw:.4f} .. {bn:.4f},{be:.4f}", file=sys.stderr, flush=True)

    def handle(way) -> None:
        stats["scanned"] += 1
        if stats["scanned"] % 1_000_000 == 0:
            print(f"  {stats['scanned']:,} ways scanned, "
                  f"{stats['near']} near the target, "
                  f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        try:
            coords = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
        except Exception:
            return
        if not coords:
            return
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        if (max(lons) < bw or min(lons) > be
                or max(lats) < bs or min(lats) > bn):
            return
        stats["near"] += 1
        touched: set[tuple[int, int]] = set()
        walk_polyline(coords, touched)
        touched &= targets
        if not touched:
            return
        ok, reason = accept(way.tags)
        rec = (way.id, reason, dict(way.tags))
        for t in touched:
            hits[t]["yes" if ok else "no"].append(rec)

    for path in a.pbf:
        if hasattr(osmium, "FileProcessor"):
            fp = osmium.FileProcessor(path).with_locations(a.idx)
            fp = fp.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
            fp = fp.with_filter(osmium.filter.KeyFilter("highway"))
            for obj in fp:
                handle(obj)
        else:
            class _H(osmium.SimpleHandler):   # type: ignore[misc]
                def way(self, w):
                    handle(w)
            _H().apply_file(path, locations=True, idx=a.idx)

    print(f"done: {stats['scanned']:,} ways in {time.time() - t0:.0f}s",
          file=sys.stderr, flush=True)

    for t in sorted(targets):
        w, s_, e, n = tile_bounds(*t)
        yes, no = hits[t]["yes"], hits[t]["no"]
        print(f"\n=== tile {t[0]},{t[1]} "
              f"(centre {(s_+n)/2:.5f},{(w+e)/2:.5f}) ===")
        print(f"verdict: {'GREEN' if yes else 'RED'}   "
              f"({len(yes)} accepted, {len(no)} rejected, preset {a.accept})")
        print(f"osm.org: https://www.openstreetmap.org/#map=15/"
              f"{(s_+n)/2:.5f}/{(w+e)/2:.5f}")
        if yes:
            print("\n  ACCEPTED -- these are why it is green:")
            for wid, reason, tags in yes[:a.limit]:
                keep = {k: v for k, v in tags.items()
                        if k in ("highway", "surface", "tracktype", "bicycle",
                                 "access", "vehicle", "service", "smoothness",
                                 "motorroad", "cycleway", "name", "ref")}
                print(f"    way {wid:<12} [{reason}]  {keep}")
            if len(yes) > a.limit:
                print(f"    ... {len(yes) - a.limit} more")
        if no and a.show_rejected:
            print("\n  rejected:")
            for wid, reason, tags in no[:a.limit]:
                keep = {k: v for k, v in tags.items()
                        if k in ("highway", "surface", "tracktype", "bicycle",
                                 "access", "vehicle", "service", "smoothness",
                                 "motorroad", "cycleway", "name", "ref")}
                print(f"    way {wid:<12} [{reason}]  {keep}")
            if len(no) > a.limit:
                print(f"    ... {len(no) - a.limit} more")


# --------------------------------------------------------------------------- #
# command: mask                                                                #
# --------------------------------------------------------------------------- #

def cmd_mask(a: argparse.Namespace) -> None:
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


# --------------------------------------------------------------------------- #
# routing: the road graph                                                      #
# --------------------------------------------------------------------------- #
#
# Mode 1, "square": given a rectangle of explorer tiles plus a start and an end
# tile, find the shortest ride that passes through every tile of the rectangle.
#
# This is a Generalised TSP with a path (not cycle) objective: one cluster per
# tile, each cluster holding a few candidate junctions inside that tile, and the
# cost between two candidates being the shortest path on the road graph. The
# solver is nearest-neighbour, then 2-opt and Or-opt on the cluster order, then
# an exact DP that picks the best candidate within each cluster for that order.
#
# One refinement matters more than any of the above: a ride claims every tile it
# *passes through*, not only the ones it stops in. So after a tour is built the
# route is traced and tiles that are already swept incidentally have their own
# visit dropped, which typically takes a tenth off the distance.


def bike_directions(tags) -> tuple[bool, bool]:
    """
    (forward, backward): may a bicycle ride the way each way round?

    A plain `oneway=yes` blocks the contraflow, but French towns sign most of
    their one-way streets as double-sens cyclable, which OSM records as
    `oneway:bicycle=no` or a `cycleway=opposite*` tag. Those come back to
    bidirectional. `oneway:bicycle=yes` does the reverse: a street that cars
    may take both ways but bicycles only one.
    """
    ow = (tags.get("oneway") or "").lower()
    if not ow and (tags.get("junction") or "").lower() in ("roundabout", "circular"):
        ow = "yes"
    fwd = bwd = True
    if ow in ("yes", "true", "1"):
        bwd = False
    elif ow in ("-1", "reverse"):
        fwd = False

    cw = " ".join(str(tags.get(k) or "").lower() for k in
                  ("cycleway", "cycleway:both", "cycleway:left", "cycleway:right"))
    owb = (tags.get("oneway:bicycle") or "").lower()
    if owb in ("no", "false", "0") or "opposite" in cw:
        fwd = bwd = True
    elif owb in ("yes", "true", "1"):
        fwd, bwd = True, False
    elif owb in ("-1", "reverse"):
        fwd, bwd = False, True
    if not fwd and not bwd:               # contradictory tagging
        fwd = bwd = True
    return fwd, bwd


# The ideal: a perfect 45-degree line running exactly along tile corners
# touches three tiles per diagonal step of a*sqrt(2), so it claims a tile every
# a*sqrt(2)/3 of riding. No road network offers that, which is the point --
# the figure is a floor nothing can beat, not a target.
IDEAL_PER_TILE = math.sqrt(2.0) / 3.0          # ~0.4714 tile sides


def tile_side_m(lat: float) -> float:
    """Ground length of one explorer tile side at this latitude."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / NTILES


def theoretical_min_m(n_tiles: int, mid_lat: float) -> float:
    """The shortest ride that could conceivably claim n_tiles explorer tiles."""
    return n_tiles * tile_side_m(mid_lat) * IDEAL_PER_TILE


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat = r2 - r1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat * 0.5) ** 2
         + math.cos(r1) * math.cos(r2) * math.sin(dlon * 0.5) ** 2)
    return 2.0 * 6371008.8 * math.asin(math.sqrt(a))


def csr_min_duplicates(n: int, u, v, w):
    """
    Build a CSR adjacency keeping the SHORTEST weight for a repeated arc.

    scipy's coo -> csr conversion SUMS duplicates, which silently multiplied
    the length of every way that happens to be stored in more than one
    database block. The solver then optimised against distances that were two
    or three times too long near block boundaries.
    """
    from scipy.sparse import coo_matrix
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    if len(u) == 0:
        return coo_matrix((n, n)).tocsr()
    key = u * n + v
    order = np.lexsort((w, key))          # by arc, then by weight
    key_s, w_s, u_s, v_s = key[order], w[order], u[order], v[order]
    first = np.ones(len(key_s), dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]   # keep the cheapest of each arc
    return coo_matrix((w_s[first], (u_s[first].astype(np.int32),
                                    v_s[first].astype(np.int32))),
                      shape=(n, n)).tocsr()


class RoadGraph:
    """An undirected road graph: node coordinates plus a CSR adjacency."""

    def __init__(self, lon, lat, csr, osm_ids=None, directed=False):
        self.lon = lon
        self.lat = lat
        self.csr = csr
        self.osm_ids = osm_ids
        self.directed = bool(directed)
        self.n = len(lon)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, lon=self.lon, lat=self.lat,
            data=self.csr.data, indices=self.csr.indices, indptr=self.csr.indptr,
            shape=np.array(self.csr.shape, dtype=np.int64),
            directed=np.array([1 if self.directed else 0], dtype=np.int8),
        )

    @classmethod
    def load(cls, path: str) -> "RoadGraph":
        from scipy.sparse import csr_matrix
        with np.load(path, allow_pickle=False) as z:
            csr = csr_matrix((z["data"], z["indices"], z["indptr"]),
                             shape=tuple(int(v) for v in z["shape"]))
            dirc = bool(int(z["directed"][0])) if "directed" in z.files else False
            return cls(z["lon"], z["lat"], csr, directed=dirc)

    def tiles_of_nodes(self) -> np.ndarray:
        """(n, 2) int array of the z14 tile each node falls in."""
        x = (self.lon + 180.0) / 360.0 * NTILES
        s = np.sin(np.radians(np.clip(self.lat, -MAX_LAT, MAX_LAT)))
        y = (0.5 - np.log((1.0 + s) / (1.0 - s)) / (4.0 * math.pi)) * NTILES
        return np.stack([np.floor(x), np.floor(y)], axis=1).astype(np.int64)


def build_graph(paths: Sequence[str], bbox: tuple[float, float, float, float],
                accept, idx: str, verbose: bool = False,
                oneway: bool = True) -> RoadGraph:
    """Read the accepted ways inside bbox and assemble an undirected graph."""
    try:
        import osmium
    except ImportError:
        raise SystemExit("pyosmium is required:  pip install osmium")
    from scipy.sparse import coo_matrix

    west, south, east, north = bbox
    SLACK = 0.5                     # degrees, ~55 km: far longer than any way
    coords: dict[int, tuple[float, float]] = {}
    eu: list[int] = []
    ev: list[int] = []
    ew: list[float] = []
    seen = kept = 0
    t0 = time.time()

    def handle(way) -> None:
        nonlocal seen, kept
        seen += 1
        if verbose and seen % 500_000 == 0:
            # keyed on ways read, not ways kept: a small zone keeps almost
            # nothing, so a counter on `kept` would never print at all
            print(f"[graph]   {seen:,} ways read, {kept:,} in the zone, "
                  f"{len(coords):,} nodes, {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
        ok, _ = accept(way.tags)
        if not ok:
            return
        # Cheap reject first: look at a single node and drop the way unless it
        # lands within a generous slack of the area of interest. Highway ways
        # are a few km long at most, so a half-degree margin cannot discard one
        # that really does reach the box -- and this skips materialising the
        # coordinates of ~99.9% of a country's ways.
        try:
            first = None
            for n in way.nodes:
                if n.location.valid():
                    first = (n.lon, n.lat)
                    break
            if first is None:
                return
            if not (west - SLACK <= first[0] <= east + SLACK
                    and south - SLACK <= first[1] <= north + SLACK):
                return
            pts = [(n.ref, n.lon, n.lat) for n in way.nodes if n.location.valid()]
        except Exception:
            return
        if len(pts) < 2:
            return
        if not any(west <= lo <= east and south <= la <= north
                   for _, lo, la in pts):
            return
        kept += 1
        fwd, bwd = bike_directions(way.tags) if oneway else (True, True)
        for (ia, loa, laa), (ib, lob, lab) in zip(pts, pts[1:]):
            if ia == ib:
                continue
            coords[ia] = (loa, laa)
            coords[ib] = (lob, lab)
            d = haversine_m(loa, laa, lob, lab)
            if fwd:
                eu.append(ia); ev.append(ib); ew.append(d)
            if bwd:
                eu.append(ib); ev.append(ia); ew.append(d)


    for path in paths:
        if verbose:
            print(f"[graph] reading {path}", file=sys.stderr, flush=True)
        if hasattr(osmium, "FileProcessor"):
            fp = osmium.FileProcessor(path).with_locations(idx)
            fp = fp.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
            fp = fp.with_filter(osmium.filter.KeyFilter("highway"))
            for obj in fp:
                handle(obj)
        else:
            class _H(osmium.SimpleHandler):      # type: ignore[misc]
                def way(self, w):
                    handle(w)
            _H().apply_file(path, locations=True, idx=idx)

    if not coords:
        raise SystemExit("no roads found in that bounding box")

    ids = np.fromiter(coords.keys(), dtype=np.int64, count=len(coords))
    ids.sort()
    index = {int(v): i for i, v in enumerate(ids)}
    lon = np.array([coords[int(i)][0] for i in ids], dtype=np.float64)
    lat = np.array([coords[int(i)][1] for i in ids], dtype=np.float64)

    u = np.fromiter((index[i] for i in eu), dtype=np.int32, count=len(eu))
    v = np.fromiter((index[i] for i in ev), dtype=np.int32, count=len(ev))
    w = np.asarray(ew, dtype=np.float64)
    n = len(ids)
    csr = csr_min_duplicates(n, u, v, w)
    if verbose:
        print(f"[graph] {seen:,} ways read, {kept:,} kept -> {n:,} nodes, "
              f"{csr.nnz:,} arcs, {time.time()-t0:.0f}s",
              file=sys.stderr, flush=True)
    return RoadGraph(lon, lat, csr, ids, directed=oneway)


def components(g: RoadGraph):
    """(labels, sizes) of the connected components; strong when directed."""
    from scipy.sparse.csgraph import connected_components
    _, labels = connected_components(
        g.csr, directed=g.directed,
        connection="strong" if g.directed else "weak")
    return labels, np.bincount(labels)


def largest_component(g: RoadGraph) -> np.ndarray:
    """Boolean mask of the nodes in the biggest connected component."""
    labels, sizes = components(g)
    return labels == int(sizes.argmax())


def nodes_in_tile(g: RoadGraph, node_tiles: np.ndarray,
                  tile: tuple[int, int]) -> np.ndarray:
    return np.where((node_tiles[:, 0] == tile[0])
                    & (node_tiles[:, 1] == tile[1]))[0]


def seed_in_tile(g: RoadGraph, node_tiles: np.ndarray, tile,
                 labels: np.ndarray, sizes: np.ndarray):
    """
    A node inside `tile` to anchor the ride on: the one whose own component is
    the largest. Anchoring on an arbitrary node is not safe -- a one-way
    cul-de-sac is a component of size one, and seeding there would make the
    entire network look unreachable.
    """
    sel = nodes_in_tile(g, node_tiles, tile)
    if sel.size == 0:
        return None
    comp = sizes[labels[sel]]
    best = comp.max()
    pool = sel[comp == best]
    cx, cy = tilef_to_lonlat(tile[0] + 0.5, tile[1] + 0.5)
    d = (g.lon[pool] - cx) ** 2 + (g.lat[pool] - cy) ** 2
    return int(pool[int(d.argmin())]), int(best)


def ridable_mask(g: RoadGraph, seed_start: int, seed_end: int) -> np.ndarray:
    """
    Nodes the ride can actually use: reachable from the start node and able to
    reach the end node.

    More permissive than the largest strongly connected component -- a
    cul-de-sac you can ride into and back out of is usable, yet on a directed
    graph it forms a strong component of its own.
    """
    from scipy.sparse.csgraph import dijkstra, connected_components
    if not g.directed:
        _, labels = connected_components(g.csr, directed=False)
        return labels == labels[seed_start]
    out = np.isfinite(dijkstra(g.csr, directed=True, indices=seed_start))
    back = np.isfinite(dijkstra(g.csr.T.tocsr(), directed=True, indices=seed_end))
    return out & back


# --------------------------------------------------------------------------- #
# routing: candidates, distances                                               #
# --------------------------------------------------------------------------- #

def parse_latlon(v: str) -> tuple[float, float]:
    """`48.4712,-1.3856` -> (lat, lon). Latitude first, as every map shows it."""
    parts = [q.strip() for q in str(v).replace(";", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected LAT,LON (e.g. 48.4712,-1.3856), got {v!r}")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v!r} is not a pair of numbers")
    if not -90.0 <= lat <= 90.0:
        raise argparse.ArgumentTypeError(
            f"latitude {lat} is out of range -- the order is LAT,LON")
    if not -180.0 <= lon <= 180.0:
        raise argparse.ArgumentTypeError(f"longitude {lon} is out of range")
    return lat, lon


def snap_to_graph(g: RoadGraph, lat: float, lon: float,
                  allowed: np.ndarray | None = None) -> tuple[int, float]:
    """
    The graph node closest to a point, and how far away it is.

    Restricted to `allowed` when given -- normally the biggest component of
    the area. Snapping to the plain nearest node is a trap: a driveway stub or
    a fenced-off service road is a component of its own, and anchoring a ride
    there makes the whole network look unreachable. This is the same lesson
    seed_in_tile learned for tiles.

    Equirectangular distance is enough to rank candidates a few hundred metres
    apart, and the winner's distance is then measured properly.
    """
    idx = np.flatnonzero(allowed) if allowed is not None else None
    if idx is not None and idx.size == 0:
        raise SystemExit("no usable road anywhere in the loaded area")
    lons = g.lon if idx is None else g.lon[idx]
    lats = g.lat if idx is None else g.lat[idx]
    k = math.cos(math.radians(lat))
    d2 = ((lons - lon) * k) ** 2 + (lats - lat) ** 2
    j = int(np.argmin(d2))
    node = j if idx is None else int(idx[j])
    return node, haversine_m(lon, lat, float(g.lon[node]), float(g.lat[node]))


def pick_candidates(g: RoadGraph, node_tiles: np.ndarray, usable: np.ndarray,
                    tile: tuple[int, int], k: int) -> list[int]:
    """Up to k well-spread usable graph nodes inside one tile."""
    sel = np.where(usable
                   & (node_tiles[:, 0] == tile[0])
                   & (node_tiles[:, 1] == tile[1]))[0]
    if sel.size == 0:
        return []
    if sel.size <= k:
        return [int(i) for i in sel]
    # farthest-point sampling, seeded at the node nearest the tile centre
    cx, cy = tilef_to_lonlat(tile[0] + 0.5, tile[1] + 0.5)
    px, py = g.lon[sel], g.lat[sel]
    d0 = (px - cx) ** 2 + (py - cy) ** 2
    chosen = [int(d0.argmin())]
    far = (px - px[chosen[0]]) ** 2 + (py - py[chosen[0]]) ** 2
    while len(chosen) < k:
        nxt = int(far.argmax())
        if nxt in chosen:
            break
        chosen.append(nxt)
        far = np.minimum(far, (px - px[nxt]) ** 2 + (py - py[nxt]) ** 2)
    return [int(sel[c]) for c in chosen]


def many_to_many(g: RoadGraph, sources: Sequence[int], targets: Sequence[int],
                 batch: int = 48, verbose: bool = False) -> np.ndarray:
    """Shortest-path distance matrix between two node lists, in metres."""
    from scipy.sparse.csgraph import dijkstra
    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)
    out = np.empty((len(src), len(tgt)), dtype=np.float64)
    t0 = time.time()
    for i in range(0, len(src), batch):
        chunk = src[i:i + batch]
        d = dijkstra(g.csr, directed=g.directed, indices=chunk)
        out[i:i + len(chunk)] = d[:, tgt]
        if verbose:
            print(f"[dist]  {min(i+batch, len(src))}/{len(src)} sources, "
                  f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return out


class PathCache:
    """Shortest paths between graph nodes, reconstructed on demand."""

    def __init__(self, g: RoadGraph, flat: Sequence[int]):
        from scipy.sparse.csgraph import dijkstra
        self._dijkstra = dijkstra
        self.g = g
        # `flat` maps a solver index (a column of the distance matrix) to the
        # graph node it stands for. Every public method below takes solver
        # indices; nothing outside this class should handle graph nodes.
        self.flat = list(flat)
        self._pred: dict[int, np.ndarray] = {}
        self._tiles: dict[tuple[int, int], frozenset] = {}

    def _preds(self, src: int) -> np.ndarray:
        p = self._pred.get(src)
        if p is None:
            _, p = self._dijkstra(self.g.csr, directed=self.g.directed,
                                  indices=src, return_predecessors=True)
            self._pred[src] = p
        return p

    def nodes(self, pa: int, pb: int) -> list[int]:
        """Graph node sequence between two solver indices. Empty if unreachable."""
        a, b = self.flat[pa], self.flat[pb]
        if a == b:
            return [a]
        pred = self._preds(a)
        out = [b]
        cur = b
        guard = self.g.n + 2
        while cur != a and guard > 0:
            guard -= 1
            nxt = int(pred[cur])
            if nxt < 0:
                return []
            out.append(nxt)
            cur = nxt
        if cur != a:
            return []
        out.reverse()
        return out

    def tiles(self, pa: int, pb: int) -> frozenset:
        """The explorer tiles the shortest path between two solver indices sweeps."""
        key = (pa, pb) if pa <= pb else (pb, pa)
        hit = self._tiles.get(key)
        if hit is None:
            seq = self.nodes(pa, pb)
            cells: set[tuple[int, int]] = set()
            if seq:
                walk_polyline([(self.g.lon[i], self.g.lat[i]) for i in seq], cells)
            hit = frozenset(cells)
            self._tiles[key] = hit
        return hit


# --------------------------------------------------------------------------- #
# routing: the generalised TSP                                                 #
# --------------------------------------------------------------------------- #

class Progress:
    """Prints the best distance found so far, throttled so it stays readable."""

    def __init__(self, enabled: bool = True, every: float = 0.15,
                 theory_km: float = 0.0, tiles: int = 0):
        self.t0 = time.time()
        self.best = math.inf
        self.last = 0.0
        self.enabled = enabled
        self.every = every
        self.theory_km = theory_km      # the ideal floor, for the ratio
        self.tiles = tiles

    def update(self, km: float, note: str = "", force: bool = False) -> None:
        improved = km < self.best - 1e-6
        if improved:
            self.best = km
        if not self.enabled or (not improved and not force):
            return
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        extra = ""
        if self.theory_km > 0:
            extra += f"   {100.0 * km / self.theory_km:4.0f}% of ideal"
        if self.tiles:
            extra += f"   {km / self.tiles:.2f} km/tile"
        print(f"[{now - self.t0:7.1f}s] best route so far: {km:9.2f} km"
              f"{extra}{note}", file=sys.stderr, flush=True)


def cluster_matrix(D: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Cluster-to-cluster cost: the cheapest candidate pair between them."""
    c = len(groups)
    M = np.full((c, c), np.inf)
    for i, gi in enumerate(groups):
        rows = D[np.ix_(gi, range(D.shape[1]))]
        for j, gj in enumerate(groups):
            if i == j:
                M[i, j] = 0.0
            else:
                M[i, j] = rows[:, gj].min()
    return M


def nn_path(M: np.ndarray, start: int, end: int) -> list[int]:
    """Nearest-neighbour open path from start to end through every cluster."""
    c = M.shape[0]
    unvisited = set(range(c)) - {start, end}
    order = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: M[cur, j])
        order.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    order.append(end)
    return order


def two_opt(M: np.ndarray, order: list[int], deadline: float,
            on_improve=None) -> list[int]:
    """2-opt then Or-opt on an open path, endpoints held fixed."""
    o = np.array(order, dtype=np.int64)
    n = len(o)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 2-opt: reverse o[i..j]; on a symmetric matrix the interior is unchanged
        for i in range(1, n - 2):
            if time.time() > deadline:
                break
            a, b = o[i - 1], o[i]
            js = np.arange(i + 1, n - 1)
            cur = M[a, b] + M[o[js], o[js + 1]]
            new = M[a, o[js]] + M[b, o[js + 1]]
            gain = cur - new
            k = int(gain.argmax())
            if gain[k] > 1e-6:
                j = int(js[k])
                o[i:j + 1] = o[i:j + 1][::-1]
                improved = True
                if on_improve:
                    on_improve([int(v) for v in o])
        # Or-opt: move runs of 1..3 clusters elsewhere
        for seg in (1, 2, 3):
            for i in range(1, n - seg):
                if time.time() > deadline:
                    break
                j = i + seg - 1
                if j >= n - 1:
                    break
                prev, first, last, nxt = o[i - 1], o[i], o[j], o[j + 1]
                removed = M[prev, first] + M[last, nxt] - M[prev, nxt]
                if removed <= 1e-6:
                    continue
                rest = np.concatenate([o[:i], o[j + 1:]])
                ks = np.arange(0, len(rest) - 1)
                add = (M[rest[ks], first] + M[last, rest[ks + 1]]
                       - M[rest[ks], rest[ks + 1]])
                add_rev = (M[rest[ks], last] + M[first, rest[ks + 1]]
                           - M[rest[ks], rest[ks + 1]])
                best_f = int(add.argmin())
                best_r = int(add_rev.argmin())
                if add[best_f] <= add_rev[best_r]:
                    gain, pos, rev = removed - add[best_f], int(ks[best_f]), False
                else:
                    gain, pos, rev = removed - add_rev[best_r], int(ks[best_r]), True
                if gain > 1e-6:
                    seg_vals = o[i:j + 1][::-1] if rev else o[i:j + 1]
                    o = np.concatenate([rest[:pos + 1], seg_vals, rest[pos + 1:]])
                    improved = True
                    if on_improve:
                        on_improve([int(v) for v in o])
    return [int(v) for v in o]


def double_bridge(order: list[int], rng: random.Random) -> list[int]:
    """
    Cut the sequence in four and reconnect A-C-B-D.

    No sequence of 2-opt or Or-opt moves can undo this in one step, which is
    exactly the point: it lands the search somewhere genuinely different
    instead of walking back into the optimum it just left. The first and last
    visits stay put, since they are the start and finish tiles.
    """
    n = len(order)
    if n < 8:
        return list(order)
    p, q, r = sorted(rng.sample(range(1, n - 1), 3))
    return order[:p] + order[q:r] + order[p:q] + order[r:]


def iterated_local_search(M: np.ndarray, order: list[int], cost_of,
                          deadline: float, rng: random.Random,
                          accept=None, prog=None, label: str = ""):
    """
    Escape the local optimum the plain local search stops in.

    2-opt and Or-opt stop as soon as a whole pass finds no improvement, which
    on these instances happens within a second or two -- all the remaining
    time budget was being thrown away. Here each round perturbs the best known
    sequence with a double bridge, runs the local search again, and keeps the
    result only if it is better (and, when a check is supplied, still covers
    every required tile).
    """
    best = list(order)
    best_cost = cost_of(best)
    cur = best
    rounds = 0
    while time.time() < deadline:
        rounds += 1
        cand = two_opt(M, double_bridge(cur, rng), deadline)
        c = cost_of(cand)
        if c < best_cost - 1e-6 and (accept is None or accept(cand)):
            best, best_cost, cur = cand, c, cand
            if prog:
                prog.update(best_cost / 1000.0, label)
        else:
            cur = best
    return best, best_cost, rounds


def choose_candidates(D: np.ndarray, groups: list[list[int]], order: list[int],
                      fixed_first: int | None = None,
                      fixed_last: int | None = None
                      ) -> tuple[list[int], float]:
    """
    Exact DP: given the cluster order, pick the candidate in each cluster that
    minimises the total length. Returns (node indices, total metres).
    """
    layers = [groups[c] for c in order]
    if fixed_first is not None:
        layers[0] = [fixed_first]
    if fixed_last is not None:
        layers[-1] = [fixed_last]

    cost = [0.0] * len(layers[0])
    back: list[list[int]] = []
    for li in range(1, len(layers)):
        prev, cur = layers[li - 1], layers[li]
        sub = D[np.ix_(prev, cur)]
        tot = np.asarray(cost)[:, None] + sub
        arg = tot.argmin(axis=0)
        cost = [float(tot[arg[j], j]) for j in range(len(cur))]
        back.append([int(a) for a in arg])

    j = int(np.argmin(cost))
    total = float(cost[j])
    pick = [j]
    for li in range(len(layers) - 1, 0, -1):
        j = back[li - 1][j]
        pick.append(j)
    pick.reverse()
    return [layers[li][pick[li]] for li in range(len(layers))], total


# --------------------------------------------------------------------------- #
# routing: solve, prune, trace                                                 #
# --------------------------------------------------------------------------- #

def trace_route(cache: "PathCache", visits: Sequence[int]) -> list[int]:
    """Full node sequence of the ride, following shortest paths between visits."""
    out: list[int] = []
    for a, b in zip(visits, visits[1:]):
        leg = cache.nodes(a, b)
        if not leg:
            return []
        out.extend(leg if not out else leg[1:])
    return out or list(visits[:1])


def route_length_m(g: "RoadGraph", seq: Sequence[int]) -> float:
    if len(seq) < 2:
        return 0.0
    lo = g.lon[list(seq)]
    la = g.lat[list(seq)]
    return float(sum(haversine_m(lo[i], la[i], lo[i + 1], la[i + 1])
                     for i in range(len(seq) - 1)))


def covered_tiles(cache: "PathCache", visits: Sequence[int]) -> set:
    cells: set = set()
    for a, b in zip(visits, visits[1:]):
        cells |= cache.tiles(a, b)
    if visits:
        g = cache.g
        n0 = cache.flat[visits[0]]
        walk_polyline([(g.lon[n0], g.lat[n0])], cells)
    return cells


def prune_redundant(D: np.ndarray, groups: list[list[int]], order: list[int],
                    visits: list[int], cache: "PathCache",
                    required: set, fixed_first: int, fixed_last: int,
                    deadline: float, verbose: bool, prog=None,
                    keep_clusters: set | None = None
                    ) -> tuple[list[int], list[int], float]:
    """
    Drop visits whose tile the ride already sweeps on its way past.

    The cost of a visit is its detour: how much longer the ride is for going
    through it rather than straight from the previous visit to the next. The
    most expensive are tried first, and a removal is kept only if every
    required tile is still covered.

    Three things make this fast enough to actually finish:

      * the visits live in a doubly linked list, so a removal is O(1) and the
        neighbours of any visit are always current;
      * a max-heap holds the detours, re-pushed lazily when a neighbour moves;
      * coverage is counted per tile rather than recomputed. Removing a visit
        swaps two legs for one, so only those three legs are touched -- and
        the new leg starts at a node whose shortest-path tree is already in
        the cache, so it usually costs no Dijkstra at all.

    The earlier version re-ran the whole dynamic program and re-traced the
    entire route for every candidate removal, which on a 13x13 zone spent the
    whole budget on a handful of tests and left obvious there-and-back spurs
    in the result.

    `keep_clusters` names clusters that must survive whatever they cost: the
    waypoints you asked to ride through. Their tile is usually swept anyway on
    the way past, so without this the pruning would quietly delete the very
    thing you asked for.
    """
    import heapq

    n = len(visits)
    total = float(sum(D[visits[i], visits[i + 1]] for i in range(n - 1)))
    if n <= 3:
        return order, visits, total

    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(0, n - 1))
    alive = [True] * n

    legs: dict[int, frozenset] = {}
    cover: Counter = Counter()
    for i in range(n - 1):
        t = cache.tiles(visits[i], visits[i + 1])
        legs[i] = t
        cover.update(t)
    g = cache.g
    n0 = cache.flat[visits[0]]
    tx, ty = lonlat_to_tilef(g.lon[n0], g.lat[n0])
    cover[(math.floor(tx), math.floor(ty))] += 1

    def detour(i: int) -> float:
        p, q = prv[i], nxt[i]
        return float(D[visits[p], visits[i]] + D[visits[i], visits[q]]
                     - D[visits[p], visits[q]])

    locked = keep_clusters or set()
    heap = [(-detour(i), i) for i in range(1, n - 1)
            if order[i] not in locked]
    heapq.heapify(heap)
    removed = 0

    while heap and time.time() < deadline:
        negd, i = heapq.heappop(heap)
        if not alive[i] or prv[i] == -1 or nxt[i] == -1:
            continue
        d = detour(i)
        if d <= 1e-6:
            continue
        if -negd < d - 1e-6:            # stale: a neighbour moved, re-queue
            heapq.heappush(heap, (-d, i))
            continue

        p, q = prv[i], nxt[i]
        new = cache.tiles(visits[p], visits[q])
        if not new:
            continue                    # the shortcut is not ridable
        net: Counter = Counter()
        net.subtract(legs[p])
        net.subtract(legs[i])
        net.update(new)
        if any(c < 0 and t in required and cover[t] + c <= 0
               for t, c in net.items()):
            continue                    # a required tile would be lost

        for t, c in net.items():
            if c:
                cover[t] += c
        total -= d
        alive[i] = False
        nxt[p] = q
        prv[q] = p
        legs[p] = new
        legs.pop(i, None)
        removed += 1
        for j in (p, q):
            if alive[j] and prv[j] != -1 and nxt[j] != -1:
                heapq.heappush(heap, (-detour(j), j))
        if prog:
            prog.update(total / 1000.0, f"   ({n - removed} visits)")

    keep = []
    i = 0
    while i != -1:
        keep.append(i)
        i = nxt[i]
    order = [order[k] for k in keep]
    visits = [visits[k] for k in keep]
    total = float(sum(D[visits[i], visits[i + 1]] for i in range(len(visits) - 1)))

    # With fewer clusters the best junction inside each one may have changed,
    # so re-run the exact DP -- but only keep it if coverage survives.
    if time.time() < deadline and len(order) > 2:
        cand_visits, cand_total = choose_candidates(
            D, groups, order, fixed_first, fixed_last)
        if cand_total < total - 1e-6 and required <= covered_tiles(cache, cand_visits):
            visits, total = cand_visits, cand_total
            if prog:
                prog.update(total / 1000.0, f"   ({len(order)} visits)")

    if verbose:
        print(f"[prune] removed {removed} visit(s), {len(order)} left",
              file=sys.stderr, flush=True)
    return order, visits, total


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


def resample_track(lon: np.ndarray, lat: np.ndarray, step_m: float):
    """Points every step_m along the polyline, with their running distance."""
    seg = np.zeros(len(lon))
    for i in range(1, len(lon)):
        seg[i] = haversine_m(lon[i - 1], lat[i - 1], lon[i], lat[i])
    cum = np.cumsum(seg)
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 0:
        return np.array([0.0]), lon[:1].copy(), lat[:1].copy()
    want = np.arange(0.0, total + step_m, step_m)
    return want, np.interp(want, cum, lon), np.interp(want, cum, lat)


def smooth(values: np.ndarray, window_pts: int) -> np.ndarray:
    if window_pts < 2 or len(values) < 3:
        return values
    k = min(window_pts | 1, len(values) if len(values) % 2 else len(values) - 1)
    if k < 3:
        return values
    pad = k // 2
    padded = np.concatenate([np.full(pad, values[0]), values,
                             np.full(pad, values[-1])])
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")


def ascent_descent(elev: np.ndarray, threshold: float = 0.0
                   ) -> tuple[float, float]:
    """
    Metres climbed and descended, counting a reversal only once the profile
    has turned by `threshold` metres.

    Smoothing alone does not save you: summing every positive difference of a
    real DEM along a flat road still invents hundreds of metres. The
    hysteresis below is what trackers actually do -- a wobble smaller than the
    threshold is not a hill.
    """
    if len(elev) < 2:
        return 0.0, 0.0
    if threshold <= 0:
        d = np.diff(elev)
        return float(d[d > 0].sum()), float(-d[d < 0].sum())

    up = down = 0.0
    lo = hi = float(elev[0])
    trend = 0                                   # 0 unknown, 1 up, -1 down
    for e in map(float, elev[1:]):
        if trend >= 0:
            if e > hi:
                hi = e
            # an opening climb must establish the trend, or its ascent is
            # never credited when the profile finally turns over
            if trend == 0 and hi - lo >= threshold:
                trend = 1
            if hi - e >= threshold:
                if trend == 1:
                    up += hi - lo
                trend, lo = -1, e
        else:
            if e < lo:
                lo = e
            if e - lo >= threshold:
                down += hi - lo
                trend, hi = 1, e
    if trend == 1:
        up += hi - lo
    elif trend == -1:
        down += hi - lo
    return up, down


def build_profile(g: "RoadGraph", seq: Sequence[int], dem: DEM,
                  step_m: float, smooth_m: float, threshold_m: float = 3.0,
                  max_points: int = 1200):
    """Elevation profile of the ride, plus the climbing totals."""
    lon = np.array([g.lon[i] for i in seq], dtype=np.float64)
    lat = np.array([g.lat[i] for i in seq], dtype=np.float64)
    return profile_from_coords(lon, lat, dem, step_m, smooth_m,
                               threshold_m, max_points)


def profile_from_coords(lon: np.ndarray, lat: np.ndarray, dem: DEM,
                        step_m: float, smooth_m: float,
                        threshold_m: float = 3.0, max_points: int = 1200):
    dist, slon, slat = resample_track(lon, lat, step_m)
    raw = dem.sample(slon, slat)
    if np.isnan(raw).all():
        return None
    # fill the holes so a missing tile does not swallow the whole profile
    idx = np.arange(len(raw))
    good = ~np.isnan(raw)
    raw = np.interp(idx, idx[good], raw[good])
    sm = smooth(raw, int(round(smooth_m / step_m)))
    up, down = ascent_descent(sm, threshold_m)

    keep = max(1, len(sm) // max_points)
    thin = sm[::keep]
    return {
        "step_m": round(step_m * keep, 2),
        "elev": [int(round(v)) for v in thin],
        "up": round(up),
        "down": round(down),
        "min": int(round(float(sm.min()))),
        "max": int(round(float(sm.max()))),
        "km": round(float(dist[-1]) / 1000.0, 2),
        "covered": float(good.mean()),
        "src": sorted(dem.sources),
    }


# --------------------------------------------------------------------------- #
# routing: output                                                              #
# --------------------------------------------------------------------------- #

def write_gpx(path: str, g: "RoadGraph", seq: Sequence[int], name: str,
              dem=None) -> None:
    esc = (name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    ele = None
    if dem is not None:
        lon = np.array([g.lon[i] for i in seq])
        lat = np.array([g.lat[i] for i in seq])
        ele = dem.sample(lon, lat)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<gpx version="1.1" creator="tilehunt" '
                 'xmlns="http://www.topografix.com/GPX/1/1">\n'
                 f'  <metadata><name>{esc}</name></metadata>\n'
                 f'  <trk><name>{esc}</name><trkseg>\n')
        for k, i in enumerate(seq):
            if ele is not None and not math.isnan(ele[k]):
                fh.write(f'    <trkpt lat="{g.lat[i]:.7f}" lon="{g.lon[i]:.7f}">'
                         f'<ele>{ele[k]:.1f}</ele></trkpt>\n')
            else:
                fh.write(f'    <trkpt lat="{g.lat[i]:.7f}" '
                         f'lon="{g.lon[i]:.7f}"/>\n')
        fh.write('  </trkseg></trk>\n</gpx>\n')


def default_dem(given: str | None) -> str:
    """Without --dem, the standard folder -- built on demand if it is absent."""
    return given or str(DATA_DIR / "DEM_DB")


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


def cmd_route(a: argparse.Namespace) -> None:
    """
    Both routing modes, because only their first twenty lines differ.

      square  -- cover every tile of a rectangle
      seeker  -- collect a chosen set of tiles, wherever they are

    From the clusters onwards the problem is identical: one cluster per tile
    to cover, one visit per cluster, the shortest tour between two fixed ends.
    """
    seeker = bool(getattr(a, "want", None))
    if seeker:
        rect = {tuple(t) for t in a.want}
        if not rect:
            raise SystemExit("give at least one tile with --want")
    else:
        (ax, ay), (bx, by) = a.corners
        x0, x1 = min(ax, bx), max(ax, bx)
        y0, y1 = min(ay, by), max(ay, by)
        rect = {(x, y) for x in range(x0, x1 + 1)
                for y in range(y0, y1 + 1)}

    # Two ways to say where the ride begins: a tile, which lets the solver
    # choose the best junction inside it, or an exact point, which is pinned
    # to the nearest junction of the network and may sit outside the area.
    # The exact form wins when both are given.
    if seeker:
        ordered = sorted(rect)
        start_tile = a.start or ordered[0]
        end_tile = a.end or ordered[-1]
    else:
        start_tile = a.start or (x0, y0)
        end_tile = a.end or (x1, y1)
    if a.start_at:
        start_tile = tuple(int(v) for v in lonlat_to_tilef(a.start_at[1],
                                                           a.start_at[0]))
    if a.end_at:
        end_tile = tuple(int(v) for v in lonlat_to_tilef(a.end_at[1],
                                                         a.end_at[0]))
    if not seeker:
        for t, lab, exact in ((start_tile, "--start", a.start_at),
                              (end_tile, "--end", a.end_at)):
            if t not in rect and not exact:
                raise SystemExit(f"{lab} tile {t[0]},{t[1]} is outside "
                                 f"the rectangle")
    if seeker:
        # the smallest rectangle holding the wanted tiles and both ends: the
        # margin then widens it, so only the roads that could plausibly be
        # ridden are ever loaded
        span = rect | {start_tile, end_tile}
        x0 = min(t[0] for t in span); x1 = max(t[0] for t in span)
        y0 = min(t[1] for t in span); y1 = max(t[1] for t in span)
    # A point may legitimately sit outside the area, but not on another
    # continent. The usual cause is LAT,LON given the other way round, and
    # both halves of a swapped pair can be valid on their own, so range checks
    # cannot catch it -- distance can.
    #
    # The distance measured is how far the point lies OUTSIDE the area, not
    # how far it is from the middle of it. Measuring from the centre made the
    # check scale with the length of the ride: a legitimate Lille-to-Paris
    # seeker run put both ends 110 km from the midpoint and was refused. A
    # point inside the area is at zero distance however large the area is,
    # while a swapped LAT,LON still lands thousands of kilometres out.
    box_w, _, _, box_n = tile_bounds(x0, y0)
    _, box_s, box_e, _ = tile_bounds(x1, y1)

    def outside_km(lat: float, lon: float) -> float:
        near_lon = min(max(lon, box_w), box_e)
        near_lat = min(max(lat, box_s), box_n)
        return haversine_m(lon, lat, near_lon, near_lat) / 1000.0

    checks = [(a.start_at, "--start-at"), (a.end_at, "--end-at")]
    checks += [(v, "--via") for v in (a.via or [])]
    for pt, lab in checks:
        if pt is None:
            continue
        far = outside_km(pt[0], pt[1])
        if far > a.away_max:
            swap = outside_km(pt[1], pt[0])
            hint = (f"  Did you mean {pt[1]},{pt[0]}? That one is inside."
                    if swap == 0 else
                    (f"  Did you mean {pt[1]},{pt[0]}? That is only "
                     f"{swap:.0f} km out." if swap < far / 10 else ""))
            raise SystemExit(
                f"{lab} {pt[0]},{pt[1]} lies {far:,.0f} km outside the area "
                f"this ride covers. The order is LAT,LON.{hint}  "
                f"Raise --away-max to allow it.")

    w, _, _, n = tile_bounds(x0 - a.margin, y0 - a.margin)
    _, s, e, _ = tile_bounds(x1 + a.margin, y1 + a.margin)
    # an exact point may be outside the rectangle, so widen the window that is
    # loaded, with enough slack around it to hold a usable piece of network
    for pt in (a.start_at, a.end_at):
        if pt:
            pad = 0.03
            w, s = min(w, pt[1] - pad), min(s, pt[0] - pad)
            e, n = max(e, pt[1] + pad), max(n, pt[0] + pad)
    if seeker:
        box = (x1 - x0 + 1) * (y1 - y0 + 1)
        print(f"wanted      : {len(rect)} tile(s) to collect")
        print(f"window      : x {x0}..{x1}  y {y0}..{y1}  "
              f"({x1-x0+1} x {y1-y0+1} = {box} tiles, margin {a.margin}) -- "
              f"only this is loaded")
    else:
        print(f"rectangle   : x {x0}..{x1}  y {y0}..{y1}  "
              f"({x1-x0+1} x {y1-y0+1} = {len(rect)} tiles)")
    if not a.start_at and not a.end_at:
        print(f"start / end : {start_tile[0]},{start_tile[1]}  ->  "
              f"{end_tile[0]},{end_tile[1]}")

    # ---- graph ----------------------------------------------------------- #
    if a.graph_cache and os.path.exists(a.graph_cache):
        print(f"graph       : {a.graph_cache} (cached)")
        g = RoadGraph.load(a.graph_cache)
    elif a.db:
        g = load_db_graph(a.db, (w, s, e, n), a.accept, not a.no_cobbles,
                          a.strict_surface, a.trust_tracktype,
                          oneway=not a.ignore_oneway, verbose=a.verbose)
        print(f"graph       : from database {a.db}")
        if a.graph_cache:
            g.save(a.graph_cache)
    else:
        if not a.pbf:
            raise SystemExit("give --db, --pbf, or an existing --graph-cache")
        g = build_graph(a.pbf, (w, s, e, n), pick_classifier(a), a.idx,
                        a.verbose, oneway=not a.ignore_oneway)
        if a.graph_cache:
            g.save(a.graph_cache)
            print(f"graph       : cached to {a.graph_cache}")
    node_tiles = g.tiles_of_nodes()
    labels, sizes = components(g)
    big = labels == int(np.argmax(sizes))        # the network, as opposed to
    pinned: dict[str, dict] = {}                 # the odd isolated stub

    def anchor(pt, tile, label):
        """A node to anchor on: snapped to an exact point, or seeded in a tile."""
        if pt is None:
            hit = seed_in_tile(g, node_tiles, tile, labels, sizes)
            if hit is None:
                raise SystemExit(
                    f"no road at all inside the {label} tile -- check the "
                    f"tile id against the map, and that the database covers "
                    f"this area")
            return hit
        node, dist = snap_to_graph(g, pt[0], pt[1], big)
        rec = {"asked": [round(pt[1], 6), round(pt[0], 6)],
               "node": [round(float(g.lon[node]), 6),
                        round(float(g.lat[node]), 6)],
               "m": round(dist, 1)}
        if label.startswith("via"):
            pinned.setdefault("via", []).append(rec)
        else:
            pinned[label] = rec
        print(f"{label:<12}: {g.lat[node]:.5f},{g.lon[node]:.5f}  "
              f"({dist:.0f} m from the point you asked for)")
        if dist > a.snap_max:
            raise SystemExit(
                f"the nearest road is {dist:.0f} m away, more than "
                f"--snap-max {a.snap_max:.0f} m. Either the point is off the "
                f"network, or the database does not cover it. Check it with "
                f"`snap --at {pt[0]},{pt[1]}`, or raise --snap-max.")
        return node, int(sizes[labels[node]])

    hit_s = anchor(a.start_at, start_tile, "start")
    hit_e = anchor(a.end_at, end_tile, "end")
    (seed_s, comp_s), (seed_e, comp_e) = hit_s, hit_e
    pin_s = seed_s if a.start_at else None
    pin_e = seed_e if a.end_at else None

    # waypoints: points the ride must go through, in whatever order suits it
    vias = []
    for k, pt in enumerate(a.via or [], 1):
        node, _ = anchor(pt, None, f"via {k}")
        vias.append((node, tuple(int(v) for v in
                                 lonlat_to_tilef(pt[1], pt[0]))))
    usable = ridable_mask(g, seed_s, seed_e)
    print(f"graph       : {g.n:,} nodes, {g.csr.nnz:,} arcs "
          f"({'one-way respected' if g.directed else 'one-way ignored'}), "
          f"{int(usable.sum()):,} ridable between your start and end")
    print(f"anchors     : start in a component of {comp_s:,} nodes, "
          f"end in one of {comp_e:,}  (biggest in the area: {int(sizes.max()):,})")
    tiles_sorted = sorted(rect)
    groups: list[list[int]] = []
    kept_tiles: list[tuple[int, int]] = []
    empty: list[tuple[int, int]] = []
    for t in tiles_sorted:
        cands = pick_candidates(g, node_tiles, usable, t, a.cands)
        if cands:
            groups.append(cands)
            kept_tiles.append(t)
        else:
            empty.append(t)
    if empty:
        # A tile with no usable road cannot be ridden through, so it is dropped
        # from what the route must cover and the rest of the square is ridden
        # normally. This is a warning and not an error: a square touching a
        # lake, a motorway junction or a military zone is perfectly ordinary,
        # and refusing the whole thing over one such tile helps nobody.
        no_road = sum(1 for t in empty
                      if nodes_in_tile(g, node_tiles, t).size == 0)
        cut_off = len(empty) - no_road
        print(f"WARNING     : {len(empty)} of the {len(rect)} tile(s) "
              f"{'you asked for' if seeker else 'in the rectangle'} cannot "
              f"be covered and are skipped")
        print(f"              {no_road} with no usable road at all (red on "
              f"the map), {cut_off} with roads cut off from your start or end")
        for t in empty[:12]:
            why = ("no road" if nodes_in_tile(g, node_tiles, t).size == 0
                   else "cut off")
            print(f"   {t[0]},{t[1]}   {why}")
        if len(empty) > 12:
            print(f"   ... {len(empty)-12} more")
        rest = len(rect) - len(empty)
        print(f"              the other {rest} tile(s) are "
              + ("collected as asked" if seeker
                 else "ridden as usual; percentages below count those only"))
        if a.require_all:
            raise SystemExit("--require-all was given, so this is an error")
    required = set(kept_tiles)        # what the ride must cover: rectangle only

    # A pinned endpoint becomes a cluster of exactly one junction, which is all
    # it takes to force it: the DP picks a candidate per cluster, and a cluster
    # of one leaves it no choice. Inside the rectangle the tile's own cluster
    # is replaced; outside it a cluster is appended, which adds the approach
    # leg to the tour without adding a tile to cover.
    outside: list[tuple[int, int]] = []
    claimed: set = set()          # tiles whose own cluster is already pinned
    locked: set = set()           # clusters the pruning may not drop

    def tile_cluster(tile, label):
        """
        The cluster for a tile given as an endpoint rather than as a point.

        In square mode it is always one of the rectangle's own tiles. In
        seeker mode the start and the end are usually not tiles you asked to
        collect, so a cluster is appended for them: the ride has to begin and
        end somewhere, but that somewhere is not something to cover.
        """
        if tile in kept_tiles:
            return kept_tiles.index(tile)
        cands = pick_candidates(g, node_tiles, usable, tile, a.cands)
        if not cands:
            raise SystemExit(
                f"the {label} tile {tile[0]},{tile[1]} has no usable road "
                f"reachable from the rest of the ride")
        groups.append(cands)
        kept_tiles.append(tile)
        return len(groups) - 1

    def pin_cluster(node, tile):
        if tile in kept_tiles and tile not in claimed:
            i = kept_tiles.index(tile)
            groups[i] = [node]
            claimed.add(tile)
            return i
        groups.append([node])
        kept_tiles.append(tile)
        if tile not in required:
            outside.append(tile)
        return len(groups) - 1

    if pin_s is not None and pin_e is not None and pin_s == pin_e:
        si = ei = pin_cluster(pin_s, start_tile)
    else:
        if pin_s is not None:
            si = pin_cluster(pin_s, start_tile)
        if pin_e is not None:
            ei = pin_cluster(pin_e, end_tile)
        if pin_s is None:
            si = tile_cluster(start_tile, "start")
        if pin_e is None:
            ei = tile_cluster(end_tile, "end")
    for node, tile in vias:
        if node in (pin_s, pin_e):
            continue              # already an endpoint; asking twice is free
        locked.add(pin_cluster(node, tile))
    if vias:
        print(f"waypoints   : {len(vias)} forced; the solver decides the "
              f"order it rides them in")
    if outside and not seeker:
        print(f"approach    : {len(outside)} endpoint(s) outside the "
              f"rectangle -- the lead-in and lead-out are counted in the "
              f"distance but not in the ideal floor")
    flat = [nd for grp in groups for nd in grp]
    offs, k = [], 0
    for grp in groups:
        offs.append(list(range(k, k + len(grp))))
        k += len(grp)

    print(f"clusters    : {len(groups)} tiles, {len(flat)} candidate junctions")

    # ---- distances -------------------------------------------------------- #
    D = many_to_many(g, flat, flat, batch=a.batch, verbose=a.verbose)
    if not np.isfinite(D).all():
        big = np.nanmax(D[np.isfinite(D)]) * 10.0 + 1e6
        D = np.where(np.isfinite(D), D, big)

    # ---- solve ------------------------------------------------------------ #
    _, s_mid, _, n_mid = tile_bounds(x0, (y0 + y1) // 2)
    mid_lat = (s_mid + n_mid) * 0.5
    # The floor n*sqrt(2)/3 describes a diagonal sweep of a solid block. For a
    # handful of scattered tiles it means nothing, so it is not shown rather
    # than shown wrong.
    theory = 0.0 if seeker else theoretical_min_m(len(required), mid_lat)
    if theory:
        print(f"ideal floor : {theory/1000:.1f} km for {len(required)} tiles "
              f"({theoretical_min_m(1, mid_lat):.0f} m per tile)")

    prog = Progress(enabled=not a.no_progress,
                    theory_km=theory / 1000.0, tiles=len(required))
    deadline = time.time() + a.seconds
    M = cluster_matrix(D, offs)
    # 2-opt reverses stretches of the tour, which only preserves the cost on a
    # symmetric matrix. With one-way streets the matrix is not symmetric, so
    # the ORDER is searched on the average of the two directions while the
    # actual cost is always taken from the true directed matrix in the DP
    # below -- the reported distance is never the symmetrised approximation.
    M_ord = (M + M.T) * 0.5 if g.directed else M
    fixed_first = offs[si][0] if a.pin_ends else None
    fixed_last = offs[ei][0] if a.pin_ends else None

    def report(o):
        _, km = choose_candidates(D, offs, o, fixed_first, fixed_last)
        prog.update(km / 1000.0)

    def cost_of(o):
        _, km = choose_candidates(D, offs, o, fixed_first, fixed_last)
        return km

    rng = random.Random(a.seed)
    order = nn_path(M_ord, si, ei)
    report(order)
    order = two_opt(M_ord, order, deadline, on_improve=report)
    order, _, rounds = iterated_local_search(
        M_ord, order, cost_of, deadline, rng, prog=prog)

    visits, total = choose_candidates(D, offs, order, fixed_first, fixed_last)
    if rounds:
        print(f"search      : {rounds:,} double-bridge restarts")
    prog.update(total / 1000.0, force=True)
    print(f"first pass  : {total/1000:.1f} km over {len(order)} visits")

    cache = PathCache(g, flat)
    if not a.no_prune:
        before = total
        order, visits, total = prune_redundant(
            D, offs, order, visits, cache, required,
            visits[0], visits[-1], deadline + a.prune_seconds, a.verbose,
            prog, keep_clusters=locked)
        if total < before - 1e-6:
            prog.update(total / 1000.0, f"   ({len(order)} visits, "
                                        f"{100*(before-total)/before:.0f}% off)",
                        force=True)

        # The order was optimised for the full set of visits. With some of them
        # gone the sequence is no longer the best one, so run the local search
        # again over what is left -- and keep it only if coverage survives.
        if len(order) > 3:
            sub = list(order)
            Msub = M_ord[np.ix_(sub, sub)]

            def cost_sub(local):
                return cost_of([sub[k] for k in local])

            def ok_sub(local):
                o = [sub[k] for k in local]
                v, _ = choose_candidates(D, offs, o, fixed_first, fixed_last)
                return required <= covered_tiles(cache, v)

            local0 = two_opt(Msub, list(range(len(sub))),
                             time.time() + a.reopt_seconds * 0.25)
            local, _, r2 = iterated_local_search(
                Msub, local0, cost_sub, time.time() + a.reopt_seconds, rng,
                accept=ok_sub, prog=prog, label="   (reordered)")
            cand_order = [sub[k] for k in local]
            cand_visits, cand_total = choose_candidates(
                D, offs, cand_order, fixed_first, fixed_last)
            if (cand_total < total - 1e-6
                    and required <= covered_tiles(cache, cand_visits)):
                order, visits, total = cand_order, cand_visits, cand_total
                prog.update(total / 1000.0,
                            f"   (reordered, {len(order)} visits)", force=True)
            if r2:
                print(f"reorder     : {r2:,} restarts after pruning")

    seq = trace_route(cache, visits)
    if not seq:
        raise SystemExit("the visits are not all connected on the road network")

    # ---- verify ----------------------------------------------------------- #
    swept: set = set()
    walk_polyline([(g.lon[i], g.lat[i]) for i in seq], swept)
    missing = required - swept
    length = route_length_m(g, seq)

    print(f"route       : {length/1000:.1f} km, {len(seq):,} points, "
          f"{len(order)} explicit visits")
    if theory:
        print(f"vs ideal    : {100*length/theory:.0f}% of the floor "
              f"({length/theory:.2f}x),  "
              f"{length/1000/len(required):.2f} km per tile")
    else:
        print(f"per tile    : {length/1000/len(required):.2f} km "
              f"per wanted tile")
    print(f"coverage    : {len(required & swept)}/{len(required)} required tiles"
          + (f", {len(swept - required)} extra swept" if swept - required else ""))
    if missing:
        print(f"MISSING     : {len(missing)} tile(s) not covered -- "
              f"{sorted(missing)[:8]}")

    # ---- elevation --------------------------------------------------------- #
    profile = None
    dem = None
    if not a.no_dem:
        a.dem = default_dem(a.dem)
        lonr = np.array([g.lon[i] for i in seq], dtype=np.float64)
        latr = np.array([g.lat[i] for i in seq], dtype=np.float64)
        local = os.path.exists(os.path.join(a.dem, DEM_INDEX))
        if ensure_dem(a.dem, lonr, latr, a.dem_res, a.dem_src, a.verbose,
                      a.no_dem_fetch):
            dem = open_dem(a.dem)
            if local:
                print(f"elevation   : read from {os.path.basename(a.dem)} on "
                      f"disk, nothing fetched")
            profile = profile_from_coords(lonr, latr, dem, a.dem_step,
                                          a.dem_smooth, a.dem_threshold)
            if dem.missing:
                print(f"elevation   : {len(dem.missing)} spot(s) with no "
                      f"height data, interpolated across")
        if profile:
            print(f"source      : {', '.join(profile['src']) or 'none'}")
            print(f"elevation   : +{profile['up']:,} m / -{profile['down']:,} m"
                  f"   (min {profile['min']} m, max {profile['max']} m"
                  + ("" if profile["covered"] > 0.999
                     else f", {100*profile['covered']:.0f}% of the track covered")
                  + ")")
        elif dem is not None:
            print("elevation   : no height data anywhere on this route")

    # ---- output ----------------------------------------------------------- #
    if a.gpx:
        write_gpx(a.gpx, g, seq, a.name or "tilehunt route",
                  dem)
        print(f"gpx         : {a.gpx}")
    if a.out:
        payload = {
            "mode": "seeker" if seeker else "square",
            "coords": [[round(float(g.lon[i]), 6), round(float(g.lat[i]), 6)]
                       for i in seq],
            "visits": [[int(node_tiles[flat[p]][0]), int(node_tiles[flat[p]][1])]
                       for p in visits],
            "rect": None if seeker else [x0, y0, x1, y1],
            "profile": profile,
            "start": list(start_tile), "end": list(end_tile),
            "pinned": pinned or None,
            "skipped": [list(t) for t in empty] or None,
            "wanted": ([list(t) for t in sorted(rect)] if seeker else None),
            "stats": {
                "km": round(length / 1000.0, 2),
                "tiles": len(required),
                "covered": len(required & swept),
                "extra": len(swept - required),
                "visits": len(order),
                "points": len(seq),
                "theory": round(theory / 1000.0, 2),
                "ratio": round(length / theory, 3) if theory else None,
                "pct": round(100.0 * length / theory, 1) if theory else None,
                "kmPerTile": round(length / 1000.0 / max(len(required), 1), 3),
                "up": profile["up"] if profile else None,
                "down": profile["down"] if profile else None,
            },
        }
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"route json  : {a.out}")
        print(f"\nOverlay it with:\n  tilehunt.py buildmap --tiles <dataset> "
              f"--clip-to-mask --route {os.path.basename(a.out)} --out route.html")


# --------------------------------------------------------------------------- #
# the road database                                                            #
# --------------------------------------------------------------------------- #
#
# One pass over the .osm.pbf writes two things: the z14 tile dataset the map is
# built from, and a spatial database of the road geometry under Data/Database_OSM.
# After that the PBF is never needed again -- routing a zone loads only the few
# blocks that cover it.
#
# Layout
#   Database_OSM/
#     index.json          what zoom the blocks use, what the flag bits mean
#     b_08312_05704.npz   one file per block; the name is its block x and y
#
# A block is a tile at BLOCK_ZOOM (11 by default), so it spans 8 x 8 explorer
# tiles, roughly 12 km across. A 13x13 ride plus its margin touches nine of them.
#
# Each block stores ways as chains of node references rather than as an edge
# list, which is about half the size, and keeps the OSM node ids so that two
# blocks can be merged by simple de-duplication -- no geometric stitching.
#
# What goes in is deliberately wider than any one filter: every way that any
# cycling preset might accept, each carrying a byte of flags. The preset is
# then applied when the blocks are loaded, so changing --accept, --no-cobbles,
# --strict-surface or --trust-tracktype never means rebuilding the database.

BLOCK_ZOOM = 11                 # 8 x 8 explorer tiles per block
SHARD_ZOOM = 8                  # coarse buckets used only while building

F_PAVED = 1 << 0                # accepted by paved-cycle, most permissive reading
F_ROAD = 1 << 1                 # accepted by all-roads
F_SURFACE = 1 << 2              # carries an explicit paved surface tag
F_COBBLE = 1 << 3               # that surface is sett / cobblestone
F_TRACK1 = 1 << 4               # accepted only because tracktype=grade1
F_FWD = 1 << 5                  # a bicycle may ride it forwards
F_BWD = 1 << 6                  # ... and backwards

FLAG_DOC = {
    "paved_cycle": F_PAVED, "all_roads": F_ROAD, "explicit_paved_surface": F_SURFACE,
    "cobbled": F_COBBLE, "tracktype_grade1_only": F_TRACK1,
    "forward": F_FWD, "backward": F_BWD,
}


def db_flags(tags) -> int:
    """The byte of flags stored for one way, or 0 if it is of no use to us."""
    paved, _ = classify_way(tags, cobbles=True, strict_surface=False,
                            trust_tracktype=True)
    road, _ = classify_all_roads(tags)
    if not paved and not road:
        return 0

    f = 0
    if paved:
        f |= F_PAVED
    if road:
        f |= F_ROAD

    surface = (tags.get("surface") or "").lower()
    if surface:
        parts = [p.strip() for p in surface.split(";") if p.strip()]
        if parts and all(p in PAVED_SURFACES or p in COBBLE_SURFACES for p in parts):
            f |= F_SURFACE
            if any(p in COBBLE_SURFACES for p in parts):
                f |= F_COBBLE
    elif paved and (tags.get("highway") in ASSUME_UNPAVED):
        f |= F_TRACK1               # only tracktype=grade1 got it through

    fwd, bwd = bike_directions(tags)
    if fwd:
        f |= F_FWD
    if bwd:
        f |= F_BWD
    return f


def flag_mask(accept: str, cobbles: bool, strict_surface: bool,
              trust_tracktype: bool):
    """A predicate over the stored flags reproducing the command-line options."""
    def keep(f: np.ndarray) -> np.ndarray:
        if accept == "all-roads":
            ok = (f & F_ROAD) != 0
        else:                                        # paved-cycle
            ok = (f & F_PAVED) != 0
            if not trust_tracktype:
                ok &= (f & F_TRACK1) == 0
            if not cobbles:
                ok &= (f & F_COBBLE) == 0
            if strict_surface:
                ok &= (f & F_SURFACE) != 0
        return ok
    return keep


def block_of(lon: float, lat: float, z: int) -> tuple[int, int]:
    x, y = lonlat_to_tilef(lon, lat, z)
    return math.floor(x), math.floor(y)


def blocks_for_bbox(bbox, z: int) -> list[tuple[int, int]]:
    west, south, east, north = bbox
    x0, y0 = block_of(west, north, z)
    x1, y1 = block_of(east, south, z)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


class DBWriter:
    """
    Phase one: stream every usable way into coarse shards on disk.

    A way is written to every shard it touches, so that in phase two all the
    ways reaching a fine block are guaranteed to sit in that block's shard --
    a block at BLOCK_ZOOM is wholly inside one shard at SHARD_ZOOM.
    """

    def __init__(self, root: str):
        self.root = root
        self.tmp = os.path.join(root, "_shards")
        os.makedirs(self.tmp, exist_ok=True)
        self._fh: dict[tuple[int, int], object] = {}
        self.ways = 0
        self.refs = 0

    def _file(self, b):
        fh = self._fh.get(b)
        if fh is None:
            fh = open(os.path.join(self.tmp, f"s_{b[0]}_{b[1]}.bin"), "wb")
            self._fh[b] = fh
        return fh

    def add(self, tags, pts) -> None:
        """pts is a list of (osm_id, lon, lat)."""
        f = db_flags(tags)
        if not f or len(pts) < 2:
            return
        ids = np.fromiter((p[0] for p in pts), dtype="<i8", count=len(pts))
        lon = np.fromiter((p[1] for p in pts), dtype=np.float64, count=len(pts))
        lat = np.fromiter((p[2] for p in pts), dtype=np.float64, count=len(pts))
        loni = np.rint(lon * 1e7).astype("<i4")
        lati = np.rint(lat * 1e7).astype("<i4")
        head = np.array([len(pts), f], dtype="<i4").tobytes()
        rec = head + ids.tobytes() + loni.tobytes() + lati.tobytes()

        bx0, by0 = block_of(lon.min(), lat.max(), SHARD_ZOOM)
        bx1, by1 = block_of(lon.max(), lat.min(), SHARD_ZOOM)
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                self._file((bx, by)).write(rec)
        self.ways += 1
        self.refs += len(pts)

    def close_shards(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    def consolidate(self, meta: dict, verbose: bool = False) -> dict:
        """Phase two: split each shard into fine blocks and write them out."""
        self.close_shards()
        nblocks = 0
        nbytes = 0
        shards = sorted(os.listdir(self.tmp))
        t0 = time.time()
        step = BLOCK_ZOOM - SHARD_ZOOM
        for si, name in enumerate(shards):
            path = os.path.join(self.tmp, name)
            # which shard this file is, so that we only emit the blocks that
            # belong to it: a way straddling a shard boundary is present in
            # both shards and would otherwise have the neighbour's block
            # rewritten from a partial view of it, wiping out the real content
            part = name[2:-4].split("_")
            shard_xy = (int(part[0]), int(part[1]))
            per: dict[tuple[int, int], list] = {}
            with open(path, "rb") as fh:
                blob = fh.read()
            off = 0
            while off < len(blob):
                n, f = np.frombuffer(blob, dtype="<i4", count=2, offset=off)
                off += 8
                n = int(n)
                ids = np.frombuffer(blob, dtype="<i8", count=n, offset=off)
                off += 8 * n
                lon = np.frombuffer(blob, dtype="<i4", count=n, offset=off)
                off += 4 * n
                lat = np.frombuffer(blob, dtype="<i4", count=n, offset=off)
                off += 4 * n
                flo, fla = lon / 1e7, lat / 1e7
                bx0, by0 = block_of(flo.min(), fla.max(), BLOCK_ZOOM)
                bx1, by1 = block_of(flo.max(), fla.min(), BLOCK_ZOOM)
                for bx in range(bx0, bx1 + 1):
                    for by in range(by0, by1 + 1):
                        if (bx >> step, by >> step) != shard_xy:
                            continue          # not ours to write
                        per.setdefault((bx, by), []).append((ids, lon, lat, int(f)))
            os.remove(path)

            for (bx, by), ways in per.items():
                all_ids = np.concatenate([w[0] for w in ways])
                all_lon = np.concatenate([w[1] for w in ways])
                all_lat = np.concatenate([w[2] for w in ways])
                uniq, inv = np.unique(all_ids, return_inverse=True)
                ulon = np.zeros(len(uniq), dtype="<i4")
                ulat = np.zeros(len(uniq), dtype="<i4")
                ulon[inv] = all_lon
                ulat[inv] = all_lat
                lens = np.fromiter((len(w[0]) for w in ways), dtype="<i4",
                                   count=len(ways))
                wptr = np.zeros(len(ways) + 1, dtype="<i4")
                np.cumsum(lens, out=wptr[1:])
                out = os.path.join(self.root, f"b_{bx:05d}_{by:05d}.npz")
                np.savez_compressed(
                    out,
                    nid=uniq.astype("<i8"), lon=ulon, lat=ulat,
                    wptr=wptr, wnodes=inv.astype("<i4"),
                    wflags=np.fromiter((w[3] for w in ways), dtype=np.uint8,
                                       count=len(ways)),
                )
                nblocks += 1
                nbytes += os.path.getsize(out)
            if verbose:
                print(f"[db]    shard {si+1}/{len(shards)}, {nblocks} blocks, "
                      f"{nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

        info = dict(meta, block_zoom=BLOCK_ZOOM, zoom=ZOOM, blocks=nblocks,
                    bytes=nbytes, ways=self.ways, node_refs=self.refs,
                    flags=FLAG_DOC, built=time.strftime("%Y-%m-%d %H:%M"))
        with open(os.path.join(self.root, "index.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        return info


def cmd_profile(a: argparse.Namespace) -> None:
    """Attach an elevation profile to a route that has already been solved."""
    if not a.route:
        raise SystemExit("give --route JSON, or --runname NAME to take "
                         "Runs/Square/NAME/NAME.route.json")
    with open(a.route, "r", encoding="utf-8") as fh:
        rj = json.load(fh)
    cs = rj.get("coords") or []
    if len(cs) < 2:
        raise SystemExit(f"{a.route} holds no track")
    lon = np.array([c[0] for c in cs], dtype=np.float64)
    lat = np.array([c[1] for c in cs], dtype=np.float64)

    a.dem = default_dem(a.dem)
    if not ensure_dem(a.dem, lon, lat, a.dem_res, a.dem_src, False,
                      getattr(a, "no_dem_fetch", False)):
        raise SystemExit("no heights available")
    dem = open_dem(a.dem)
    prof = profile_from_coords(lon, lat, dem, a.dem_step, a.dem_smooth,
                               a.dem_threshold)
    if dem.missing:
        print(f"missing     : {len(dem.missing)} tile(s) -- download either:")
        for la, lo in sorted(dem.missing):
            print(f"   copernicus: {copernicus_url(la, lo)}")
            print(f"   srtm      : {hgt_url(la, lo)}")
    if not prof:
        raise SystemExit("no tile covers this route")

    rj["profile"] = prof
    rj.setdefault("stats", {})["up"] = prof["up"]
    rj["stats"]["down"] = prof["down"]
    with open(a.route, "w", encoding="utf-8") as fh:
        json.dump(rj, fh, separators=(",", ":"))
    print(f"source      : {', '.join(prof['src']) or 'none'}")
    print(f"elevation   : +{prof['up']:,} m / -{prof['down']:,} m   "
          f"(min {prof['min']} m, max {prof['max']} m)")
    print(f"updated     : {a.route}")

    if a.gpx:
        ele = dem.sample(lon, lat)
        esc = (a.name or "tilehunt route").replace("&", "&amp;")
        with open(a.gpx, "w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<gpx version="1.1" creator="tilehunt" '
                     'xmlns="http://www.topografix.com/GPX/1/1">\n'
                     f'  <metadata><name>{esc}</name></metadata>\n'
                     f'  <trk><name>{esc}</name><trkseg>\n')
            for k in range(len(lon)):
                if not math.isnan(ele[k]):
                    fh.write(f'    <trkpt lat="{lat[k]:.7f}" lon="{lon[k]:.7f}">'
                             f'<ele>{ele[k]:.1f}</ele></trkpt>\n')
                else:
                    fh.write(f'    <trkpt lat="{lat[k]:.7f}" '
                             f'lon="{lon[k]:.7f}"/>\n')
            fh.write('  </trkseg></trk>\n</gpx>\n')
        print(f"gpx         : {a.gpx}")


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


def cmd_snap(a: argparse.Namespace) -> None:
    """Where a point lands on the network, before committing to a long solve."""
    lat, lon = a.at
    pad = a.pad
    bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
    if a.db:
        g = load_db_graph(a.db, bbox, a.accept, not a.no_cobbles,
                          a.strict_surface, a.trust_tracktype,
                          oneway=not a.ignore_oneway, verbose=a.verbose)
    elif a.pbf:
        g = build_graph(a.pbf, bbox, pick_classifier(a), a.idx, a.verbose,
                        oneway=not a.ignore_oneway)
    else:
        raise SystemExit("give --db or --pbf")
    if g.n == 0:
        raise SystemExit(f"no road within {pad:.3f} degrees of the point -- "
                         f"the database may not cover it")
    labels, sizes = components(g)
    big = labels == int(np.argmax(sizes))

    tx, ty = lonlat_to_tilef(lon, lat)
    print(f"asked for   : {lat:.5f},{lon:.5f}   in tile "
          f"{int(tx)},{int(ty)}")
    print(f"loaded      : {g.n:,} nodes within {pad:.2f} degrees")

    near, d_near = snap_to_graph(g, lat, lon, None)
    node, dist = snap_to_graph(g, lat, lon, big)
    ntx, nty = lonlat_to_tilef(float(g.lon[node]), float(g.lat[node]))
    print(f"snaps to    : {g.lat[node]:.5f},{g.lon[node]:.5f}   "
          f"{dist:.0f} m away, in tile {int(ntx)},{int(nty)}")
    if node != near:
        print(f"note        : there is a closer node {d_near:.0f} m away, but "
              f"it sits on an isolated piece of road "
              f"({int(sizes[labels[near]]):,} node(s)); the ride would be "
              f"trapped there")
    if g.osm_ids is not None:
        print(f"osm node    : {int(g.osm_ids[node])}   "
              f"https://www.openstreetmap.org/node/{int(g.osm_ids[node])}")
    print(f"use it with : --start-at {lat},{lon}   (or --end-at)")


# --------------------------------------------------------------------------- #
# Statshunters: the tiles you have already ridden
#
# There is no OAuth for third parties, so there is no "sign in with
# Statshunters" to build. What Statshunters offers instead is a share link the
# user creates themselves, which exposes their activities as JSON. Every tool
# in this ecosystem works that way. The link is a secret -- whoever holds it
# can read the whole history -- so it is kept in a file of its own rather than
# typed into a command line that lands in a shell history.
# --------------------------------------------------------------------------- #

SHARE_RE = re.compile(r"^https?://(?:www\.)?statshunters\.com/share/(\w+)/?$")
SHARE_FILE = "statshunters.txt"


def share_url(value: str, base: str | None = None) -> str:
    """
    Accept a full share link or just its code, and return the link.

    `base` exists so the client can be pointed at a local stand-in in the
    tests; in normal use it is Statshunters itself. Refusing anything else by
    default is deliberate: a share link is a secret, and a typo that sent it
    to the wrong host would hand over the whole history.
    """
    v = str(value).strip().strip('"').strip("'")
    host = (base or "https://www.statshunters.com").rstrip("/")
    if base:
        m = re.match(r"^https?://[^/]+/share/(\w+)/?$", v)
        if m:
            return f"{host}/share/{m.group(1)}"
    m = SHARE_RE.match(v)
    if m:
        return f"{host}/share/{m.group(1)}"
    if re.fullmatch(r"\w{6,}", v):
        return f"{host}/share/{v}"
    raise SystemExit(
        f"{value!r} is not a Statshunters share link. Create one at "
        f"https://www.statshunters.com/share -- it looks like "
        f"https://www.statshunters.com/share/abcdef123456")


def remembered_share() -> str | None:
    path = DATA_DIR / SHARE_FILE
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def remember_share(link: str) -> None:
    path = DATA_DIR / SHARE_FILE
    path.write_text(
        "# Your Statshunters share link. Anyone holding it can read your\n"
        "# whole activity history, so keep this file out of git. Revoke or\n"
        "# replace it at https://www.statshunters.com/share\n"
        f"{link}\n", encoding="utf-8")
    print(f"remembered  : {path}  (keep it out of git)")


def fetch_activities(link: str, timeout: float = 60.0,
                     verbose: bool = False):
    """
    Every activity behind a share link, page by page.

    The endpoint is the share link with /api/activities appended and a page
    number; it answers with {"activities": [...]} and an empty list once there
    is nothing left. Pages hold 500 activities.
    """
    import urllib.error
    import urllib.request

    base = link.rstrip("/") + "/api/activities"
    seen = 0
    for page in itertools.count(1):
        url = f"{base}?page={page}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "tilehunt/1.0 (explorer tile routing)",
                          "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SystemExit(
                    f"Statshunters answered 404 for {url}\n"
                    f"The share link is wrong, or it has been revoked. "
                    f"Create a new one at https://www.statshunters.com/share")
            raise SystemExit(f"Statshunters answered {exc.code} for {url}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"could not reach Statshunters: {exc.reason}")
        except json.JSONDecodeError:
            raise SystemExit(
                f"{url} did not answer with JSON. If you pasted the link in a "
                f"browser and saw your page, the link is fine and the API "
                f"path may have changed.")

        acts = payload.get("activities")
        if acts is None:
            raise SystemExit(f"unexpected answer from {url}: no 'activities'")
        if not acts:
            return
        seen += len(acts)
        if verbose:
            print(f"   page {page}: {len(acts)} activities "
                  f"({seen:,} so far)", flush=True)
        yield from acts


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


def cmd_visited(a: argparse.Namespace) -> None:
    """Pull the tiles you have already ridden out of Statshunters."""
    link = (share_url(a.share, a.share_base) if a.share
            else remembered_share())
    if not link:
        raise SystemExit(
            "no share link given and none remembered.\n"
            "Create one at https://www.statshunters.com/share, then:\n"
            "  python tilehunt.py genVisited --share "
            "https://www.statshunters.com/share/YOURCODE")
    print(f"share link  : {link[:38]}...  (kept out of this output)")

    want_types = {t.strip().lower() for t in a.type.split(",")} if a.type else None
    tiles: set[tuple[int, int]] = set()
    kept = skipped = 0
    oldest = newest = None
    for act in fetch_activities(link, a.timeout, a.verbose):
        when = str(act.get("date") or act.get("start_date") or "")[:10]
        if a.since and when and when < a.since:
            skipped += 1
            continue
        if a.until and when and when > a.until:
            skipped += 1
            continue
        if want_types and str(act.get("type", "")).lower() not in want_types:
            skipped += 1
            continue
        kept += 1
        if when:
            oldest = when if oldest is None else min(oldest, when)
            newest = when if newest is None else max(newest, when)
        for t in act.get("tiles") or ():
            try:
                tiles.add((int(t["x"]), int(t["y"])))
            except (KeyError, TypeError, ValueError):
                continue

    if not tiles:
        raise SystemExit(
            "no tiles came back. If you filtered by --type or by date, try "
            "without; otherwise the account may have no processed activities "
            "yet (Statshunters imports in the background).")

    print(f"activities  : {kept:,} kept"
          + (f", {skipped:,} filtered out" if skipped else ""))
    if oldest:
        print(f"period      : {oldest} to {newest}")
    grid = TileGrid.from_tiles(sorted(tiles), meta={
        "source": "statshunters", "activities": kept,
        "from": oldest, "to": newest,
        "type": a.type or "all",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S")})
    side, sx_, sy_ = largest_square(grid.bits)
    print(f"tiles       : {len(tiles):,} visited, "
          f"spanning x {grid.x0}..{grid.x0+grid.w-1}  "
          f"y {grid.y0}..{grid.y0+grid.h-1}")
    print(f"max square  : {side} x {side}  at "
          f"{grid.x0+sx_},{grid.y0+sy_} .. "
          f"{grid.x0+sx_+side-1},{grid.y0+sy_+side-1}")
    grid.meta["max_square"] = [int(side), int(grid.x0 + sx_),
                               int(grid.y0 + sy_)]
    grid.save(a.out)
    print(f"written     : {a.out}")
    if a.share and a.remember:
        remember_share(link)
    print(f"use it with : --visited {os.path.basename(a.out)}")


def cmd_dbinfo(a: argparse.Namespace) -> None:
    """Say exactly what the database holds for a tile, and what it does not."""
    root = a.db
    idx_path = os.path.join(root, "index.json")
    if not os.path.exists(idx_path):
        raise SystemExit(f"{root}/index.json is missing -- the build did not finish")
    with open(idx_path, encoding="utf-8") as fh:
        info = json.load(fh)
    bz = int(info.get("block_zoom", BLOCK_ZOOM))
    print(f"index       : block zoom {bz}, {info.get('blocks','?')} blocks, "
          f"{info.get('bytes',0)/1e6:,.0f} MB, {info.get('ways','?'):,} ways, "
          f"built {info.get('built','?')}")
    present = [f for f in os.listdir(root) if f.startswith("b_") and f.endswith(".npz")]
    print(f"on disk     : {len(present)} block files"
          + ("  <-- FEWER THAN THE INDEX SAYS" if len(present) < int(info.get("blocks", 0))
             else ""))
    if os.path.isdir(os.path.join(root, "_shards")):
        left = len(os.listdir(os.path.join(root, "_shards")))
        print(f"WARNING     : _shards/ still holds {left} file(s) -- the "
              f"consolidation did not run to the end")

    tiles = list(a.tile or [])
    if a.corners:
        (ax, ay), (bx, by) = a.corners
        tiles += [(x, y) for x in range(min(ax, bx), max(ax, bx) + 1)
                  for y in range(min(ay, by), max(ay, by) + 1)]
    seen_blocks = {}
    empties = []
    for t in tiles:
        blk = (t[0] >> (ZOOM - bz), t[1] >> (ZOOM - bz))
        path = os.path.join(root, f"b_{blk[0]:05d}_{blk[1]:05d}.npz")
        if blk not in seen_blocks:
            if not os.path.exists(path):
                seen_blocks[blk] = None
            else:
                with np.load(path, allow_pickle=False) as z:
                    seen_blocks[blk] = {
                        "size": os.path.getsize(path),
                        "nid": z["nid"], "lon": z["lon"], "lat": z["lat"],
                        "wflags": z["wflags"], "wptr": z["wptr"],
                    }
        b = seen_blocks[blk]
        if b is None:
            print(f"{t[0]},{t[1]}  block {blk[0]},{blk[1]}  FILE MISSING  {path}")
            empties.append(t)
            continue
        lon = b["lon"] / 1e7
        lat = b["lat"] / 1e7
        x = np.floor((lon + 180.0) / 360.0 * NTILES).astype(np.int64)
        sn = np.sin(np.radians(np.clip(lat, -MAX_LAT, MAX_LAT)))
        y = np.floor((0.5 - np.log((1 + sn) / (1 - sn)) / (4 * math.pi))
                     * NTILES).astype(np.int64)
        here = int(((x == t[0]) & (y == t[1])).sum())
        if here == 0:
            empties.append(t)
        if len(tiles) <= 40 or here == 0:
            print(f"{t[0]},{t[1]}  block {blk[0]},{blk[1]}  "
                  f"{b['size']/1e6:.2f} MB  {len(b['nid']):,} nodes, "
                  f"{len(b['wflags']):,} ways in block  ->  {here:,} nodes in "
                  f"this tile" + ("   <-- EMPTY" if here == 0 else ""))

    if seen_blocks:
        any_b = next((v for v in seen_blocks.values() if v is not None), None)
        if any_b is not None:
            f = any_b["wflags"].astype(np.int64)
            print("\nflag census of one block:")
            for name, bit in FLAG_DOC.items():
                print(f"   {name:24s} {int(((f & bit) != 0).sum()):,}")
    print(f"\n{len(empties)} of {len(tiles)} tile(s) hold no node at all")


def load_db_graph(root: str, bbox, accept: str, cobbles: bool,
                  strict_surface: bool, trust_tracktype: bool,
                  oneway: bool = True, verbose: bool = False) -> "RoadGraph":
    """Assemble a RoadGraph from the blocks covering bbox."""
    from scipy.sparse import coo_matrix

    idx_path = os.path.join(root, "index.json")
    if not os.path.exists(idx_path):
        raise SystemExit(f"{root} has no index.json -- build it with "
                         f"`tiles --db {os.path.basename(root)}`")
    with open(idx_path, encoding="utf-8") as fh:
        info = json.load(fh)
    bz = int(info.get("block_zoom", BLOCK_ZOOM))
    keep_fn = flag_mask(accept, cobbles, strict_surface, trust_tracktype)

    coords: dict[int, tuple[int, int]] = {}
    chains: list[tuple[np.ndarray, int]] = []
    found = missing = 0
    for bx, by in blocks_for_bbox(bbox, bz):
        path = os.path.join(root, f"b_{bx:05d}_{by:05d}.npz")
        if not os.path.exists(path):
            missing += 1
            continue
        found += 1
        with np.load(path, allow_pickle=False) as z:
            nid, lon, lat = z["nid"], z["lon"], z["lat"]
            wptr, wnodes, wflags = z["wptr"], z["wnodes"], z["wflags"]
        keep = keep_fn(wflags.astype(np.int64))
        for i in np.where(keep)[0]:
            seg = wnodes[wptr[i]:wptr[i + 1]]
            ids = nid[seg]
            for j, osm in enumerate(ids):
                if osm not in coords:
                    coords[int(osm)] = (int(lon[seg[j]]), int(lat[seg[j]]))
            chains.append((ids, int(wflags[i])))
    print(f"database    : {found} block(s) loaded"
          + (f", {missing} ABSENT -- the area is only partly built" if missing
             else "") + f", {len(chains):,} ways kept")
    if not chains:
        raise SystemExit("no roads in that area -- is the database built for it?")

    ids_sorted = np.array(sorted(coords), dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(ids_sorted)}
    lon = np.array([coords[int(i)][0] for i in ids_sorted], dtype=np.float64) / 1e7
    lat = np.array([coords[int(i)][1] for i in ids_sorted], dtype=np.float64) / 1e7

    eu: list[int] = []
    ev: list[int] = []
    ew: list[float] = []
    for ids, f in chains:
        fwd = bool(f & F_FWD) or not oneway
        bwd = bool(f & F_BWD) or not oneway
        loc = [pos[int(i)] for i in ids]
        for p, q in zip(loc, loc[1:]):
            if p == q:
                continue
            d = haversine_m(lon[p], lat[p], lon[q], lat[q])
            if fwd:
                eu.append(p); ev.append(q); ew.append(d)
            if bwd:
                eu.append(q); ev.append(p); ew.append(d)

    n = len(ids_sorted)
    csr = csr_min_duplicates(n, eu, ev, ew)
    return RoadGraph(lon, lat, csr, ids_sorted, directed=oneway)


# --------------------------------------------------------------------------- #
# the viewer                                                                   #
# --------------------------------------------------------------------------- #

VIEWER_HTML = r"""<!DOCTYPE html>
<!-- generated by tilehunt.py -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22cc; --line:#2b3340; --ink:#e6edf3;
    --dim:#8b97a7; --green:#2ea043; --red:#da3633; --accent:#539bf5;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);
    color:var(--ink);font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  /* a canvas is a replaced element: inset:0 alone leaves it at its intrinsic
     300x150, so width/height must be given explicitly */
  #map{position:absolute;inset:0;width:100%;height:100%;display:block;
    cursor:grab;touch-action:none}
  #map.drag{cursor:grabbing}
  .panel{position:absolute;background:var(--panel);backdrop-filter:blur(10px);
    border:1px solid var(--line);border-radius:10px;padding:10px 12px;
    box-shadow:0 8px 24px #0008}
  #hud{top:12px;left:12px;width:330px}
  #hud h1{margin:0 0 6px;font-size:14px;font-weight:650;letter-spacing:.2px}
  #hud .sub{color:var(--dim);font-size:11px;margin-bottom:8px}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-variant-numeric:tabular-nums}
  .kv b{font-weight:600}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
    vertical-align:baseline}
  /* one column down the left: legend, View, Layers, in that order */
  #leftstack{position:absolute;left:12px;bottom:12px;display:flex;
    flex-direction:column;gap:10px;align-items:flex-start;
    max-height:calc(100% - 24px);overflow-y:auto}
  #leftstack .panel{position:static;flex:none}
  #layers{padding:9px 10px}
  #layers h2,#actions h2,#legend h2{margin:0 0 7px;font-size:10px;
    text-transform:uppercase;letter-spacing:.7px;color:var(--dim);font-weight:600}
  #layers .btns{display:grid;grid-template-columns:1fr 1fr;gap:5px}
  /* actions, bottom right */
  #routebox{min-width:184px;padding:9px 10px}
  #plan{min-width:250px;width:100%;box-sizing:border-box;padding:9px 10px}
  /* the elevation profile sits between the two bottom panels */
  #profile{left:236px;right:12px;bottom:22px;padding:8px 10px}
  #profile .head{display:flex;gap:14px;align-items:baseline;
    font-size:11px;color:var(--dim);margin-bottom:4px}
  #profile .head b{color:var(--ink);font-weight:600;
    font-variant-numeric:tabular-nums}
  #prof{display:block;width:100%;height:86px;cursor:crosshair}
  @media (max-width:1100px){ #profile{display:none!important} }
  #plan-text{white-space:pre-wrap;line-height:1.5;max-height:46vh;
    overflow-y:auto;word-break:normal;overflow-wrap:anywhere}
  #plan ol{margin:0 0 6px;padding-left:18px;color:var(--dim);font-size:11px}
  #plan li.now{color:var(--ink);font-weight:600}
  #plan li.done{color:var(--green);text-decoration:line-through}
  #run-name{width:100%;box-sizing:border-box;padding:5px 7px;border-radius:5px;
    border:1px solid var(--line);background:rgba(0,0,0,0.25);color:var(--ink);
    font:500 12px ui-sans-serif,system-ui,sans-serif}
  #run-name:focus{outline:none;border-color:#ffd24a}
  #run-name.bad{border-color:var(--red)}
  #run-path{margin-top:4px;color:var(--dim);word-break:break-all;
    font:400 10px ui-monospace,SFMono-Regular,Menlo,monospace}
  #map.plan{cursor:crosshair}
  #actions{padding:9px 10px;display:flex;
    flex-direction:column;gap:5px;min-width:150px;width:100%;
    box-sizing:border-box}
  #actions button{width:100%;text-align:left}
  #legend{padding:9px 10px;min-width:168px;width:100%;box-sizing:border-box}
  button{white-space:nowrap}
  button{font:inherit;color:var(--ink);background:#21262d;border:1px solid var(--line);
    border-radius:7px;padding:5px 10px;cursor:pointer}
  button:hover{background:#2d333b;border-color:#3d4654}
  button.on{background:#1f6feb33;border-color:var(--accent);color:#cfe3ff}
  /* nothing sits bottom-right any more, so the column runs the full height
     and the commands can be read in one go */
  #rightstack{position:absolute;top:12px;right:12px;display:flex;
    flex-direction:column;gap:10px;align-items:flex-end;max-width:430px;
    max-height:calc(100% - 24px);overflow-y:auto;padding-right:2px}
  #rightstack .panel{position:static}
  #read{min-width:220px;font-variant-numeric:tabular-nums}
  #read .t{font-size:16px;font-weight:650;letter-spacing:.3px}
  #zone{position:absolute;right:12px;top:140px;min-width:240px;display:none}
  #zone h2{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.6px;
    color:var(--dim);font-weight:600}
  code{background:#0b0f14;border:1px solid var(--line);border-radius:5px;
    padding:1px 5px;font-size:11px;display:inline-block;word-break:break-all}
  .hint{color:var(--dim);font-size:11px;margin-top:6px}
  #note{color:#f0b429;font-size:11px;margin-top:8px;border-top:1px solid var(--line);
    padding-top:7px;line-height:1.4}
  #scale{position:absolute;left:50%;transform:translateX(-50%);bottom:14px;
    color:var(--dim);font-size:11px;text-align:center}
  #scale .ln{height:4px;border:1px solid var(--dim);border-top:0}
  #attr{position:absolute;right:6px;bottom:2px;color:#6b7684;font-size:10px}
  #attr a{color:#6b7684}
  @media (max-width:760px){
    #hud{max-width:none;right:12px}
    #read{top:auto;bottom:112px;right:12px}
    #zone{display:none!important}
  }
</style>
</head>
<body>
<canvas id="map"></canvas>

<div class="panel" id="hud">
  <h1 id="ttl"></h1>
  <div class="sub" id="sub"></div>
  <div class="kv">
    <b><span class="sw" style="background:var(--green)"></span><span id="lab-green">green</span></b><span id="s-green"></span>
    <b><span class="sw" style="background:var(--red)"></span><span id="lab-red">red</span></b><span id="s-red"></span>
    <b><span class="sw" style="background:#39404d"></span><span id="lab-out">out</span></b><span id="s-out"></span>
    <b><span id="lab-pct">covered</span></b><span id="s-pct"></span>
  </div>
  <div class="kv" id="vis-kv" style="display:none">
    <b><span class="sw" style="background:#4aa3ff"></span>ridden</b><span id="s-have"></span>
    <b><span class="sw" style="background:var(--green)"></span>to get</b><span id="s-need"></span>
    <b>max square</b><span id="s-sq"></span>
  </div>
  <div id="note"></div>
</div>

<div id="rightstack">
<div class="panel" id="read">
  <div class="t" id="r-tile">&mdash;</div>
  <div class="kv" style="margin-top:4px">
    <b>status</b><span id="r-stat">&mdash;</span>
    <b>centre</b><span id="r-ll">&mdash;</span>
    <b>size</b><span id="r-sz">&mdash;</span>
    <b>road</b><span id="r-km">&mdash;</span>
  </div>
  <div class="hint">Shift-click two tiles to measure a zone.</div>
</div>

<div class="panel" id="routebox" style="display:none">
  <h2>Route</h2>
  <div class="kv" id="route-rows"></div>
</div>

<div class="panel" id="plan" style="display:none">
  <h2 id="plan-title">SquarePlanner</h2>
  <ol id="plan-steps-square">
    <li id="ps1">Ctrl-drag a rectangle of tiles</li>
    <li id="ps2">Click the exact point you start from</li>
    <li id="ps3">Click the exact point you finish at</li>
    <li id="ps4">Add waypoints &mdash; optional
      <button id="b-via-done" style="display:none">done</button></li>
    <li id="ps5">Name the ride</li>
  </ol>
  <ol id="plan-steps-seek" style="display:none">
    <li id="qs1">Click the exact point you start from</li>
    <li id="qs2">Click the exact point you finish at</li>
    <li id="qs3">Click the tiles to collect &mdash; ctrl-drag for a block
      <button id="b-seek-done" style="display:none">done</button></li>
    <li id="qs4">Add waypoints &mdash; optional
      <button id="b-seek-via-done" style="display:none">done</button></li>
    <li id="qs5">Name the ride</li>
  </ol>
  <div class="kv" id="via-list" style="display:none"></div>
  <div id="plan-name" style="display:none;margin:6px 0 2px">
    <input id="run-name" type="text" spellcheck="false" autocomplete="off"
           placeholder="name this ride">
    <div id="run-path"></div>
  </div>
  <div class="kv" id="plan-rows"></div>
  <div id="plan-cmd" style="margin-top:8px;display:none">
    <code id="plan-text"></code>
    <div style="margin-top:6px;display:flex;gap:6px">
      <button id="b-copy">copy</button>
      <button id="b-plan-reset">start over</button>
    </div>
  </div>
</div>
</div>

<div class="panel" id="zone">
  <h2>Zone</h2>
  <div class="kv">
    <b>corners</b><span id="z-cor"></span>
    <b>grid</b><span id="z-grid"></span>
    <b>green</b><span id="z-green"></span>
    <b>red</b><span id="z-red"></span>
    <b>max square</b><span id="z-sq"></span>
  </div>
  <div style="margin-top:8px"><code id="z-cmd"></code></div>
  <div class="hint">“max square” = largest all-green square inside this zone.</div>
</div>

<div id="leftstack">
<div class="panel" id="legend" style="display:none">
  <h2>km of road per tile</h2>
  <div id="leg-rows" class="kv"></div>
</div>

<div class="panel" id="actions">
  <h2>View</h2>
  <button id="b-mode">density</button>
  <button id="b-plan">SquarePlanner</button>
  <button id="b-seek">TileSeeker</button>
  <button id="b-fit-route">zoom to route</button>
  <button id="b-sq">find max square</button>
  <button id="b-reset">reset view</button>
</div>

<div class="panel" id="layers">
  <h2>Layers</h2>
  <div class="btns">
    <button id="b-tiles" class="on">tiles</button>
    <button id="b-visited" class="on">visited</button>
    <button id="b-green" class="on">remove green</button>
    <button id="b-red" class="on">remove red</button>
    <button id="b-base" class="on">basemap</button>
    <button id="b-cities" class="on">cities</button>
    <button id="b-grid" class="on">grid</button>
    <button id="b-route" class="on">route</button>
    <button id="b-reg" class="on">regions</button>
    <button id="b-dep" class="on">departments</button>
  </div>
</div>
</div>

<div class="panel" id="profile" style="display:none">
  <div class="head">
    <span>elevation</span>
    <b id="pf-up"></b><b id="pf-down"></b>
    <span id="pf-at" style="margin-left:auto"></span>
  </div>
  <canvas id="prof"></canvas>
</div>

<div id="scale"><div id="sc-lab">&mdash;</div><div class="ln" id="sc-ln"></div></div>
<div id="attr"></div>

<script>
"use strict";
const D = __DATA__;

/* ---------- unpack bitsets ---------------------------------------------- */
function unpack(b64, n){
  if(!b64) return null;
  const raw = atob(b64), out = new Uint8Array(n);
  for(let i=0;i<n;i++){
    out[i] = (raw.charCodeAt(i>>3) >> (7-(i&7))) & 1;
  }
  return out;
}
const W = D.w, H = D.h, NT = W*H;
const ROAD = unpack(D.road, NT);
const LAND = unpack(D.land, NT);
const VISITED = unpack(D.visited, NT);
let showVisited = true;
const Z = D.z, NZ = Math.pow(2, Z);

/* road length per tile, in hundredths of a km, little-endian uint16 */
function unpack16(b64, n){
  if(!b64) return null;
  const raw = atob(b64), out = new Uint16Array(n);
  for(let i=0;i<n;i++) out[i] = raw.charCodeAt(2*i) | (raw.charCodeAt(2*i+1) << 8);
  return out;
}
const KMQ = unpack16(D.km, NT);
const KM_SCALE = D.kmScale || 0.01;
const kmOf = i => KMQ ? KMQ[i] * KM_SCALE : 0;

/* Sequential encoding: one hue, light to dark. The lightest step means
   "near zero", so the ramp is reversed on a dark surface, where it is the
   darkest step that recedes into the background. */
const BLUE = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"];
const ZERO_COLOUR = "#d03b3b";          /* a state, not a magnitude */
/* the tiles you hold: a hue of its own, away from the green of the roads, the
   red of the gaps and the muted blues of the density ramp */
const HELD = [74,163,255];              /* #4aa3ff */

const inWin = (x,y)=> x>=D.x0 && y>=D.y0 && x<D.x0+W && y<D.y0+H;
const idx   = (x,y)=> (y-D.y0)*W + (x-D.x0);
function status(x,y){                       // 2 green, 1 red, 0 out of scope
  if(!inWin(x,y)) return 0;
  const i = idx(x,y);
  if(ROAD[i]) return 2;
  if(LAND && LAND[i]) return 1;
  return 0;
}

/* ---------- projection -------------------------------------------------- */
const MAXLAT = 85.0511287798066;
function ll2t(lon,lat){
  const la = Math.max(-MAXLAT, Math.min(MAXLAT, lat));
  const s = Math.sin(la*Math.PI/180);
  return [ (lon+180)/360*NZ,
           (0.5 - Math.log((1+s)/(1-s))/(4*Math.PI))*NZ ];
}
function t2ll(x,y){
  return [ x/NZ*360-180,
           Math.atan(Math.sinh(Math.PI*(1-2*y/NZ)))*180/Math.PI ];
}

/* ---------- offscreen tile raster (1 px = 1 explorer tile) -------------- */
const grid = document.createElement("canvas");
grid.width = W; grid.height = H;
const gctx = grid.getContext("2d");
let showRed = true;
let showGreen = true;
let mode = "coverage";
let BINS = null;            /* six thresholds, in km */
function computeBins(){
  if(!KMQ) return;
  const v = [];
  for(let i=0;i<NT;i++) if(ROAD[i] && KMQ[i] > 0) v.push(KMQ[i]);
  if(!v.length){ BINS = null; return; }
  v.sort((a,b)=>a-b);
  BINS = [];
  for(let k=1;k<=6;k++) BINS.push(v[Math.floor(v.length * k / 7)] * KM_SCALE);
}
function binOf(km){
  if(!BINS) return 0;
  let b = 0;
  while(b < BINS.length && km > BINS[b]) b++;
  return b;                 /* 0..6 */
}
function hex2rgb(h){
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
const TA = Math.round(255 * (D.tileAlpha === undefined ? 0.80 : D.tileAlpha));
const TA_RED = Math.round(TA * 0.83);
const BM = D.basemap || null;
// With a raster basemap underneath, the vector country shapes must not be
// filled or they would hide it; and on a light basemap the labels have to
// flip to dark ink on a light halo.
const LIGHTBG = !!(BM && BM.light && !BM.grey);
const INK  = LIGHTBG ? "#16202e" : "#cfd8e3";
const INK2 = LIGHTBG ? "#0b1018" : "#e6edf3";
const HALO = LIGHTBG ? "rgba(255,255,255,0.9)" : "rgba(10,14,20,0.85)";
function paintGrid(){
  const img = gctx.createImageData(W,H), p = img.data;
  // A tile you already hold wins over everything else, in both views: that is
  // what makes "show me only my tiles" a matter of switching the other two
  // layers off, and what makes a red tile you have ridden read as held.
  const vis = showVisited && VISITED;
  if(mode === "density" && KMQ){
    if(!BINS) computeBins();
    const ramp = LIGHTBG ? BLUE : BLUE.slice().reverse();
    const rgb = ramp.map(hex2rgb);
    const zero = hex2rgb(ZERO_COLOUR);
    for(let i=0;i<NT;i++){
      const o = i<<2;
      if(vis && VISITED[i]){
        p[o]=HELD[0]; p[o+1]=HELD[1]; p[o+2]=HELD[2]; p[o+3]=TA;
      } else if(ROAD[i] && KMQ[i] > 0){
        if(!showGreen) continue;
        const c = rgb[binOf(KMQ[i] * KM_SCALE)];
        p[o]=c[0]; p[o+1]=c[1]; p[o+2]=c[2]; p[o+3]=TA;
      } else if(LAND && LAND[i]){
        if(!showRed) continue;
        p[o]=zero[0]; p[o+1]=zero[1]; p[o+2]=zero[2]; p[o+3]=TA_RED;
      }
    }
  } else {
    for(let i=0;i<NT;i++){
      const o = i<<2;
      if(vis && VISITED[i]){
        p[o]=HELD[0]; p[o+1]=HELD[1]; p[o+2]=HELD[2]; p[o+3]=TA;
      }
      else if(ROAD[i]){
        if(!showGreen) continue;
        p[o]=46; p[o+1]=160; p[o+2]=67; p[o+3]=TA;
      }
      else if(showRed && LAND && LAND[i]){ p[o]=218; p[o+1]=54; p[o+2]=51; p[o+3]=TA_RED; }
    }
  }
  gctx.putImageData(img,0,0);
}
paintGrid();

/* ---------- view -------------------------------------------------------- */
const cv = document.getElementById("map"), ctx = cv.getContext("2d");
let DPR = 1, VW = 0, VH = 0;
const view = { cx:0, cy:0, scale:1 };   // scale = css px per tile unit

function resize(){
  if(document.getElementById("legend").style.display === "block") placeLegend();
  DPR = Math.min(window.devicePixelRatio||1, 2);
  VW = cv.clientWidth; VH = cv.clientHeight;
  cv.width = Math.round(VW*DPR); cv.height = Math.round(VH*DPR);
  draw();
}
function fitAll(){
  const m = 0.06;
  view.scale = Math.min(VW*(1-m)/W, VH*(1-m)/H);
  view.cx = D.x0 + W/2; view.cy = D.y0 + H/2;
}
const sx = tx => (tx - view.cx)*view.scale + VW/2;
const sy = ty => (ty - view.cy)*view.scale + VH/2;
const ix = px => (px - VW/2)/view.scale + view.cx;
const iy = py => (py - VH/2)/view.scale + view.cy;

/* ---------- raster basemap (local files only; blocked in sandboxes) ----- */
const timg = new Map();
let rasterSkipped = false;
function basemapTile(z,x,y){
  const k = z+"/"+x+"/"+y;
  let e = timg.get(k);
  if(e === undefined){
    const im = new Image();
    im.onload = ()=>{ e.ok = true; requestDraw(); };
    im.onerror = ()=>{ e.ok = false; };
    e = {im:im, ok:false};
    timg.set(k,e);
    const sub = BM.sub ? BM.sub[Math.abs(x + y) % BM.sub.length] : "";
    const r2x = (window.devicePixelRatio || 1) > 1.3 ? "@2x" : "";
    im.src = BM.url.replace("{s}", sub).replace("{z}", z)
                   .replace("{x}", x).replace("{y}", y).replace("{r}", r2x);
    if(timg.size > 600){ for(const kk of timg.keys()){ timg.delete(kk); if(timg.size<400) break; } }
  }
  return e;
}
// The raster basemap is confined to the mapped country: everything outside
// keeps the plain dark background. The regions outline is used when present
// because it is far sharper than the 1:10m country shape.
let _clip = null;
function countryClip(){
  if(_clip) return _clip;
  _clip = (REGIONS && REGIONS.rings.length)
            ? REGIONS.rings
            : RINGS.filter(r => r.h).map(r => r.pts);
  return _clip;
}

function drawRaster(alpha, blend){
  // pick the slippy zoom whose 256-px tiles best match the current scale:
  // one z14 tile spans view.scale css px, so a z-tile spans scale*2^(Z-z) px,
  // and we want that close to 256.
  let z = Math.round(Math.log2(view.scale) + Z - 8);
  z = Math.max(2, Math.min(16, z));
  const f = Math.pow(2, z - Z);                 // basemap tiles per z14 tile
  const bx0 = Math.floor(ix(0)*f), bx1 = Math.ceil(ix(VW)*f);
  const by0 = Math.floor(iy(0)*f), by1 = Math.ceil(iy(VH)*f);
  const n = 1<<z;
  // OSM's tile policy forbids bulk downloading. If the current view would need
  // more than a modest number of tiles, draw none: the vector outlines carry
  // the map until the user zooms into something reasonable.
  const need = (bx1-bx0+1) * (by1-by0+1);
  if(need > (BM.budget || 120)){ rasterSkipped = true; return; }
  rasterSkipped = false;
  ctx.save();
  if(BM.clip){
    const rings = countryClip();
    if(rings.length){
      ctx.beginPath();
      for(const pts of rings){
        ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
        for(let i=1;i<pts.length;i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
        ctx.closePath();
      }
      ctx.clip();
    }
  }
  ctx.globalAlpha = alpha;
  if(blend) ctx.globalCompositeOperation = blend;
  if(BM.grey && "filter" in ctx) ctx.filter = "grayscale(1)";
  for(let bx=bx0; bx<=bx1; bx++){
    for(let by=by0; by<=by1; by++){
      if(by<0||by>=n) continue;
      const wx = ((bx % n)+n)%n;
      const e = basemapTile(z,wx,by);
      if(!e.ok) continue;
      // Snap each tile to whole pixels and derive its size from the next
      // boundary, so neighbours abut exactly. Drawing them 1px oversized
      // instead would composite the shared strip twice and leave a bright
      // seam -- visible as a spurious half-tile grid at some zooms.
      const X0 = Math.round(sx(bx/f)),     Y0 = Math.round(sy(by/f));
      const X1 = Math.round(sx((bx+1)/f)), Y1 = Math.round(sy((by+1)/f));
      ctx.drawImage(e.im, X0, Y0, X1 - X0, Y1 - Y0);
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  if("filter" in ctx) ctx.filter = "none";
  ctx.restore();
}

/* ---------- vector basemap --------------------------------------------- */
const RINGS = (D.base && D.base.rings) ? D.base.rings.map(r=>{
  const pts = []; let lx=0, ly=0;
  for(let i=0;i<r.d.length;i+=2){
    lx += r.d[i]; ly += r.d[i+1];
    pts.push(ll2t(lx/1e4, ly/1e4));
  }
  return {h:r.h, pts:pts};
}) : [];

/* ---------- administrative outlines ------------------------------------ */
function decodeAdmin(layer){
  if(!layer) return null;
  const rings = layer.rings.map(d=>{
    const pts = []; let lx=0, ly=0;
    for(let i=0;i<d.length;i+=2){
      lx += d[i]; ly += d[i+1];
      pts.push(ll2t(lx/1e4, ly/1e4));
    }
    return pts;
  });
  const labels = (layer.labels||[]).map(l=>({name:l[0], code:l[1], t:ll2t(l[2], l[3])}));
  return {rings:rings, labels:labels};
}
const REGIONS = decodeAdmin(D.admin && D.admin.regions);
const DEPTS   = decodeAdmin(D.admin && D.admin.departments);

function drawAdmin(layer, col, lw){
  if(!layer) return;
  ctx.strokeStyle = col;
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for(const p of layer.rings){
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.closePath();
  }
  ctx.stroke();
}

/* Admin labels are laid out before the cities are drawn, so a city can drop
   its own text where an admin name already occupies the spot (Paris the city
   vs Paris the departement). */
let activeLabels = [];
function layoutAdminLabels(){
  activeLabels = [];
  const useDep = showDep && DEPTS && view.scale >= 2.2;
  const useReg = showReg && REGIONS && view.scale < 2.2;
  const layer = useDep ? DEPTS : (useReg ? REGIONS : null);
  if(!layer) return;
  const size = useDep ? 12 : 13;
  const col  = LIGHTBG ? (useDep ? "#223042" : "#0b1018")
                       : (useDep ? "#dbe7f5" : "#ffffff");
  for(const l of layer.labels){
    const X = sx(l.t[0]), Y = sy(l.t[1]);
    if(X < -60 || X > VW+60 || Y < -20 || Y > VH+20) continue;
    activeLabels.push({name:l.name, X:X, Y:Y, size:size, col:col});
  }
}
function paintAdminLabels(){
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for(const l of activeLabels){
    ctx.font = "600 " + l.size + "px ui-sans-serif,system-ui,sans-serif";
    ctx.strokeStyle = HALO;
    ctx.lineWidth = 3.5;
    ctx.strokeText(l.name, l.X, l.Y);
    ctx.fillStyle = l.col;
    ctx.fillText(l.name, l.X, l.Y);
  }
  ctx.textAlign = "left";
}
function labelOccupied(X, Y){
  for(const l of activeLabels){
    if(Math.abs(l.X - X) < 90 && Math.abs(l.Y - Y) < 16) return true;
  }
  return false;
}

function drawVector(){
  for(const r of RINGS){
    ctx.beginPath();
    const p = r.pts;
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.closePath();
    if(!BM || BM.clip || !r.h){    // keep the dark land everywhere the raster
      ctx.fillStyle = r.h ? "#1c2531" : "#141a22";   // will not cover
      ctx.fill();
    }
    ctx.strokeStyle = (LIGHTBG && !BM.clip) ? (r.h ? "#6b7a8d" : "#aab4c0")
                                            : (r.h ? "#4a5a70" : "#262f3c");
    ctx.lineWidth = r.h ? 1.2 : 0.8;
    ctx.stroke();
  }
}

/* ---------- the route --------------------------------------------------- */
const ROUTE = (function(){
  const r = D.route;
  if(!r || !r.d || !r.d.length) return null;
  const pts = []; let lx = 0, ly = 0;
  for(let i=0;i<r.d.length;i+=2){
    lx += r.d[i]; ly += r.d[i+1];
    pts.push(ll2t(lx/1e5, ly/1e5));
  }
  // cumulative length along the track, so the profile can be pointed at a
  // place on the map
  const cum = [0];
  for(let i=1;i<pts.length;i++)
    cum.push(cum[i-1] + Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]));
  return {pts:pts, cum:cum, total:cum[cum.length-1],
          stats:r.stats||{}, start:r.start, end:r.end, rect:r.rect,
          profile:r.profile||null, pinned:r.pinned||null,
          skipped:r.skipped||null, wanted:r.wanted||null};
})();

/* ---------- elevation profile ------------------------------------------ */
let profMark = null;          /* 0..1 along the ride, or null */

function routePointAt(frac){
  if(!ROUTE || !ROUTE.total) return null;
  const want = frac * ROUTE.total, c = ROUTE.cum;
  let lo = 0, hi = c.length - 1;
  while(lo < hi - 1){ const mid = (lo + hi) >> 1; (c[mid] <= want ? lo = mid : hi = mid); }
  const span = c[hi] - c[lo] || 1, t = (want - c[lo]) / span;
  return [ROUTE.pts[lo][0] + (ROUTE.pts[hi][0]-ROUTE.pts[lo][0]) * t,
          ROUTE.pts[lo][1] + (ROUTE.pts[hi][1]-ROUTE.pts[lo][1]) * t];
}

function drawProfile(){
  const pr = ROUTE && ROUTE.profile;
  const box = document.getElementById("profile");
  if(!pr || !pr.elev || pr.elev.length < 2){ box.style.display = "none"; return; }
  box.style.display = "block";
  const cv2 = document.getElementById("prof");
  const w = cv2.clientWidth, h = cv2.clientHeight;
  cv2.width = Math.round(w * DPR); cv2.height = Math.round(h * DPR);
  const c = cv2.getContext("2d");
  c.setTransform(DPR,0,0,DPR,0,0);
  c.clearRect(0,0,w,h);

  const e = pr.elev, n = e.length;
  const lo = pr.min, hi = pr.max, span = Math.max(1, hi - lo);
  const pad = 6;
  const X = i => i / (n - 1) * w;
  const Y = v => h - pad - (v - lo) / span * (h - 2*pad);

  // recessive gridlines at round altitudes
  const stepM = span > 800 ? 200 : span > 300 ? 100 : span > 120 ? 50 : 20;
  c.strokeStyle = "rgba(255,255,255,0.10)"; c.lineWidth = 1;
  c.fillStyle = "rgba(180,190,205,0.75)";
  c.font = "10px ui-sans-serif,system-ui,sans-serif";
  c.textBaseline = "middle";
  for(let v = Math.ceil(lo/stepM)*stepM; v <= hi; v += stepM){
    const y = Math.round(Y(v)) + 0.5;
    c.beginPath(); c.moveTo(0,y); c.lineTo(w,y); c.stroke();
    c.fillText(v + " m", 2, y - 6);
  }

  c.beginPath();
  c.moveTo(X(0), h);
  for(let i=0;i<n;i++) c.lineTo(X(i), Y(e[i]));
  c.lineTo(X(n-1), h); c.closePath();
  const grad = c.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(57,135,229,0.55)");
  grad.addColorStop(1, "rgba(57,135,229,0.06)");
  c.fillStyle = grad; c.fill();

  c.beginPath();
  for(let i=0;i<n;i++) (i ? c.lineTo(X(i), Y(e[i])) : c.moveTo(X(i), Y(e[i])));
  c.strokeStyle = "#86b6ef"; c.lineWidth = 2; c.stroke();

  if(profMark !== null){
    const i = Math.max(0, Math.min(n-1, Math.round(profMark * (n-1))));
    const x = X(i), y = Y(e[i]);
    c.strokeStyle = "rgba(255,210,74,0.9)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke();
    c.fillStyle = "#ffd24a";
    c.beginPath(); c.arc(x, y, 3.5, 0, 6.2832); c.fill();
    document.getElementById("pf-at").textContent =
      (profMark * pr.km).toFixed(1) + " km  \u00b7  " + e[i] + " m";
  } else {
    document.getElementById("pf-at").textContent = "";
  }
}

function drawRouteBrief(){
  // the rectangle the ride was asked to cover, plus the pinned start and
  // finish tiles -- dashed, so it reads as the brief rather than as data
  if(!ROUTE) return;
  ctx.save();
  ctx.setLineDash([7, 5]);
  if(ROUTE.rect && ROUTE.rect.length === 4){
    const [x0,y0,x1,y1] = ROUTE.rect;
    outline(x0, y0, x1-x0+1, y1-y0+1, "rgba(255,255,255,0.85)", 2);
  }
  ctx.setLineDash([]);
  if(ROUTE.start) outline(ROUTE.start[0], ROUTE.start[1], 1, 1, "#2ea043", 2.5);
  if(ROUTE.end)   outline(ROUTE.end[0], ROUTE.end[1], 1, 1, "#da3633", 2.5);
  ctx.restore();
}

function drawRoute(){
  if(!ROUTE) return;
  const p = ROUTE.pts;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for(const pass of [{c:"rgba(8,12,18,0.85)", w:6}, {c:"#ffd24a", w:2.6}]){
    ctx.strokeStyle = pass.c;
    ctx.lineWidth = pass.w;
    ctx.beginPath();
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.stroke();
  }
  // the tiles this ride set out to collect
  for(const t of (ROUTE.wanted || []))
    outline(t[0], t[1], 1, 1, "#ffd24a", 2);

  // the tiles that had to be given up: shown where they are, so the gap in
  // the square is something you can see rather than a number in a warning
  for(const t of (ROUTE.skipped || [])){
    const X = sx(t[0]), Y = sy(t[1]), W = view.scale;
    ctx.save();
    ctx.beginPath(); ctx.rect(X, Y, W, W); ctx.clip();
    ctx.strokeStyle = "rgba(218,54,51,0.85)"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for(let d = -W; d < W * 2; d += 7){
      ctx.moveTo(X + d, Y); ctx.lineTo(X + d + W, Y + W);
    }
    ctx.stroke();
    ctx.restore();
    outline(t[0], t[1], 1, 1, "#da3633", 1.5);
  }

  // where you asked to start, against where the network actually let you:
  // a hollow ring on the point you clicked, a hair line to the junction
  const pin = ROUTE.pinned;
  if(pin){
    const marks = [[pin.start, "#2ea043"], [pin.end, "#da3633"]];
    for(const v of (pin.via || [])) marks.push([v, "#ffd24a"]);
    for(const [q, col] of marks){
      if(!q) continue;
      const a = ll2t(q.asked[0], q.asked[1]);
      const b = ll2t(q.node[0], q.node[1]);
      ctx.beginPath();
      ctx.moveTo(sx(a[0]), sy(a[1])); ctx.lineTo(sx(b[0]), sy(b[1]));
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(sx(a[0]), sy(a[1]), 4.5, 0, 6.2832);
      ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke();
    }
    // the waypoints keep the planner's symbol, and stay unnumbered: the ride
    // order is the solver's, not the order you clicked them in
    for(const v of (pin.via || [])){
      const b = ll2t(v.node[0], v.node[1]);
      diamond(b[0], b[1], "#ffd24a");
    }
  }
  const ends = [[p[0], "#2ea043", "S"], [p[p.length-1], "#da3633", "F"]];
  ctx.font = "700 10px ui-sans-serif,system-ui,sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for(const [q, col, lab] of ends){
    const X = sx(q[0]), Y = sy(q[1]);
    ctx.beginPath(); ctx.arc(X, Y, 7, 0, 6.2832);
    ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText(lab, X, Y);
  }
  ctx.textAlign = "left";
}

/* ---------- state ------------------------------------------------------- */
let showTiles=true, showGrid=true, showBase=true, showCities=true;
let showRoute=true;
/* planning: 0 off, 1 awaiting the rectangle, 2 the start tile,
   3 the end tile, 4 done */
let planMode = 0, planRect = null, planStart = null, planEnd = null,
    planVia = [], band = null;
let planKind = "square";          // or "seeker"
let planWant = new Map();         // "x,y" -> [x,y], the tiles to collect
const lastStep = () => 5;      // both planners end on the name
let showReg=true, showDep=true;
let hover=null, pick=[], best=null;
let raf = 0;
// The first resize() happens before fitAll(), when the view still sits at
// scale 1 over tile 0,0. Drawing then would fire a burst of basemap requests
// for a meaningless part of the world, so nothing paints until we are framed.
let ready = false;
function requestDraw(){ if(!raf) raf = requestAnimationFrame(()=>{ raf=0; draw(); }); }

function draw(){
  if(!ready) return;
  ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,VW,VH);
  ctx.fillStyle = (BM && BM.light && !BM.grey && !BM.clip) ? "#eceff3" : "#0e1116";
  ctx.fillRect(0,0,VW,VH);

  if(showBase){
    if(RINGS.length) drawVector();   // dark land shapes, as without a basemap
    if(BM) drawRaster(BM.opacity === undefined ? 1 : BM.opacity, null);
  }

  if(showTiles){
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(grid, sx(D.x0), sy(D.y0), W*view.scale, H*view.scale);
    ctx.imageSmoothingEnabled = true;
  }

  // tile grid lines, once tiles are big enough to be legible
  if(showGrid && view.scale > 7){
    const a = Math.min(0.5, (view.scale-7)/60);
    ctx.strokeStyle = (LIGHTBG ? "rgba(40,50,65," : "rgba(255,255,255,")
                      + a.toFixed(3) + ")";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const X0 = Math.floor(ix(0)), X1 = Math.ceil(ix(VW));
    const Y0 = Math.floor(iy(0)), Y1 = Math.ceil(iy(VH));
    for(let x=X0;x<=X1;x++){ const s=Math.round(sx(x))+0.5; ctx.moveTo(s,0); ctx.lineTo(s,VH); }
    for(let y=Y0;y<=Y1;y++){ const s=Math.round(sy(y))+0.5; ctx.moveTo(0,s); ctx.lineTo(VW,s); }
    ctx.stroke();
  }

  // Re-assert the basemap over the tint. Multiply leaves pale ground almost
  // untouched while the dark road lines punch back through, so the streets
  // stay readable instead of drowning under a flat wash of colour.
  if(showBase && showTiles && BM && BM.detail > 0 && !rasterSkipped){
    drawRaster(BM.detail, "multiply");
  }

  if(showDep && DEPTS) drawAdmin(DEPTS,
      LIGHTBG ? "rgba(20,30,45,0.30)" : "rgba(226,238,255,0.30)", 1);
  if(showReg && REGIONS) drawAdmin(REGIONS,
      LIGHTBG ? "rgba(10,16,26,0.70)" : "rgba(255,255,255,0.72)", 2);
  layoutAdminLabels();

  if(showCities && D.cities){
    ctx.font = "600 11px ui-sans-serif,system-ui,sans-serif";
    ctx.textBaseline = "middle";
    // tier 2 (the smaller prefectures) only appears once there is room
    const minTier = view.scale < 1.8 ? 1 : 2;
    const placed = [];
    for(const c of D.cities){
      const name = c[0], tier = c[3] || 1;
      if(tier > minTier) continue;
      const t = ll2t(c[1], c[2]), X = sx(t[0]), Y = sy(t[1]);
      if(X<-80||X>VW+80||Y<-20||Y>VH+20) continue;
      ctx.fillStyle = INK2;
      ctx.beginPath(); ctx.arc(X,Y,2.6,0,6.2832); ctx.fill();
      if(labelOccupied(X+6, Y)) continue;
      // and do not stack two city names on top of each other
      const wdt = ctx.measureText(name).width;
      let clash = false;
      for(const q of placed){
        if(X+6 < q.x1 + 6 && q.x0 - 6 < X + 6 + wdt
           && Math.abs(q.y - Y) < 13){ clash = true; break; }
      }
      if(clash) continue;
      placed.push({x0:X+6, x1:X+6+wdt, y:Y});
      ctx.strokeStyle = HALO; ctx.lineWidth = 3;
      ctx.strokeText(name, X+6, Y);
      ctx.fillStyle = INK;
      ctx.fillText(name, X+6, Y);
    }
  }

  paintAdminLabels();

  if(band){
    const r = bandRect();
    outline(r.x0, r.y0, r.x1-r.x0+1, r.y1-r.y0+1, "#ffd24a", 2);
    const d = rectDims(r);
    const txt = d.w + " \u00d7 " + d.h + "  \u00b7  " + d.n + " tiles  \u00b7  "
              + d.kx.toFixed(1) + " \u00d7 " + d.ky.toFixed(1) + " km";
    ctx.font = "600 12px ui-sans-serif,system-ui,sans-serif";
    ctx.textBaseline = "bottom";
    const X = sx(r.x0), Y = sy(r.y0) - 7;
    const wdt = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(10,14,20,0.82)";
    ctx.fillRect(X - 5, Y - 16, wdt + 10, 20);
    ctx.fillStyle = "#ffd24a";
    ctx.fillText(txt, X, Y);
    ctx.textBaseline = "middle";
  } else if(planRect){
    outline(planRect.x0, planRect.y0, planRect.x1-planRect.x0+1,
            planRect.y1-planRect.y0+1, "#ffd24a", 2);
  }
  if(flashTile){
    const t = (flashUntil - performance.now()) / 4000;
    const pulse = 0.45 + 0.55 * Math.abs(Math.sin(performance.now() / 180));
    outline(flashTile[0], flashTile[1], 1, 1,
            "rgba(255,210,74," + pulse.toFixed(2) + ")", 3);
    const r = (1 + 14 * (1 - (t * 4 % 1))) * view.scale;
    ctx.beginPath();
    ctx.arc(sx(flashTile[0] + 0.5), sy(flashTile[1] + 0.5), Math.max(r, 2),
            0, 6.2832);
    ctx.strokeStyle = "rgba(255,210,74," + (0.5 * pulse).toFixed(2) + ")";
    ctx.lineWidth = 2; ctx.stroke();
  }
  // the tiles picked for a tile hunt
  if(planKind === "seeker"){
    for(const [x, y] of planWant.values()){
      ctx.fillStyle = "rgba(255,210,74,0.30)";
      ctx.fillRect(sx(x), sy(y), view.scale, view.scale);
      outline(x, y, 1, 1, "#ffd24a", 2);
    }
  }

  // no numbers on the waypoints: the order you clicked them in is not the
  // order they will be ridden in -- the solver decides that
  for(const v of planVia) diamond(v.x, v.y, "#ffd24a");
  if(planStart) marker(planStart, "#2ea043", "S");
  if(planEnd)   marker(planEnd, "#da3633", "F");

  if(showRoute && ROUTE){
    drawRouteBrief(); drawRoute();
    if(profMark !== null){
      const q = routePointAt(profMark);
      if(q){
        const X = sx(q[0]), Y = sy(q[1]);
        ctx.beginPath(); ctx.arc(X, Y, 8, 0, 6.2832);
        ctx.fillStyle = "rgba(255,210,74,0.25)"; ctx.fill();
        ctx.beginPath(); ctx.arc(X, Y, 4.5, 0, 6.2832);
        ctx.fillStyle = "#ffd24a"; ctx.fill();
        ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }

  if(best) outline(best.x, best.y, best.n, best.n, "#9085e9", 2.5);
  if(pick.length === 2){
    const x0=Math.min(pick[0][0],pick[1][0]), x1=Math.max(pick[0][0],pick[1][0]);
    const y0=Math.min(pick[0][1],pick[1][1]), y1=Math.max(pick[0][1],pick[1][1]);
    outline(x0,y0,x1-x0+1,y1-y0+1,"#539bf5",2);
  } else if(pick.length === 1){
    outline(pick[0][0],pick[0][1],1,1,"#539bf5",2);
  }
  if(hover) outline(hover[0],hover[1],1,1,"#ffffff",1.5);

  drawScale();
}
function diamond(tx, ty, col){
  const X = sx(tx), Y = sy(ty), r = 6;
  ctx.beginPath();
  ctx.moveTo(X, Y-r); ctx.lineTo(X+r, Y); ctx.lineTo(X, Y+r);
  ctx.lineTo(X-r, Y); ctx.closePath();
  ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 1.5; ctx.stroke();
}

function marker(pt, col, letter){
  // the point you clicked, with its tile outlined behind it: the ride will
  // begin at the nearest junction, which the solver reports once it has the
  // road database in hand
  outline(pt.tx, pt.ty, 1, 1, col, 1.5);
  const X = sx(pt.x), Y = sy(pt.y);
  ctx.beginPath(); ctx.moveTo(X-9, Y); ctx.lineTo(X+9, Y);
  ctx.moveTo(X, Y-9); ctx.lineTo(X, Y+9);
  ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(X, Y, 5, 0, 6.2832);
  ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.font = "600 10px ui-sans-serif,system-ui,sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#fff"; ctx.fillText(letter, X, Y + 0.5);
  ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
}

function outline(x,y,w,h,col,lw){
  ctx.strokeStyle = col; ctx.lineWidth = lw;
  ctx.strokeRect(sx(x), sy(y), w*view.scale, h*view.scale);
}

function drawScale(){
  const mid = t2ll(view.cx, view.cy)[1];
  const mPerTile = 40075016.686 * Math.cos(mid*Math.PI/180) / NZ;
  const mPerPx = mPerTile / view.scale;
  let target = 90*mPerPx, p = Math.pow(10, Math.floor(Math.log10(target)));
  let n = p; for(const m of [1,2,5,10]){ if(p*m <= target) n = p*m; }
  document.getElementById("sc-ln").style.width = (n/mPerPx).toFixed(0)+"px";
  document.getElementById("sc-lab").textContent =
    n>=1000 ? (n/1000)+" km" : n+" m";
}

/* ---------- interaction ------------------------------------------------- */
let drag=null;
cv.addEventListener("pointerdown", e=>{
  cv.setPointerCapture(e.pointerId);
  if(((planKind === "square" && planMode === 1)
      || (planKind === "seeker" && planMode === 3))
     && (e.ctrlKey || e.metaKey)){
    const r = cv.getBoundingClientRect();
    const t = [Math.floor(ix(e.clientX-r.left)), Math.floor(iy(e.clientY-r.top))];
    band = {a:t, b:t};
    requestDraw();
    return;                       // a planning drag must not pan the map
  }
  drag = {px:e.clientX, py:e.clientY, cx:view.cx, cy:view.cy, moved:0};
  cv.classList.add("drag");
});
cv.addEventListener("pointermove", e=>{
  const r = cv.getBoundingClientRect();
  if(band){
    band.b = [Math.floor(ix(e.clientX-r.left)), Math.floor(iy(e.clientY-r.top))];
    const tx = band.b[0], ty = band.b[1];
    hover = [tx,ty]; readout(tx,ty); renderPlan(); requestDraw();
    return;
  }
  if(drag){
    const dx = e.clientX-drag.px, dy = e.clientY-drag.py;
    drag.moved = Math.max(drag.moved, Math.abs(dx)+Math.abs(dy));
    view.cx = drag.cx - dx/view.scale;
    view.cy = drag.cy - dy/view.scale;
  }
  const tx = Math.floor(ix(e.clientX-r.left)), ty = Math.floor(iy(e.clientY-r.top));
  hover = [tx,ty];
  readout(tx,ty);
  requestDraw();
});
cv.addEventListener("pointerup", e=>{
  cv.classList.remove("drag");
  if(band){
    const b = bandRect();
    band = null;
    if(planKind === "seeker"){
      for(let x = b.x0; x <= b.x1; x++)
        for(let y = b.y0; y <= b.y1; y++)
          planWant.set(x + "," + y, [x, y]);
    } else {
      planRect = b;
      planMode = 2; planStart = planEnd = null; planVia = [];
    }
    renderPlan(); requestDraw();
    return;
  }
  const wasClick = drag && drag.moved < 4;
  drag = null;
  if(!wasClick) return;
  const r = cv.getBoundingClientRect();
  const fx = ix(e.clientX-r.left), fy = iy(e.clientY-r.top);
  const tx = Math.floor(fx), ty = Math.floor(fy);
  const seeking = planKind === "seeker";
  if(planMode >= (seeking ? 1 : 2) && planMode <= 4){
    // an exact point, not a tile: it may sit outside the area, which is what
    // you want when you park or take a train to the edge of it
    const ll = t2ll(fx, fy);
    const pt = {x:fx, y:fy, lon:ll[0], lat:ll[1], tx:tx, ty:ty};
    const firstPoint = seeking ? 1 : 2;
    if(planMode === firstPoint){ planStart = pt; planMode = firstPoint + 1; }
    else if(planMode === firstPoint + 1){ planEnd = pt; planMode = firstPoint + 2; }
    else if(seeking && planMode === 3){
      // a click toggles: picking a tile twice unpicks it
      const key = tx + "," + ty;
      if(planWant.has(key)) planWant.delete(key);
      else planWant.set(key, [tx, ty]);
    }
    else { planVia.push(pt); }      // stay on 4: add as many as you like
    renderPlan(); requestDraw();
    return;
  }
  if(e.shiftKey){
    if(pick.length >= 2) pick = [];
    pick.push([tx,ty]);
    if(pick.length === 2) showZone();
  } else {
    pick = []; document.getElementById("zone").style.display="none";
  }
  requestDraw();
});
cv.addEventListener("contextmenu", e=>{
  if(planMode < 4 || !planVia.length) return;
  const r = cv.getBoundingClientRect();
  const fx = ix(e.clientX-r.left), fy = iy(e.clientY-r.top);
  let best = -1, bd = 14 / view.scale;      // within 14 screen pixels
  planVia.forEach((v, i) => {
    const d = Math.hypot(v.x - fx, v.y - fy);
    if(d < bd){ bd = d; best = i; }
  });
  if(best >= 0){
    e.preventDefault();
    planVia.splice(best, 1);
    renderPlan(); requestDraw();
  }
});
cv.addEventListener("pointerleave", ()=>{ hover=null; requestDraw(); });
cv.addEventListener("wheel", e=>{
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  const tx = ix(mx), ty = iy(my);
  const k = Math.exp(-e.deltaY * (e.deltaMode===1 ? 0.05 : 0.0016));
  view.scale = Math.max(0.05, Math.min(400, view.scale*k));
  view.cx = tx - (mx - VW/2)/view.scale;
  view.cy = ty - (my - VH/2)/view.scale;
  requestDraw();
}, {passive:false});

function readout(x,y){
  const st = status(x,y);
  document.getElementById("r-tile").textContent = x+", "+y;
  const el = document.getElementById("r-stat");
  el.innerHTML = st===2 ? '<span class="sw" style="background:var(--green)"></span>paved road'
               : st===1 ? '<span class="sw" style="background:var(--red)"></span>no paved road'
               : '<span class="sw" style="background:#555"></span>out of scope';
  const a = t2ll(x, y+1), b = t2ll(x+1, y);
  document.getElementById("r-ll").textContent =
    ((a[1]+b[1])/2).toFixed(5)+", "+((a[0]+b[0])/2).toFixed(5);
  const latm = (b[1]-a[1])*110574, lonm = (b[0]-a[0])*111320*Math.cos((a[1]+b[1])/2*Math.PI/180);
  document.getElementById("r-sz").textContent =
    (lonm/1000).toFixed(2)+" x "+(latm/1000).toFixed(2)+" km";
  const kmEl = document.getElementById("r-km");
  if(!KMQ || !inWin(x,y)) kmEl.textContent = "\u2014";
  else {
    const km = kmOf(idx(x,y));
    kmEl.textContent = km > 0 ? km.toFixed(2)+" km" : (st===1 ? "none" : "\u2014");
  }
}

function zoneStats(x0,y0,x1,y1){
  let g=0,r=0,o=0;
  for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++){
    const s=status(x,y); if(s===2) g++; else if(s===1) r++; else o++;
  }
  return {g:g,r:r,o:o};
}
function maxSquareIn(x0,y0,x1,y1){
  const w=x1-x0+1, h=y1-y0+1;
  const dp = new Int32Array(w*h);
  let bn=0,bx=0,by=0;
  for(let j=0;j<h;j++) for(let i=0;i<w;i++){
    const ok = status(x0+i, y0+j)===2;
    let v=0;
    if(ok){
      v = 1 + Math.min(i>0?dp[j*w+i-1]:0,
                       j>0?dp[(j-1)*w+i]:0,
                       (i>0&&j>0)?dp[(j-1)*w+i-1]:0);
    }
    dp[j*w+i]=v;
    if(v>bn){ bn=v; bx=x0+i-v+1; by=y0+j-v+1; }
  }
  return bn? {n:bn,x:bx,y:by} : null;
}
function showZone(){
  const x0=Math.min(pick[0][0],pick[1][0]), x1=Math.max(pick[0][0],pick[1][0]);
  const y0=Math.min(pick[0][1],pick[1][1]), y1=Math.max(pick[0][1],pick[1][1]);
  const w=x1-x0+1, h=y1-y0+1, s=zoneStats(x0,y0,x1,y1);
  const sq = maxSquareIn(x0,y0,x1,y1);
  document.getElementById("zone").style.display="block";
  document.getElementById("z-cor").textContent = x0+","+y0+"  →  "+x1+","+y1;
  document.getElementById("z-grid").textContent = w+" × "+h+" = "+(w*h);
  document.getElementById("z-green").textContent = s.g+"  ("+(100*s.g/(w*h)).toFixed(1)+"%)";
  document.getElementById("z-red").textContent = s.r + (s.o? "  (+"+s.o+" out)":"");
  document.getElementById("z-sq").textContent = sq? (sq.n+" × "+sq.n+" @ "+sq.x+","+sq.y) : "none";
  document.getElementById("z-cmd").textContent =
    "tilehunt.py zone --corners "+x0+","+y0+" "+x1+","+y1;
  best = sq;
}

function fmtKm(v){
  return v >= 10 ? v.toFixed(0) : (v >= 1 ? v.toFixed(1) : v.toFixed(2));
}
function placeLegend(){
  // the left column stacks the panels itself now; nothing to position by hand
}
function renderLegend(){
  const box = document.getElementById("legend");
  if(mode !== "density" || !KMQ || !BINS){ box.style.display = "none"; return; }
  box.style.display = "block";
  const ramp = LIGHTBG ? BLUE : BLUE.slice().reverse();
  const rows = [];
  rows.push('<b><span class="sw" style="background:'+ZERO_COLOUR+'"></span></b>'
            + '<span>none</span>');
  for(let b=0;b<ramp.length;b++){
    const lo = b === 0 ? 0 : BINS[b-1];
    const hi = b < BINS.length ? BINS[b] : null;
    const label = hi === null ? "\u2265 " + fmtKm(lo) + " km"
                              : fmtKm(lo) + " \u2013 " + fmtKm(hi);
    rows.push('<b><span class="sw" style="background:'+ramp[b]+'"></span></b>'
              + '<span>' + label + '</span>');
  }
  document.getElementById("leg-rows").innerHTML = rows.join("");
  placeLegend();
}

function rectDims(r){
  const w = r.x1 - r.x0 + 1, h = r.y1 - r.y0 + 1;
  const midLat = t2ll(r.x0, (r.y0 + r.y1) / 2 + 0.5)[1];
  const kmPerTile = 40075.016686 * Math.cos(midLat * Math.PI / 180) / NZ;
  return {w:w, h:h, n:w*h, kx:w*kmPerTile, ky:h*kmPerTile};
}

function bandRect(){
  if(!band) return null;
  return {x0:Math.min(band.a[0],band.b[0]), x1:Math.max(band.a[0],band.b[0]),
          y0:Math.min(band.a[1],band.b[1]), y1:Math.max(band.a[1],band.b[1])};
}

function safeRunName(v){
  // the name becomes a folder, so keep it to what every filesystem accepts.
  // The same cleaning runs in the tool, so what you see here is what you get.
  return String(v || "").trim()
         .normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
         .replace(/[^A-Za-z0-9._-]+/g, "-")
         .replace(/^[-._]+|[-._]+$/g, "").slice(0, 64).replace(/[-._]+$/, "");
}

function renderPlan(){
  const box = document.getElementById("plan");
  if(!planMode){ box.style.display = "none"; cv.classList.remove("plan"); return; }
  box.style.display = "block";
  const seek = planKind === "seeker";
  cv.classList.toggle("plan", seek ? planMode === 3 : planMode === 1);
  document.getElementById("plan-steps-square").style.display =
    seek ? "none" : "block";
  document.getElementById("plan-steps-seek").style.display =
    seek ? "block" : "none";
  document.getElementById("plan-title").textContent =
    seek ? "TileSeeker" : "SquarePlanner";

  const steps = seek
    ? [[1,"qs1"],[2,"qs2"],[3,"qs3"],[4,"qs4"],[5,"qs5"]]
    : [[1,"ps1"],[2,"ps2"],[3,"ps3"],[4,"ps4"],[5,"ps5"]];
  for(const [i, id] of steps){
    const li = document.getElementById(id);
    li.className = planMode > i ? "done" : (planMode === i ? "now" : "");
  }
  document.getElementById("b-seek-done").style.display =
    seek && planMode === 3 ? "inline-block" : "none";
  document.getElementById("b-seek-via-done").style.display =
    seek && planMode === 4 ? "inline-block" : "none";

  const rows = [];
  const add = (k,v)=>rows.push("<b>"+k+"</b><span>"+v+"</span>");
  if(seek){
    const picked = bandRect();
    let n = planWant.size;
    if(picked)
      n = new Set([...planWant.keys(),
        ...(function*(){ for(let x=picked.x0;x<=picked.x1;x++)
                           for(let y=picked.y0;y<=picked.y1;y++)
                             yield x+","+y; })()]).size;
    add("wanted", n + " tile" + (n === 1 ? "" : "s"));
    if(picked)
      add("adding", rectDims(picked).w + " \u00d7 " + rectDims(picked).h);
  }
  const rect = seek ? null : (bandRect() || planRect);
  if(rect){
    const d = rectDims(rect);
    add("grid", d.w+" \u00d7 "+d.h+" = "+d.n+" tiles");
    add("size", d.kx.toFixed(1)+" \u00d7 "+d.ky.toFixed(1)+" km");
    const kmTile = d.kx / d.w;
    add("ideal floor", (d.n * kmTile * Math.SQRT2 / 3).toFixed(1) + " km");
    // counting every tile on each mouse move would stutter on a huge drag,
    // so the census waits for the release once the rectangle gets big
    if(!band || d.n <= 20000){
      const st = zoneStats(rect.x0, rect.y0, rect.x1, rect.y1);
      add("green", st.g);
      add("red", st.r + (st.o ? " (+"+st.o+" out)" : ""));
      if(st.r || st.o)
        add("note", "<span style='color:#f0b429'>red tiles cannot be ridden</span>");
    } else {
      add("green", "\u2026");
      add("red", "\u2026 on release");
    }
  }
  const ptRow = (lab, pt) => {
    if(!pt) return;
    add(lab, pt.lat.toFixed(5) + "," + pt.lon.toFixed(5));
    const out = rect && (pt.tx < rect.x0 || pt.tx > rect.x1
                      || pt.ty < rect.y0 || pt.ty > rect.y1);
    add("", "tile " + pt.tx + "," + pt.ty
         + (out ? " <span style='color:#f0b429'>outside the square</span>" : ""));
  };
  ptRow("start", planStart);
  ptRow("finish", planEnd);
  if(planStart || planEnd)
    add("note", "pinned to the nearest junction; the command reports how far");
  document.getElementById("plan-rows").innerHTML = rows.join("");

  const nameBox = document.getElementById("plan-name");
  const cmdBox = document.getElementById("plan-cmd");
  document.getElementById("b-via-done").style.display =
    planMode === 4 ? "inline-block" : "none";
  const viaBox = document.getElementById("via-list");
  if(planVia.length && planMode >= 4){
    viaBox.style.display = "grid";
    viaBox.innerHTML = planVia.map((v, i) =>
      "<b>via " + (i+1) + "</b><span>" + v.lat.toFixed(5) + ","
      + v.lon.toFixed(5)
      + " <a href='#' data-via='" + i + "' style='color:var(--red);"
      + "text-decoration:none'>&times;</a></span>").join("");
    for(const a of viaBox.querySelectorAll("a[data-via]"))
      a.onclick = ev => { ev.preventDefault();
        planVia.splice(+a.dataset.via, 1); renderPlan(); requestDraw(); };
  } else {
    viaBox.style.display = "none"; viaBox.innerHTML = "";
  }
  nameBox.style.display = planMode === lastStep() ? "block" : "none";
  if(planMode === lastStep()){
    const c = D.cmd || {};
    const field = document.getElementById("run-name");
    if(!field.value && !field.dataset.touched)
      field.value = seek
        ? "seek_" + (planWant.values().next().value || [0,0]).join("_")
        : "z" + planRect.x0 + "_" + planRect.y0;
    const name = safeRunName(field.value);
    field.classList.toggle("bad", !name);
    document.getElementById("run-path").textContent = name
      ? "Runs/" + (seek ? "TileSeeker" : "Square") + "/" + name + "/" + name
        + ".{route.json, gpx, html}"
      : "letters, digits, dash or underscore";
    if(!name){ cmdBox.style.display = "none"; return; }
    const py = c.py || "python";
    const dem = c.dem || "DEM_DB";
    const corners = planRect ? planRect.x0 + "," + planRect.y0
                  + " " + planRect.x1 + "," + planRect.y1 : "";
    const ends =
      " --start-at " + planStart.lat.toFixed(6) + "," + planStart.lon.toFixed(6)
      + " --end-at " + planEnd.lat.toFixed(6) + "," + planEnd.lon.toFixed(6);
    const cmd =
      "# 1. la route : gpx + json avec le profil d'altitude\n"
      + py + " tilehunt.py "
      + (seek ? "compTileSeekerRoute" : "compSquareRoute")
      + " --db " + (c.db || "Database_OSM")
      + (seek ? "" : " --corners " + corners)
      + ends
      + (seek
         ? [...planWant.values()].map(t => " --want " + t[0] + "," + t[1]).join("")
         : "")
      + planVia.map(v => " --via " + v.lat.toFixed(6) + ","
                       + v.lon.toFixed(6)).join("")
      + (c.args ? " " + c.args : "")
      + " --runname " + name
      + "\n\n# 2. la carte avec le trace\n"
      + py + " tilehunt.py buildmap --tiles " + (c.tiles || "france.paved.npz")
      + " --clip-to-mask --basemap esri-gray --runname " + name;
    document.getElementById("plan-text").textContent = cmd;
    cmdBox.style.display = "block";
  } else {
    cmdBox.style.display = "none";
  }
}

(function(){
  const cv2 = document.getElementById("prof");
  if(!cv2) return;
  cv2.addEventListener("pointermove", e=>{
    const r = cv2.getBoundingClientRect();
    profMark = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    drawProfile(); requestDraw();
  });
  cv2.addEventListener("pointerleave", ()=>{
    profMark = null; drawProfile(); requestDraw();
  });
})();

/* ---------- buttons ----------------------------------------------------- */
function tog(id, get, set){
  const b = document.getElementById(id);
  b.onclick = ()=>{ set(!get()); b.classList.toggle("on", get()); requestDraw(); };
}
tog("b-tiles", ()=>showTiles, v=>showTiles=v);
tog("b-grid",  ()=>showGrid,  v=>showGrid=v);
tog("b-base",  ()=>showBase,  v=>showBase=v);
(function(){
  const f = document.getElementById("run-name");
  f.addEventListener("input", ()=>{ f.dataset.touched = "1"; renderPlan(); });
  // typing a name must not pan the map or toggle a layer
  for(const ev of ["keydown","keyup","keypress"])
    f.addEventListener(ev, e=>e.stopPropagation());
})();
tog("b-visited",()=>showVisited,v=>{showVisited=v; paintGrid();});
tog("b-cities",()=>showCities,v=>showCities=v);
tog("b-route", ()=>showRoute, v=>showRoute=v);
tog("b-reg",  ()=>showReg, v=>showReg=v);
tog("b-dep",  ()=>showDep, v=>showDep=v);
document.getElementById("b-mode").onclick = function(){
  mode = (mode === "density") ? "coverage" : "density";
  this.classList.toggle("on", mode === "density");
  this.textContent = mode === "density" ? "coverage" : "density";
  paintGrid(); renderLegend(); updateStats(); requestDraw();
};
document.getElementById("b-red").onclick = function(){
  showRed = !showRed; this.classList.toggle("on", showRed); paintGrid(); requestDraw();
};
document.getElementById("b-green").onclick = function(){
  showGreen = !showGreen; this.classList.toggle("on", showGreen);
  paintGrid(); requestDraw();
};
function clearPlan(){
  planRect = planStart = planEnd = null;
  planVia = []; planWant = new Map(); band = null;
}
function setPlanKind(kind){
  // the two planners are exclusive: switching starts the other one cleanly
  const same = planMode && planKind === kind;
  planKind = kind;
  planMode = same ? 0 : 1;
  clearPlan();
  document.getElementById("b-plan").classList.toggle(
    "on", planMode !== 0 && kind === "square");
  document.getElementById("b-seek").classList.toggle(
    "on", planMode !== 0 && kind === "seeker");
  renderPlan(); requestDraw();
}
document.getElementById("b-plan").onclick = () => setPlanKind("square");
document.getElementById("b-seek").onclick = () => setPlanKind("seeker");
document.getElementById("b-seek-done").onclick = e => {
  e.stopPropagation();
  if(!planWant.size) return;          // nothing to look for yet
  planMode = 4; renderPlan(); requestDraw();
};
document.getElementById("b-seek-via-done").onclick = e => {
  e.stopPropagation();
  planMode = 5; renderPlan(); requestDraw();
};
document.getElementById("qs4").onclick = e => {
  if(e.target.id === "b-seek-via-done") return;
  if(planKind === "seeker" && planMode >= 4){
    planMode = 4; renderPlan(); requestDraw();
  }
};
// re-enterable, like the waypoint step: you will want one more tile
document.getElementById("qs3").onclick = e => {
  if(e.target.id === "b-seek-done") return;
  if(planKind === "seeker" && planMode >= 3){
    planMode = 3; renderPlan(); requestDraw();
  }
};
document.getElementById("b-via-done").onclick = e => {
  e.stopPropagation();
  planMode = 5; renderPlan(); requestDraw();
};
// the step is re-enterable: you will want one more after looking at the map
document.getElementById("ps4").onclick = e => {
  if(e.target.id === "b-via-done") return;
  if(planMode >= 4){ planMode = 4; renderPlan(); requestDraw(); }
};
document.getElementById("b-plan-reset").onclick = ()=>{
  planMode = 1; clearPlan();
  renderPlan(); requestDraw();
};
document.getElementById("b-copy").onclick = function(){
  const txt = document.getElementById("plan-text").textContent;
  try {
    navigator.clipboard.writeText(txt).then(
      ()=>{ this.textContent = "copied"; setTimeout(()=>this.textContent="copy", 1200); },
      ()=>{ this.textContent = "select it by hand"; });
  } catch(err) { this.textContent = "select it by hand"; }
};
window.addEventListener("keydown", e=>{
  if(e.key === "Escape" && planMode){
    planMode = 0; clearPlan();
    document.getElementById("b-plan").classList.remove("on");
    document.getElementById("b-seek").classList.remove("on");
    renderPlan(); requestDraw();
  }
});
document.getElementById("b-fit-route").onclick = ()=>{
  if(!ROUTE) return;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const q of ROUTE.pts){
    if(q[0]<x0)x0=q[0]; if(q[0]>x1)x1=q[0];
    if(q[1]<y0)y0=q[1]; if(q[1]>y1)y1=q[1];
  }
  if(ROUTE.rect && ROUTE.rect.length === 4){
    x0=Math.min(x0,ROUTE.rect[0]); y0=Math.min(y0,ROUTE.rect[1]);
    x1=Math.max(x1,ROUTE.rect[2]+1); y1=Math.max(y1,ROUTE.rect[3]+1);
  }
  const m = 1.25;
  view.scale = Math.min(VW/((x1-x0)*m||1), VH/((y1-y0)*m||1));
  view.cx = (x0+x1)/2; view.cy = (y0+y1)/2;
  requestDraw();
};
document.getElementById("b-reset").onclick = ()=>{
  fitAll(); pick=[]; best=null;
  document.getElementById("zone").style.display="none"; requestDraw();
};
document.getElementById("b-sq").onclick = ()=>{
  const sq = maxSquareIn(D.x0, D.y0, D.x0+W-1, D.y0+H-1);
  best = sq;
  if(sq){
    view.cx = sq.x + sq.n/2; view.cy = sq.y + sq.n/2;
    view.scale = Math.max(2, Math.min(VW,VH)*0.7/sq.n);
    pick = [[sq.x,sq.y],[sq.x+sq.n-1, sq.y+sq.n-1]];
    showZone();
  }
  requestDraw();
};

/* ---------- boot -------------------------------------------------------- */
if(!ROUTE) document.getElementById("b-route").style.display="none";
if(!REGIONS) document.getElementById("b-reg").style.display="none";
if(!DEPTS) document.getElementById("b-dep").style.display="none";
document.getElementById("ttl").textContent = D.title;
const mt = D.meta || {};
document.getElementById("sub").textContent =
  "zoom "+Z+" tiles · "+W+"×"+H+" window · "
  + (D.stats.inscope||0).toLocaleString() + " in scope"
  + (mt.built ? " · " + String(mt.built).slice(0,10) : "");
function setStat(id, label, value){
  document.getElementById("lab-" + id).textContent = label;
  document.getElementById("s-" + id).textContent = value;
}
let flashTile = null, flashUntil = 0, flashTimer = null;
function gotoTile(tx, ty, z){
  // centre on a tile and pulse it, so "the busiest tile" is somewhere you can
  // actually look at rather than a number in a panel
  view.scale = z || Math.max(view.scale, 60);
  view.cx = tx + 0.5; view.cy = ty + 0.5;
  flashTile = [tx, ty]; flashUntil = performance.now() + 4000;
  if(flashTimer) clearInterval(flashTimer);
  flashTimer = setInterval(() => {
    if(performance.now() > flashUntil){
      clearInterval(flashTimer); flashTimer = null; flashTile = null;
    }
    requestDraw();
  }, 60);
  requestDraw();
}

function updateStats(){
  const k = D.kmStats;
  if(mode === "density" && k){
    setStat("green", "total",   fmtKm(k.total) + " km");
    setStat("red",   "0 km",    D.stats.red.toLocaleString());
    setStat("pct",   "median",  fmtKm(k.p50) + " km");
    setStat("out",   "busiest", fmtKm(k.max) + " km");
    const cell = document.getElementById("s-out");
    if(k.top && k.top.length){
      cell.style.cursor = "pointer";
      cell.style.textDecoration = "underline dotted";
      cell.title = "go to " + k.top[0][0] + "," + k.top[0][1];
      cell.onclick = () => gotoTile(k.top[0][0], k.top[0][1]);
    }
  } else {
    const cell = document.getElementById("s-out");
    cell.style.cursor = ""; cell.style.textDecoration = ""; cell.onclick = null;
    cell.title = "";
    if(D.visitedStats){
      const v = D.visitedStats;
      document.getElementById("vis-kv").style.display = "grid";
      document.getElementById("s-have").textContent = v.have.toLocaleString();
      document.getElementById("s-need").textContent = v.need.toLocaleString();
      const sq = v.square;
      const cell = document.getElementById("s-sq");
      cell.textContent = sq[0] + " \u00d7 " + sq[0];
      if(sq[0]){
        cell.style.cursor = "pointer";
        cell.style.textDecoration = "underline dotted";
        cell.title = "go to " + sq[1] + "," + sq[2];
        cell.onclick = () => gotoTile(sq[1] + (sq[0]>>1), sq[2] + (sq[0]>>1),
                                      Math.max(6, 420 / sq[0]));
      }
    } else {
      document.getElementById("vis-kv").style.display = "none";
    }
    setStat("green", "green",   D.stats.green.toLocaleString());
    setStat("red",   "red",     D.stats.red.toLocaleString());
    setStat("pct",   "covered", D.stats.pct + "%");
    setStat("out",   "out",     (W*H - D.stats.inscope).toLocaleString());
  }
}
if(!KMQ) document.getElementById("b-mode").style.display = "none";
if(!ROUTE) document.getElementById("b-fit-route").style.display = "none";
else {
  const st = ROUTE.stats, rows = [];
  const add = (k,v)=>rows.push("<b>"+k+"</b><span>"+v+"</span>");
  if(ROUTE.wanted && ROUTE.wanted.length)
    add("wanted", ROUTE.wanted.length + " tile(s) sought");
  if(ROUTE.skipped && ROUTE.skipped.length)
    add("skipped", ROUTE.skipped.length + " tile(s) with no usable road");
  if(ROUTE.pinned){
    for(const [k, lab] of [["start","start pinned"],["end","finish pinned"]]){
      const q = ROUTE.pinned[k];
      if(q) add(lab, q.m + " m from your point");
    }
    const vs = ROUTE.pinned.via || [];
    if(vs.length)
      add("waypoints", vs.length + " forced, up to "
          + Math.max.apply(null, vs.map(v => v.m)) + " m off");
  }
  if(ROUTE.rect && ROUTE.rect.length === 4){
    const [x0,y0,x1,y1] = ROUTE.rect;
    add("square", (x1-x0+1) + " \u00d7 " + (y1-y0+1));
  }
  if(st.km !== undefined) add("distance", st.km + " km");
  if(st.theory !== undefined)
    add("ideal floor", st.theory + " km"
        + (st.pct ? "  \u00b7  " + st.pct.toFixed(0) + "%" : ""));
  if(st.kmPerTile !== undefined) add("per tile", st.kmPerTile.toFixed(2) + " km");
  if(st.up !== undefined && st.up !== null)
    add("climb", "+" + st.up.toLocaleString() + " / \u2212"
        + (st.down||0).toLocaleString() + " m");
  if(st.tiles !== undefined) add("tiles", (st.covered||0) + " / " + st.tiles);
  if(st.extra) add("extra swept", st.extra);
  if(st.visits !== undefined) add("visits", st.visits);
  document.getElementById("route-rows").innerHTML = rows.join("");
  document.getElementById("routebox").style.display = "block";
}
updateStats();
if(D.note) document.getElementById("note").textContent = D.note;
else document.getElementById("note").style.display = "none";
let attr = BM ? BM.attr
  : 'tile data derived from <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
if(ROUTE && ROUTE.profile && ROUTE.profile.src){
  const src = ROUTE.profile.src;
  if(src.indexOf("GeoTIFF") >= 0)
    attr += ' &middot; heights from the <a href="https://spacedata.copernicus.eu/'
         +  'collections/copernicus-digital-elevation-model">Copernicus DEM</a> '
         +  '(WorldDEM-30 &copy; DLR e.V. and &copy; Airbus Defence and Space '
         +  'GmbH, provided under COPERNICUS by the European Union and ESA)';
  else if(src.length)
    attr += ' &middot; heights from NASA/USGS SRTM';
}
document.getElementById("attr").innerHTML = attr;

if(ROUTE && ROUTE.profile){
  const pr = ROUTE.profile;
  document.getElementById("pf-up").textContent = "+" + pr.up.toLocaleString() + " m";
  document.getElementById("pf-down").textContent = "\u2212" + pr.down.toLocaleString() + " m";
  document.getElementById("scale").style.bottom = "136px";
  // with a profile along the bottom, the right column stops above it
  document.getElementById("rightstack").style.maxHeight =
    "calc(100% - 160px)";
}

window.addEventListener("resize", ()=>{ resize(); drawProfile(); });
resize(); fitAll(); ready = true; draw(); drawProfile();

/* a small console API, handy for poking at the dataset by hand */
window.tilehunt = {
  data: D,
  get view(){ return {cx:view.cx, cy:view.cy, scale:view.scale, VW:VW, VH:VH}; },
  get selection(){ return pick.slice(); },
  status: status,
  tileAt: (lon,lat)=>{ const t = ll2t(lon,lat); return [Math.floor(t[0]), Math.floor(t[1])]; },
  centreOf: (x,y)=> t2ll(x+0.5, y+0.5),
  goto: (x,y,s)=>{ view.cx=x+0.5; view.cy=y+0.5; if(s) view.scale=s; requestDraw(); },
  maxSquare: ()=> maxSquareIn(D.x0, D.y0, D.x0+W-1, D.y0+H-1),
  zoneStats: zoneStats
};
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def bbox_arg(s: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in s.replace(" ", ",").split(",") if p]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be W,S,E,N")
    return parts[0], parts[1], parts[2], parts[3]


_NEGATIVE_VALUE = re.compile(r"^-\d+(\.\d+)?([,\s].*)?$")


def glue_negative_values(argv: Sequence[str]) -> list:
    """
    Let `--clip -5.3,41.2,9.7,51.2` work.

    argparse refuses a value starting with '-' unless it parses as a plain
    negative number, and a comma-separated bbox does not. Rather than make
    every user remember the `--clip=...` form, the value is glued onto the
    option here. Only values that really look like numbers are glued, so a
    genuine missing-argument mistake is still caught by argparse.
    """
    out, i = [], 0
    argv = list(argv)
    while i < len(argv):
        tok = argv[i]
        if (tok.startswith("--") and "=" not in tok and i + 1 < len(argv)
                and _NEGATIVE_VALUE.match(argv[i + 1])):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="tilehunt.py",
        description="VeloViewer z14 explorer-tile coverage from OpenStreetMap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1].split("NOTES")[0] if __doc__ else None,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", metavar="DIR",
                        help="folder holding downloads and .npz datasets "
                             "(default: Data/ next to this script)")
    common.add_argument("--maps-dir", metavar="DIR",
                        help="folder for generated .html "
                             "(default: Maps/ next to this script)")
    common.add_argument("--runs-dir", metavar="DIR",
                        help="folder holding one sub-folder per named ride "
                             "(default: Runs/ next to this script)")

    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("genBDD", parents=[common], aliases=["tiles"],
                       help="build the database: rasterise paved cyclable "
                            "roads onto the z14 grid, and optionally the "
                            "road graph and the heights alongside")
    t.add_argument("--pbf", action="append", help="input .osm.pbf (repeatable)")
    t.add_argument("--geojson", action="append", help="input line GeoJSON (repeatable)")
    t.add_argument("--out", required=True)
    t.add_argument("--idx", default="flex_mem",
                   help="pyosmium node-location index; use "
                        "'sparse_file_array,nodes.idx' if short on RAM")
    t.add_argument("--bbox", type=bbox_arg, help="clip result to W,S,E,N")
    t.add_argument("--pad", type=int, default=0, help="pad the output frame by N tiles")
    t.add_argument("--no-tag-filter", action="store_true",
                   help="for --geojson: accept every line without tag checks")
    t.add_argument("--accept",
                   choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle",
                   help="which ways count. 'paved-cycle' (default): paved roads "
                        "where cycling is allowed. 'all-roads': every road except "
                        "motorways, ignoring surface and access. 'any-highway': "
                        "every highway at all, including tracks and footpaths - "
                        "the right choice for building a scope mask.")
    t.add_argument("--db", metavar="DIR",
                   help="also build the road database used by `route`, so the "
                        "PBF never needs reading again (try Database_OSM)")
    t.add_argument("--no-lengths", action="store_true",
                   help="skip measuring road length per tile (slightly faster, "
                        "but the map then has no density mode)")
    t.add_argument("--no-cobbles", action="store_true",
                   help="treat sett/cobblestone (pave) as unpaved")
    t.add_argument("--trust-tracktype", action="store_true",
                   help="accept track + tracktype=grade1 with no surface tag; "
                        "off by default because grade1 also covers compacted "
                        "hardcore, not just tarmac")
    t.add_argument("--strict-surface", action="store_true",
                   help="require an explicit paved surface tag; ways with no "
                        "surface at all are rejected instead of assumed paved")
    t.add_argument("--dump-rejects", metavar="FILE",
               help="write every rejected way and the rule that rejected it")
    t.add_argument("-v", "--verbose", action="store_true")
    t.add_argument("--dem-db", metavar="DIR",
                   help="build the elevation half of the database in the same "
                        "run, covering the tiles that hold a road: one npz "
                        "per 8x8-tile block, read on demand afterwards")
    t.add_argument("--dem-src", metavar="DIR",
                   help="build the elevation from 1-degree tiles already on "
                        "disk; without it they are read straight out of the "
                        "Copernicus bucket, fetching only the bytes needed")
    t.add_argument("--dem-res", type=int, default=DEM_RES,
                   help=f"elevation samples per explorer tile edge "
                        f"(default {DEM_RES}, about 25 m at 48N)")
    t.add_argument("--dem-force", action="store_true",
                   help="rebuild elevation blocks that are already complete")
    t.set_defaults(func=cmd_tiles)

    m = sub.add_parser("mask", parents=[common], help="which tiles are in scope (land / country)")
    m.add_argument("--boundary",
                   help="polygon GeoJSON (default: countries.geojson in the Data folder)")
    m.add_argument("--iso", help="keep only features with this ISO_A3 (e.g. FRA)")
    m.add_argument("--clip", type=bbox_arg, help="clip to W,S,E,N (drops overseas parts)")
    m.add_argument("--min-part-area", type=float, default=0.0,
                   help="drop polygon parts smaller than this, in square degrees")
    m.add_argument("--out", required=True)
    m.add_argument("-v", "--verbose", action="store_true")
    m.set_defaults(func=cmd_mask)

    g = sub.add_parser("buildmap", parents=[common], aliases=["map"],
                       help="build the self-contained interactive HTML")
    g.add_argument("--tiles", required=True)
    g.add_argument("--mask")
    g.add_argument("--visited", metavar="NPZ",
                   help="the tiles you have already ridden, from genVisited: "
                        "adds a layer, the count still to get, and your max "
                        "square")
    g.add_argument("--out")
    g.add_argument("--boundary", help="polygon GeoJSON for the vector basemap")
    g.add_argument("--iso", default="FRA", help="ISO_A3 to highlight in the basemap")
    g.add_argument("--simplify", type=float, default=0.012,
                   help="basemap simplification tolerance, degrees")
    g.add_argument("--basemap",
                   choices=["none", "osm", "esri-gray", "esri-dark-gray"],
                   default="none",
                   help="raster tiles under the grid. 'none' (default) keeps "
                        "the page fully offline. 'osm' is OpenStreetMap's own "
                        "(rate-limited) servers; add --basemap-grey to "
                        "desaturate them. 'esri-gray' / 'esri-dark-gray' are "
                        "ready-made grey canvases needing no key.")
    g.add_argument("--basemap-url", metavar="TEMPLATE",
                   help="any XYZ tile template, e.g. "
                        "'https://{s}.example.com/{z}/{x}/{y}.png'. Supports "
                        "{s} {z} {x} {y} and {r} (which becomes @2x on retina). "
                        "Use this for a provider needing a key.")
    g.add_argument("--basemap-sub", metavar="LETTERS",
                   help="subdomain letters for {s}, e.g. abc")
    g.add_argument("--basemap-attr", metavar="HTML",
                   help="attribution line for a custom basemap (required by "
                        "most providers' terms)")
    g.add_argument("--basemap-dark", action="store_true",
                   help="tell the viewer the basemap is dark, so labels stay light")
    g.add_argument("--basemap-budget", type=int, default=400,
                   help="most tiles to request for one view (400)")
    g.add_argument("--basemap-everywhere", action="store_true",
                   help="draw the basemap over the whole world; by default it "
                        "is clipped to the mapped country so the surroundings "
                        "keep the plain dark background")
    g.add_argument("--basemap-grey", action="store_true",
                   help="desaturate the raster basemap in the browser; use it "
                        "to grey out --basemap osm")
    g.add_argument("--basemap-opacity", type=float, default=1.0,
                   help="0..1, how strongly the raster basemap shows (1.0)")
    g.add_argument("--basemap-detail", type=float, default=0.6,
                   help="0..1, how strongly the basemap is re-drawn on top of "
                        "the tiles in multiply blending, so roads and town "
                        "shapes stay readable through the colour (0.6; 0 off)")
    g.add_argument("--tile-opacity", type=float, default=None,
                   help="0..1, how opaque the green/red tiles are. Defaults to "
                        "0.8 with no basemap and 0.45 with one, so the map "
                        "underneath stays readable.")
    g.add_argument("--regions", metavar="GEOJSON",
                   help="polygon GeoJSON of regions, drawn as bold outlines")
    g.add_argument("--departments", "--departements", metavar="GEOJSON",
                   dest="departments",
                   help="polygon GeoJSON of departements, drawn as fine outlines")
    g.add_argument("--admin-simplify", type=float, default=0.0022,
                   help="simplification tolerance for the admin outlines, in "
                        "degrees (0.0022 is about 200 m); lower is sharper and "
                        "bigger")
    g.add_argument("--dem-name", default="DEM_DB", metavar="DIR",
                   help="elevation folder named in the commands the plan "
                        "panel hands you (default DEM_DB, the block "
                        "database)")
    g.add_argument("--route-args", default="", metavar="FLAGS",
                   help="extra flags written into the command the plan panel "
                        "hands you. Empty by default, since --cands 6, "
                        "--seconds 80 and --reopt-seconds 30 are now what the "
                        "solver does anyway.")
    g.add_argument("--python-cmd", default="python", metavar="NAME",
                   help="how the commands in the plan panel should invoke "
                        "Python (python)")
    g.add_argument("--db-name", metavar="DIR",
                   help="name of the road database, used only to write the "
                        "ready-made command in the map's plan panel "
                        "(Database_OSM)")
    g.add_argument("--route", metavar="JSON",
                   help="a route written by the `route` command, drawn on top")
    g.add_argument("--clip-to-mask", action="store_true",
                   help="discard road tiles outside --mask and crop the window "
                        "to the mask (keeps the map about one country)")
    g.add_argument("--title", default="Explorer tile coverage")
    g.add_argument("--note", default="")
    g.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="build the map of a named ride: reads "
                        "Runs/<kind>/NAME/NAME.route.json and writes "
                        "NAME.html beside it")
    g.set_defaults(func=cmd_map, run_kind="Square",
                   run_files=(("route", ".route.json"), ("out", ".html")))

    z = sub.add_parser("zone", parents=[common], help="report on a rectangle of tiles")
    z.add_argument("--corners", nargs=2, type=parse_tile, required=True,
                   metavar="X,Y", help="two opposite corner tiles")
    z.add_argument("--tiles", help="dataset, to report green/red")
    z.add_argument("--list-limit", type=int, default=40)
    z.set_defaults(func=cmd_zone)

    y = sub.add_parser("why", parents=[common], help="explain a tile: which ways made it green")
    y.add_argument("--pbf", action="append", required=True)
    y.add_argument("--tile", action="append", type=parse_tile, required=True,
                   metavar="X,Y", help="tile to explain (repeatable)")
    y.add_argument("--accept",
                   choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle")
    y.add_argument("--no-cobbles", action="store_true")
    y.add_argument("--strict-surface", action="store_true")
    y.add_argument("--trust-tracktype", action="store_true")
    y.add_argument("--show-rejected", action="store_true",
                   help="also list the ways that were thrown out, and why")
    y.add_argument("--limit", type=int, default=25)
    y.add_argument("--idx", default="flex_mem")
    y.set_defaults(func=cmd_why)

    # Both routing modes take the same options but for how the area is
    # described, so the options are declared once and shared.
    route_opts = argparse.ArgumentParser(add_help=False)
    route_opts.add_argument("--db", metavar="DIR",
                   help="the road database built by `genBDD --db`; with this, "
                        "no .osm.pbf is read at all")
    route_opts.add_argument("--pbf", action="append",
                   help="OSM extract to build the road graph from, when there "
                        "is no database")
    route_opts.add_argument("--graph-cache", metavar="NPZ",
                   help="reuse (or create) a cached graph for this area; makes "
                        "repeated runs on the same zone instant")
    route_opts.add_argument("--start", type=parse_tile, metavar="X,Y",
                   help="tile to start in (default: the NW corner)")
    route_opts.add_argument("--end", type=parse_tile, metavar="X,Y",
                   help="tile to finish in (default: the SE corner)")
    route_opts.add_argument("--start-at", type=parse_latlon, metavar="LAT,LON",
                   help="exact point to start from, pinned to the nearest "
                        "junction of the network. May sit outside the "
                        "rectangle. Overrides --start.")
    route_opts.add_argument("--end-at", type=parse_latlon, metavar="LAT,LON",
                   help="exact point to finish at. Overrides --end.")
    route_opts.add_argument("--away-max", type=float, default=100.0,
                   metavar="KM",
                   help="refuse an exact endpoint lying more than this far "
                        "OUTSIDE the area covered (100 km). A point inside is "
                        "never refused, however long the ride; the usual "
                        "cause is LAT,LON given the wrong way round")
    route_opts.add_argument("--via", action="append", type=parse_latlon,
                   metavar="LAT,LON",
                   help="a point the ride must go through, pinned like the "
                        "endpoints. Repeat for several. The solver chooses "
                        "the order it rides them in.")
    route_opts.add_argument("--snap-max", type=float, default=150.0, metavar="M",
                   help="refuse to pin an endpoint further than this from the "
                        "point you gave (150 m)")
    route_opts.add_argument("--ignore-oneway", action="store_true",
                   help="treat every road as two-way. Off by default: one-way "
                        "streets are respected, except where OSM says bicycles "
                        "may take the contraflow (oneway:bicycle=no, "
                        "cycleway=opposite*), which is how French double-sens "
                        "cyclable is tagged.")
    route_opts.add_argument("--margin", type=int, default=2,
                   help="tiles of road to load around the rectangle, so the "
                        "ride may leave it when that is shorter (2)")
    route_opts.add_argument("--cands", type=int, default=6,
                   help="candidate junctions per tile (6). More is slower but "
                        "lets the route enter a tile at a better point.")
    route_opts.add_argument("--seconds", type=float, default=80.0,
                   help="time budget for improving the visit order (80)")
    route_opts.add_argument("--prune-seconds", type=float, default=60.0,
                   help="extra budget for dropping redundant visits (60)")
    route_opts.add_argument("--seed", type=int, default=12345,
                   help="random seed for the restarts, so a run is repeatable")
    route_opts.add_argument("--reopt-seconds", type=float, default=30.0,
                   help="budget for re-optimising the visit order once the "
                        "redundant visits have been dropped (30)")
    route_opts.add_argument("--no-progress", action="store_true",
                   help="do not print the running best distance")
    route_opts.add_argument("--no-prune", action="store_true",
                   help="keep one explicit visit per tile; much worse, but "
                        "useful to see what the pruning is worth")
    route_opts.add_argument("--pin-ends", action="store_true",
                   help="force the exact start and end junctions rather than "
                        "letting the solver choose inside those tiles")
    route_opts.add_argument("--require-all", action="store_true",
                   help="stop rather than skip, if any tile of the rectangle "
                        "has no usable road. By default such tiles are "
                        "reported and left out, and the rest is ridden.")
    route_opts.add_argument("--allow-gaps", action="store_true",
                   help=argparse.SUPPRESS)      # now the default; kept so old
                                                # command lines still run
    route_opts.add_argument("--batch", type=int, default=48,
                   help="Dijkstra sources per batch (memory vs speed)")
    route_opts.add_argument("--accept", choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle")
    route_opts.add_argument("--no-cobbles", action="store_true")
    route_opts.add_argument("--strict-surface", action="store_true")
    route_opts.add_argument("--trust-tracktype", action="store_true")
    route_opts.add_argument("--idx", default="flex_mem")
    route_opts.add_argument("--dem", metavar="DIR",
                   help="where the heights come from: the block database "
                        "built by `demdb` (or by `genBDD --dem-db`), or a "
                        "folder of 1-degree Copernicus .tif / SRTM .hgt "
                        "tiles. Adds the profile and the climbing totals to "
                        "the json, the gpx and the map. Left out, Data/DEM_DB "
                        "is used when it exists.")
    route_opts.add_argument("--no-dem-fetch", action="store_true",
                   help="never touch the network for heights: use only what "
                        "the database already holds")
    route_opts.add_argument("--no-dem", action="store_true",
                   help="skip the elevation profile entirely")
    route_opts.add_argument("--dem-res", type=int, default=DEM_RES,
                   help=f"samples per explorer tile edge when blocks have to "
                        f"be built (default {DEM_RES})")
    route_opts.add_argument("--dem-src", metavar="DIR",
                   help="build any missing heights from 1-degree tiles on "
                        "disk instead of over the network")
    route_opts.add_argument("--dem-step", type=float, default=25.0,
                   help="spacing in metres at which the track is sampled (25)")
    route_opts.add_argument("--dem-smooth", type=float, default=120.0,
                   help="smoothing window in metres before counting the "
                        "climbs (120). Raw DEM noise invents hundreds of "
                        "metres of ascent on a flat road.")
    route_opts.add_argument("--dem-threshold", type=float, default=3.0,
                   help="metres the profile must turn by before a climb is "
                        "counted (3). Lower it and DEM noise becomes ascent.")
    route_opts.add_argument("--gpx", help="write the ride as a .gpx track")
    route_opts.add_argument("--out", help="write the route as .json for `map --route`")
    route_opts.add_argument("--name", help="track name inside the gpx")
    route_opts.add_argument("-v", "--verbose", action="store_true")
    route_opts.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="name this ride: the gpx and the json are written to "
                        "Runs/Square/NAME/NAME.* and the track is named NAME")

    r = sub.add_parser("compSquareRoute", parents=[common, route_opts],
                       aliases=["route"],
                       help="mode 1 (square): shortest ride covering a "
                            "rectangle of tiles, from a start point to an "
                            "end point")
    r.add_argument("--corners", nargs=2, type=parse_tile, required=True,
                   metavar="X,Y", help="two opposite corner tiles of the rectangle")
    r.set_defaults(func=cmd_route, run_kind="Square", want=None,
                   run_files=(("gpx", ".gpx"), ("out", ".route.json")))

    k = sub.add_parser("compTileSeekerRoute", parents=[common, route_opts],
                       aliases=["seek"],
                       help="mode 3 (seeker): shortest ride collecting a "
                            "chosen set of tiles, from a start point to an "
                            "end point")
    k.add_argument("--want", action="append", type=parse_tile, metavar="X,Y",
                   required=True,
                   help="a tile the ride must go through. Repeat for each "
                        "one. Only the bounding box of these and the two "
                        "ends, widened by --margin, is ever loaded.")
    k.set_defaults(func=cmd_route, run_kind="TileSeeker", corners=None,
                   run_files=(("gpx", ".gpx"), ("out", ".route.json")))

    f = sub.add_parser("profile", parents=[common],
                       help="add the elevation profile to a route already "
                            "solved, without redoing the optimisation")
    f.add_argument("--route", metavar="JSON")
    f.add_argument("--dem", metavar="DIR",
                   help="elevation source (default Data/DEM_DB, built on "
                        "demand)")
    f.add_argument("--dem-res", type=int, default=DEM_RES)
    f.add_argument("--dem-src", metavar="DIR")
    f.add_argument("--no-dem-fetch", action="store_true")
    f.add_argument("--gpx", help="also rewrite the gpx with elevations")
    f.add_argument("--name")
    f.add_argument("--dem-step", type=float, default=25.0)
    f.add_argument("--dem-smooth", type=float, default=120.0)
    f.add_argument("--dem-threshold", type=float, default=3.0)
    f.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="work on a named ride under Runs/<kind>/NAME/")
    f.set_defaults(func=cmd_profile, run_kind="Square",
                   run_files=(("route", ".route.json"), ("gpx", ".gpx")))

    e = sub.add_parser("dem", parents=[common],
                       help="which elevation tiles a zone needs, and a script "
                            "to fetch the missing ones")
    e.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y",
                   help="the two opposite tiles of a square")
    e.add_argument("--margin", type=int, default=2,
                   help="tiles of slack around the square (default 2, the "
                        "same slack the solver is allowed)")
    e.add_argument("--tiles", metavar="NPZ",
                   help="instead of a square, every tile holding a road in a "
                        "coverage npz -- this is the whole-country answer")
    e.add_argument("--dem", metavar="DIR",
                   help="folder already holding elevation tiles; only what is "
                        "missing is listed")
    e.add_argument("--source", choices=("copernicus", "srtm"),
                   default="copernicus",
                   help="which dataset to fetch (default copernicus)")
    e.add_argument("--out", metavar="FILE",
                   help="write a download script (.ps1 for PowerShell, "
                        "anything else for sh) instead of printing URLs")
    e.set_defaults(func=cmd_dem)

    q = sub.add_parser("demdb", parents=[common],
                       help="cut elevation into per-block files, so a ride "
                            "only ever loads what it crosses")
    q.add_argument("--out", required=True, metavar="DIR",
                   help="the database folder, e.g. DEM_DB")
    q.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y",
                   help="build just this square")
    q.add_argument("--margin", type=int, default=2,
                   help="tiles of slack around the square (default 2)")
    q.add_argument("--tiles", metavar="NPZ",
                   help="instead of a square, every tile holding a road in a "
                        "coverage npz -- the whole-country build")
    q.add_argument("--src", metavar="DIR",
                   help="build from 1-degree tiles already on disk; without "
                        "it the heights are read straight out of the "
                        "Copernicus bucket, fetching only the bytes needed")
    q.add_argument("--res", type=int, default=DEM_RES,
                   help=f"samples per explorer tile edge (default {DEM_RES}, "
                        f"about 25 m at 48N; 32 halves the size)")
    q.add_argument("--force", action="store_true",
                   help="rebuild blocks that are already there")
    q.set_defaults(func=cmd_demdb)

    sn = sub.add_parser("snap", parents=[common],
                        help="where a point lands on the road network")
    sn.add_argument("--at", required=True, type=parse_latlon, metavar="LAT,LON")
    sn.add_argument("--db", metavar="DIR")
    sn.add_argument("--pbf")
    sn.add_argument("--pad", type=float, default=0.02, metavar="DEG",
                    help="half-width of the area loaded around the point "
                         "(0.02 degrees, about 2 km)")
    sn.add_argument("--accept", choices=("paved-cycle", "all-roads",
                                         "any-highway"), default="paved-cycle")
    sn.add_argument("--no-cobbles", action="store_true")
    sn.add_argument("--strict-surface", action="store_true")
    sn.add_argument("--trust-tracktype", action="store_true")
    sn.add_argument("--ignore-oneway", action="store_true")
    sn.add_argument("--idx")
    sn.add_argument("-v", "--verbose", action="store_true")
    sn.set_defaults(func=cmd_snap)

    vi = sub.add_parser("genVisited", parents=[common], aliases=["visited"],
                        help="the tiles you have already ridden, from your "
                             "Statshunters share link")
    vi.add_argument("--share", metavar="LINK",
                    help="your share link from https://www.statshunters.com/"
                         "share (or just its code). Remembered afterwards, so "
                         "you only pass it once.")
    vi.add_argument("--out", default="visited.npz",
                    help="where to write the grid (visited.npz)")
    vi.add_argument("--type", metavar="LIST",
                    help="keep only these activity types, comma separated, "
                         "e.g. Ride or Ride,VirtualRide. Default: all.")
    vi.add_argument("--since", metavar="YYYY-MM-DD",
                    help="ignore activities before this date")
    vi.add_argument("--until", metavar="YYYY-MM-DD",
                    help="ignore activities after this date")
    vi.add_argument("--timeout", type=float, default=60.0)
    vi.add_argument("--no-remember", dest="remember", action="store_false",
                    help="do not write the share link to Data/"
                         + SHARE_FILE)
    vi.add_argument("--share-base", help=argparse.SUPPRESS)
    vi.add_argument("-v", "--verbose", action="store_true")
    vi.set_defaults(func=cmd_visited)

    d = sub.add_parser("dbinfo", parents=[common],
                       help="what the road database holds for given tiles")
    d.add_argument("--db", required=True, metavar="DIR")
    d.add_argument("--tile", action="append", type=parse_tile, metavar="X,Y")
    d.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y")
    d.set_defaults(func=cmd_dbinfo)

    i = sub.add_parser("info", parents=[common], help="describe a dataset")
    i.add_argument("--top", type=int, default=10,
                   help="how many of the busiest tiles to list (10)")
    i.add_argument("--tiles", required=True)
    i.set_defaults(func=cmd_info)

    a = p.parse_args(glue_negative_values(
        sys.argv[1:] if argv is None else argv))
    apply_layout(a)
    a.func(a)


if __name__ == "__main__":
    main()
