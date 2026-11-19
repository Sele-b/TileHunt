"""What counts as a paved road you may cycle on, and reading ways out of a .osm.pbf or a GeoJSON."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Sequence

import numpy as np

from .tiles import walk_polyline


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
    from .db import DBWriter
    from .dem import LocalSource, RemoteCopernicus, dem_blocks_for_tiles, dem_build
    from .tiles import TileGrid, lonlat_to_tilef, tile_size_m
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
    from .tiles import tile_bounds, walk_polyline
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
