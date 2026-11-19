"""The road database: one npz per z11 block, written in the same pass as the tile grid, read back without the pbf."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from .tiles import ZOOM, haversine_m, lonlat_to_tilef
from .osm import ASSUME_UNPAVED, COBBLE_SURFACES, PAVED_SURFACES, classify_all_roads, classify_way
from .graph import RoadGraph, bike_directions, csr_min_duplicates


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


def cmd_dbinfo(a: argparse.Namespace) -> None:
    """Say exactly what the database holds for a tile, and what it does not."""
    from .tiles import MAX_LAT, NTILES, ZOOM
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
