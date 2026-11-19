"""The road graph: building it, connectivity, snapping points onto it, candidate junctions and shortest paths."""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Sequence

import numpy as np

from .tiles import MAX_LAT, NTILES, haversine_m, tilef_to_lonlat, walk_polyline


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


def cmd_snap(a: argparse.Namespace) -> None:
    """Where a point lands on the network, before committing to a long solve."""
    from .db import load_db_graph
    from .osm import pick_classifier
    from .tiles import lonlat_to_tilef
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
