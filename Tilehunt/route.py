"""compSquareRoute and compTileSeekerRoute: from a request to a gpx and a route json."""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from .tiles import haversine_m, lonlat_to_tilef, tile_bounds, walk_polyline
from .osm import pick_classifier
from .graph import PathCache, RoadGraph, build_graph, components, many_to_many, nodes_in_tile, pick_candidates, ridable_mask, seed_in_tile, snap_to_graph
from .db import load_db_graph
from .solver import Progress, choose_candidates, cluster_matrix, covered_tiles, iterated_local_search, nn_path, prune_redundant, route_length_m, theoretical_min_m, trace_route, two_opt
from .dem import DEM_INDEX, default_dem, ensure_dem, open_dem
from .profile import profile_from_coords, write_gpx


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
