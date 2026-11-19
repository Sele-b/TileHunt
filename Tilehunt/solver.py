"""The generalised TSP: ordering, local search, pruning visits made redundant by tiles swept on the way, and the ideal floor."""
from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter
from typing import Sequence

import numpy as np

from .tiles import EARTH_CIRCUMFERENCE_M, NTILES, haversine_m, lonlat_to_tilef, walk_polyline
from .graph import PathCache, RoadGraph


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
