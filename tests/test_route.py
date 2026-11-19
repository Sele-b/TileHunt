#!/usr/bin/env python3
"""
End-to-end test of mode 1 on a synthetic grid whose optimum is known.

The network is a regular lattice with one road through the centre of every tile
row and every tile column. On such a lattice a boustrophedon sweep is optimal,
so for a W x H rectangle the shortest covering path from one corner to the
opposite one is  H*(W-1) tile widths + (H-1) tile heights.
"""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import json
import math
import os
import time
import shutil
import subprocess
import sys

import osmium
from tilehunt import tilef_to_lonlat, tile_size_m

OUT = os.path.join(tempfile.gettempdir(), "tilehunt-routetest")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
PBF = os.path.join(OUT, "grid.osm.pbf")

X0, Y0, W, H = 8320, 5700, 6, 5          # rectangle of tiles
PAD = 1                                   # lattice extends one tile beyond

nodes, ways = [], []
nid = wid = 0


def N(lon, lat):
    global nid
    nid += 1
    nodes.append((nid, lon, lat))
    return nid


def W_(tags, ids):
    global wid
    wid += 1
    ways.append((wid, tags, ids))


# lattice points at the centre of every tile, one tile of padding all round
cols = range(X0 - PAD, X0 + W + PAD)
rows = range(Y0 - PAD, Y0 + H + PAD)
pt = {}
for ty in rows:
    for tx in cols:
        lon, lat = tilef_to_lonlat(tx + 0.5, ty + 0.5)
        pt[(tx, ty)] = N(lon, lat)

for ty in rows:                            # east-west roads
    W_({"highway": "residential", "name": f"row {ty}"},
       [pt[(tx, ty)] for tx in cols])
for tx in cols:                            # north-south roads
    W_({"highway": "residential", "name": f"col {tx}"},
       [pt[(tx, ty)] for ty in rows])

writer = osmium.SimpleWriter(PBF)
for i, lon, lat in nodes:
    writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), version=1))
for i, tags, nd in ways:
    writer.add_way(osmium.osm.mutable.Way(id=i, nodes=nd, tags=tags, version=1))
writer.close()
print(f"grid: {len(nodes)} nodes, {len(ways)} ways -> {PBF}")

tw, th = tile_size_m(Y0 + H // 2)
optimum_km = (H * (W - 1) * tw + (H - 1) * th) / 1000.0
print(f"tile {tw:.0f} x {th:.0f} m; boustrophedon optimum = {optimum_km:.2f} km")

# A second, finer network: three lattice lines per tile instead of one, so a
# tile holds several candidate junctions and the DP has a real choice to make.
PBF_FINE = os.path.join(OUT, "fine.osm.pbf")
nodes, ways = [], []
nid = wid = 0
STEP = 3
fcols = [X0 - PAD + i / STEP for i in range((W + 2 * PAD) * STEP + 1)]
frows = [Y0 - PAD + j / STEP for j in range((H + 2 * PAD) * STEP + 1)]
fpt = {}
for fy in frows:
    for fx in fcols:
        lon, lat = tilef_to_lonlat(fx + 0.5 / STEP, fy + 0.5 / STEP)
        fpt[(fx, fy)] = N(lon, lat)
for fy in frows:
    W_({"highway": "residential"}, [fpt[(fx, fy)] for fx in fcols])
for fx in fcols:
    W_({"highway": "residential"}, [fpt[(fx, fy)] for fy in frows])
writer = osmium.SimpleWriter(PBF_FINE)
for i, lon, lat in nodes:
    writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), version=1))
for i, tags, nd in ways:
    writer.add_way(osmium.osm.mutable.Way(id=i, nodes=nd, tags=tags, version=1))
writer.close()
print(f"fine grid: {len(nodes)} nodes, {len(ways)} ways")

fails = []
results = {}
for label, extra, pbf in (("no-prune", ["--no-prune"], PBF),
                          ("pruned", [], PBF),
                          ("fine", [], PBF_FINE)):
    out_json = os.path.join(OUT, f"route-{label}.json")
    cmd = [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "route",
           "--pbf", pbf,
           "--corners", f"{X0},{Y0}", f"{X0+W-1},{Y0+H-1}",
           "--start", f"{X0},{Y0}", "--end", f"{X0+W-1},{Y0+H-1}",
           "--margin", "1", "--cands", "4", "--seconds", "10",
           "--prune-seconds", "25",
           "--out", out_json,
           "--gpx", os.path.join(OUT, f"route-{label}.gpx")] + extra
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"\n----- {label} -----")
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr[-2000:])
        fails.append(label)
        continue
    with open(out_json, encoding="utf-8") as fh:
        data = json.load(fh)
    results[label] = data
    st = data["stats"]
    if st["covered"] != st["tiles"]:
        print(f"FAIL {label}: only {st['covered']}/{st['tiles']} covered")
        fails.append(label)
    # the traced polyline must be continuous: no jump longer than a tile
    cs = data["coords"]
    worst = 0.0
    for (lo1, la1), (lo2, la2) in zip(cs, cs[1:]):
        d = math.hypot((lo2 - lo1) * 111320 * math.cos(math.radians(la1)),
                       (la2 - la1) * 110574)
        worst = max(worst, d)
    print(f"   longest step between consecutive points: {worst:.0f} m")
    if worst > tw * 1.2:
        print(f"FAIL {label}: the track jumps {worst:.0f} m, it is not continuous")
        fails.append(label)
    # endpoints must be in the requested tiles
    if data["visits"][0] != data["start"] or data["visits"][-1] != data["end"]:
        print(f"FAIL {label}: route does not start/end in the requested tiles")
        fails.append(label)

# ---------------------------------------------------------------- oneway ---
# A lattice whose rows alternate direction, like a city grid. The route must
# never ride an arc the wrong way, and must still claim every tile.
PBF_OW = os.path.join(OUT, "oneway.osm.pbf")
nodes, ways = [], []
nid = wid = 0
pt = {}
for ty in rows:
    for tx in cols:
        lon, lat = tilef_to_lonlat(tx + 0.5, ty + 0.5)
        pt[(tx, ty)] = N(lon, lat)
for k, ty in enumerate(rows):
    ids = [pt[(tx, ty)] for tx in cols]
    if k % 2:
        ids = ids[::-1]
    W_({"highway": "residential", "oneway": "yes"}, ids)
for k, tx in enumerate(cols):
    ids = [pt[(tx, ty)] for ty in rows]
    if k % 2:
        ids = ids[::-1]
    W_({"highway": "residential", "oneway": "yes"}, ids)
writer = osmium.SimpleWriter(PBF_OW)
for i, lon, lat in nodes:
    writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), version=1))
for i, tags, nd in ways:
    writer.add_way(osmium.osm.mutable.Way(id=i, nodes=nd, tags=tags, version=1))
writer.close()
print(f"\none-way grid: {len(nodes)} nodes, {len(ways)} ways")

ow_json = os.path.join(OUT, "route-oneway.json")
res = subprocess.run(
    [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "route",
     "--pbf", PBF_OW, "--corners", f"{X0},{Y0}", f"{X0+W-1},{Y0+H-1}",
     "--start", f"{X0},{Y0}", "--end", f"{X0+W-1},{Y0+H-1}",
     "--margin", "1", "--seconds", "10", "--prune-seconds", "20",
     "--graph-cache", os.path.join(OUT, "ow.graph.npz"),
     "--out", ow_json], capture_output=True, text=True)
print(res.stdout.strip())
if res.returncode != 0:
    print(res.stderr[-1500:]); fails.append("oneway-run")
else:
    from tilehunt import RoadGraph
    import numpy as np
    g = RoadGraph.load(os.path.join(OUT, "ow.graph.npz"))
    if not g.directed:
        print("FAIL the cached graph is not marked directed"); fails.append("ow-flag")
    with open(ow_json, encoding="utf-8") as fh:
        ow = json.load(fh)
    st = ow["stats"]
    print(f"   {st['km']} km, {st['covered']}/{st['tiles']} tiles")
    if st["covered"] != st["tiles"]:
        fails.append("ow-coverage")
    # every consecutive pair of the track must be an arc that exists
    coord_to_i = {}
    for i in range(g.n):
        coord_to_i[(round(float(g.lon[i]), 6), round(float(g.lat[i]), 6))] = i
    csr = g.csr
    bad_arcs = 0
    seq = [coord_to_i.get((round(lo, 6), round(la, 6))) for lo, la in ow["coords"]]
    for u, v in zip(seq, seq[1:]):
        if u is None or v is None:
            continue
        if v not in csr.indices[csr.indptr[u]:csr.indptr[u + 1]]:
            bad_arcs += 1
    if bad_arcs:
        print(f"FAIL the track rides {bad_arcs} arc(s) against the one-way")
        fails.append("ow-direction")
    else:
        print(f"   every one of {len(seq)-1} steps follows a legal arc")

# --------------------------------------------------------- spur removal ---
# The symptom from the field: the ride detours up into a tile and straight
# back, when a later leg passes through that tile anyway. Such a visit must
# be dropped -- and a visit whose tile nothing else covers must be kept.
from tilehunt import prune_redundant
import types as _types


class _Cache:
    """Minimal stand-in for PathCache: answers which tiles a leg sweeps."""

    def __init__(self, legs, tile_of):
        self._legs = legs
        self.flat = list(range(len(tile_of)))
        self.g = _types.SimpleNamespace(
            lon=[0.0] * len(tile_of), lat=[0.0] * len(tile_of))
        self._tile_of = tile_of

    def tiles(self, a, b):
        return frozenset(self._legs.get((a, b), self._legs.get((b, a), frozenset())))


def _spur_case(leg02_covers_T1):
    # four visits in a line; visit 1 sits on a there-and-back spur
    T = [(0, 0), (1, 1), (2, 0), (3, 0)]
    big = 50_000.0
    D = np.array([
        [0.0,  big, 1000.0, 2000.0],
        [big,  0.0, big,    3000.0],
        [1000.0, big, 0.0,  1000.0],
        [2000.0, 3000.0, 1000.0, 0.0],
    ])
    legs = {
        (0, 1): {T[0], T[1]}, (1, 2): {T[1], T[2]},
        (0, 2): {T[0], T[2]} | ({T[1]} if leg02_covers_T1 else set()),
        (2, 3): {T[2], T[3]},
    }
    cache = _Cache(legs, T)
    order = [0, 1, 2, 3]
    visits = [0, 1, 2, 3]
    groups = [[0], [1], [2], [3]]
    required = set(T)
    o, v, tot = prune_redundant(D, groups, order, visits, cache, required,
                                0, 3, time.time() + 5, False, None)
    return o, tot


import time
o_yes, t_yes = _spur_case(True)
o_no, t_no = _spur_case(False)
print(f"\nspur covered elsewhere  -> visits {o_yes}, {t_yes:.0f} m")
print(f"spur covered nowhere    -> visits {o_no}, {t_no:.0f} m")
if 1 in o_yes:
    print("FAIL the redundant spur was not removed")
    fails.append("spur-kept")
if 1 not in o_no:
    print("FAIL a visit was removed although its tile would be lost")
    fails.append("spur-overpruned")

# ------------------------------------------------- solver/geometry match ---
# Without pruning the traced polyline must be exactly as long as the sum of
# shortest paths the solver optimised, or the solver is optimising against
# distances the ride does not actually have.
inv = subprocess.run(
    [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "route",
     "--pbf", PBF_FINE, "--corners", f"{X0},{Y0}", f"{X0+W-1},{Y0+H-1}",
     "--start", f"{X0},{Y0}", "--end", f"{X0+W-1},{Y0+H-1}",
     "--margin", "1", "--seconds", "6", "--no-prune", "--no-progress"],
    capture_output=True, text=True)
import re as _re
m1 = _re.search(r"first pass\s*:\s*([\d.]+) km", inv.stdout)
m2 = _re.search(r"route\s*:\s*([\d.]+) km", inv.stdout)
if not (m1 and m2):
    print("FAIL could not read the two distances"); fails.append("invariant-parse")
else:
    a_km, b_km = float(m1.group(1)), float(m2.group(1))
    print(f"\nsolver total {a_km:.1f} km vs traced geometry {b_km:.1f} km")
    if abs(a_km - b_km) > max(0.2, 0.002 * a_km):
        print("FAIL the solver is optimising against the wrong distances")
        fails.append("length-invariant")

print("\n===== summary =====")
print(f"optimum (boustrophedon) : {optimum_km:.2f} km")
for label, data in results.items():
    km = data["stats"]["km"]
    print(f"{label:9s} : {km:7.2f} km   "
          f"({100*(km-optimum_km)/optimum_km:+.1f}% vs optimum, "
          f"{data['stats']['visits']} visits)")

if "fine" in results:
    km = results["fine"]["stats"]["km"]
    if km > optimum_km * 1.30:
        print("FAIL fine-lattice route is more than 30% above the optimum")
        fails.append("fine-quality")
    if results["fine"]["stats"]["covered"] != results["fine"]["stats"]["tiles"]:
        fails.append("fine-coverage")

if "pruned" in results and "no-prune" in results:
    if results["pruned"]["stats"]["km"] > results["no-prune"]["stats"]["km"] + 1e-6:
        print("FAIL pruning made the route longer")
        fails.append("prune")
    gain = (1 - results["pruned"]["stats"]["km"]
            / results["no-prune"]["stats"]["km"]) * 100
    print(f"pruning saved {gain:.1f}%")
    if results["pruned"]["stats"]["km"] > optimum_km * 1.25:
        print("FAIL pruned route is more than 25% above the known optimum")
        fails.append("quality")

print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "route mode 1 OK")
sys.exit(1 if fails else 0)
