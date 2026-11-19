#!/usr/bin/env python3
"""
Regression test for the road database, aimed squarely at shard boundaries.

Ways are written to coarse shards while building, then split into fine blocks.
A way straddling a shard boundary lands in both shards, so when the second
shard is processed it must NOT rewrite the neighbour's blocks from its own
partial view of them -- that silently wiped out whole blocks, which showed up
as "60 tiles have no road in the database" over perfectly ordinary countryside.

The lattice below is made of SHORT ways, two nodes each, which is what real OSM
data looks like and what the long-chain test failed to catch.
"""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import os
import shutil
import subprocess
import sys

import numpy as np

import osmium
from Tilehunt import (BLOCK_ZOOM, SHARD_ZOOM, ZOOM, tilef_to_lonlat,
                      load_db_graph, tile_bounds)

OUT = os.path.join(tempfile.gettempdir(), "tilehunt-dbtest2")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
PBF = os.path.join(OUT, "grid.osm.pbf")
DB = os.path.join(OUT, "db")

SHARD_SPAN = 1 << (ZOOM - SHARD_ZOOM)          # tiles per shard, 64 at z8
BOUNDARY = (8320 // SHARD_SPAN) * SHARD_SPAN   # a real shard boundary
X0, Y0 = BOUNDARY - 20, 5500
W, H = 40, 24                                  # straddles the boundary

nodes, ways = [], []
nid = wid = 0


def N(lo, la):
    global nid
    nid += 1
    nodes.append((nid, lo, la))
    return nid


def W_(tags, ids):
    global wid
    wid += 1
    ways.append((wid, tags, ids))


pt = {}
for ty in range(Y0, Y0 + H + 1):
    for tx in range(X0, X0 + W + 1):
        lo, la = tilef_to_lonlat(tx + 0.5, ty + 0.5)
        pt[(tx, ty)] = N(lo, la)

# short ways: one per lattice edge, exactly as OSM splits real streets
for ty in range(Y0, Y0 + H + 1):
    for tx in range(X0, X0 + W):
        W_({"highway": "residential"}, [pt[(tx, ty)], pt[(tx + 1, ty)]])
for tx in range(X0, X0 + W + 1):
    for ty in range(Y0, Y0 + H):
        W_({"highway": "residential"}, [pt[(tx, ty)], pt[(tx, ty + 1)]])

w = osmium.SimpleWriter(PBF)
for i, lo, la in nodes:
    w.add_node(osmium.osm.mutable.Node(id=i, location=(lo, la), version=1))
for i, t, nd in ways:
    w.add_way(osmium.osm.mutable.Way(id=i, nodes=nd, tags=t, version=1))
w.close()
print(f"lattice: {len(nodes):,} nodes, {len(ways):,} short ways, "
      f"tiles x {X0}..{X0+W} y {Y0}..{Y0+H}")
print(f"shard boundary at x = {BOUNDARY} sits inside the area")

res = subprocess.run(
    [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "tiles",
     "--pbf", PBF, "--db", DB, "--out", os.path.join(OUT, "t.npz")],
    capture_output=True, text=True)
print(res.stdout.strip())
if res.returncode != 0:
    print(res.stderr[-1500:])
    sys.exit(1)

fails = []

import json
with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
    info = json.load(fh)
on_disk = len([f for f in os.listdir(DB)
               if f.startswith("b_") and f.endswith(".npz")])
print(f"\nindex says {info['blocks']} blocks, {on_disk} files on disk")
if on_disk != info["blocks"]:
    print("FAIL a block was written more than once, so one version was lost")
    fails.append("block-count")
if os.path.isdir(os.path.join(DB, "_shards")):
    print("FAIL temporary shards were left behind")
    fails.append("shards-left")

w_, _, _, n_ = tile_bounds(X0, Y0)
_, s_, e_, _ = tile_bounds(X0 + W, Y0 + H)
g = load_db_graph(DB, (w_, s_, e_, n_), "paved-cycle", True, False, False)
nt = g.tiles_of_nodes()
have = set(map(tuple, nt.tolist()))
want = [(x, y) for x in range(X0, X0 + W + 1) for y in range(Y0, Y0 + H + 1)]
missing = [t for t in want if t not in have]
print(f"graph {g.n:,} nodes; {len(want)} tiles expected, {len(missing)} empty")
if missing:
    xs = sorted({m[0] for m in missing})
    print(f"FAIL empty tiles in x {xs[0]}..{xs[-1]} ({len(xs)} columns)")
    print(f"     first few: {missing[:8]}")
    fails.append("empty-tiles")

# every node of the lattice must be present exactly once
if g.n != len(nodes):
    print(f"FAIL expected {len(nodes):,} nodes, database gave {g.n:,}")
    fails.append("node-count")

# and the graph must be connected
from scipy.sparse.csgraph import connected_components
ncomp, _ = connected_components(g.csr, directed=False)
print(f"connected components: {ncomp}")
if ncomp != 1:
    print("FAIL the lattice came back in pieces")
    fails.append("fragmented")

print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "database OK")
sys.exit(1 if fails else 0)
