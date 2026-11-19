"""A square holding tiles with no usable road: warn, skip, keep going."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import json, os, shutil, subprocess, sys, tempfile
import numpy as np

import Tilehunt as t

fails = []
ROOT = tempfile.mkdtemp()
DATA = os.path.join(ROOT, "Data"); os.makedirs(DATA)
MAPS = os.path.join(ROOT, "Maps"); os.makedirs(MAPS)

X0, Y0, N = 8124, 5656, 4
lon0, lat1 = t.tilef_to_lonlat(X0 - 3, Y0 + N + 3)
lon1, lat0 = t.tilef_to_lonlat(X0 + N + 3, Y0 - 3)

# a lattice of streets with a hole punched in it: one tile of the square gets
# no road at all, the way a lake or a military zone leaves one
HOLE = (X0 + 2, Y0 + 1)
hw, hs, he, hn = t.tile_bounds(*HOLE)
db = t.DBWriter(os.path.join(DATA, "Database_OSM"))
tags = {"highway": "residential", "surface": "asphalt"}
lons = np.linspace(lon0, lon1, (N + 6) * 5 + 1)
lats = np.linspace(lat1, lat0, (N + 6) * 5 + 1)
nid = lambda i, j: 1000 + j * (len(lons) + 1) + i


def split(pts):
    """Break a street wherever it would enter the hole."""
    run, out = [], []
    for p in pts:
        if hw <= p[1] <= he and hs <= p[2] <= hn:
            if len(run) > 1:
                out.append(run)
            run = []
        else:
            run.append(p)
    if len(run) > 1:
        out.append(run)
    return out


for j, y in enumerate(lats):
    for seg in split([(nid(i, j), float(x), float(y))
                      for i, x in enumerate(lons)]):
        db.add(tags, seg)
for i, x in enumerate(lons):
    for seg in split([(nid(i, j), float(x), float(y))
                      for j, y in enumerate(lats)]):
        db.add(tags, seg)
db.consolidate({"source": ["synthetic"]}, False)
print(f"database built with no road at all in tile {HOLE[0]},{HOLE[1]}")

t.TileGrid.from_tiles([(x, y) for x in range(X0 - 3, X0 + N + 4)
                       for y in range(Y0 - 3, Y0 + N + 4)]
                      ).save(os.path.join(DATA, "france.paved.npz"))


def run(*args, expect_fail=False):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, "--maps-dir", MAPS],
                       capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout[-1800:]); print(r.stderr[-1200:])
        raise SystemExit(f"unexpected exit: {' '.join(args)}")
    return r.stdout + r.stderr


base = ["compSquareRoute", "--db", "Database_OSM",
        "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
        "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
        "--seconds", "4", "--reopt-seconds", "2", "--cands", "3", "--no-dem"]

# ---- 1. the default is to warn and carry on ------------------------------ #
out = run(*base, "--out", "gap.route.json")
for l in out.splitlines():
    if l.startswith(("WARNING", "              ")) or "no road" in l:
        print("   " + l.rstrip())
gj = json.load(open(os.path.join(DATA, "gap.route.json")))
st = gj["stats"]
print(f"   stats: {st['covered']}/{st['tiles']} covered, {st['km']} km")
if "WARNING" not in out:
    print("FAIL no warning was printed"); fails.append("warn")
if st["tiles"] != N * N - 1:
    print(f"FAIL expected {N*N-1} tiles required"); fails.append("required")
if st["covered"] != st["tiles"]:
    print("FAIL the rest of the square was not covered"); fails.append("cover")
if gj.get("skipped") != [list(HOLE)]:
    print(f"FAIL skipped list wrong: {gj.get('skipped')}"); fails.append("list")

# ---- 2. --require-all restores the old refusal --------------------------- #
out = run(*base, "--require-all", "--out", "x.json", expect_fail=True)
print(f"   --require-all still refuses: {'--require-all was given' in out}")
if "--require-all was given" not in out:
    print("FAIL --require-all did not stop"); fails.append("strict")

# ---- 3. the old flag still runs ------------------------------------------ #
run(*base, "--allow-gaps", "--out", "old.route.json")
print("   an old command line with --allow-gaps still runs")

# ---- 4. the map shows where the hole is ---------------------------------- #
run("buildmap", "--tiles", "france.paved.npz", "--route", "gap.route.json",
    "--out", "gap.html")
html = open(os.path.join(MAPS, "gap.html"), encoding="utf-8").read()
carried = '"skipped"' in html
print(f"   the skipped tiles reach the page: {carried}")
if not carried:
    fails.append("page")

node = shutil.which("node") or shutil.which("nodejs")
if not node:
    print("   node absent: the drawing itself was not exercised")
else:
    r = subprocess.run([node, os.path.join(HERE, "test_draw.mjs"),
                        os.path.join(MAPS, "gap.html")],
                       capture_output=True, text=True)
    for line in (r.stdout or r.stderr).strip().splitlines():
        print("   " + line)
    if r.returncode != 0:
        fails.append("drawing")

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "gaps OK")
sys.exit(1 if fails else 0)
