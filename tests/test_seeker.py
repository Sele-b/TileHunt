"""Mode 3: collect a chosen set of tiles, loading only what they span."""
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

import tilehunt as t

fails = []
ROOT = tempfile.mkdtemp()
DATA = os.path.join(ROOT, "Data"); os.makedirs(DATA)
MAPS = os.path.join(ROOT, "Maps"); os.makedirs(MAPS)
RUNS = os.path.join(ROOT, "Runs")

# a wide lattice, so the window really can be much smaller than the data
X0, Y0, SPAN = 8110, 5640, 60
lon0, lat1 = t.tilef_to_lonlat(X0, Y0 + SPAN)
lon1, lat0 = t.tilef_to_lonlat(X0 + SPAN, Y0)
db = t.DBWriter(os.path.join(DATA, "Database_OSM"))
tags = {"highway": "residential", "surface": "asphalt"}
# Offset the lattice so no node sits exactly on a tile boundary. A node on
# the line is a coin toss between two tiles, and the json rounds coordinates
# to 6 decimals (~11 cm), which is enough to flip it -- a fixture artefact
# that looks exactly like a coverage bug.
step = (lon1 - lon0) / (SPAN * 3)
lons = lon0 + (np.arange(SPAN * 3) + 0.37) * step
vstep = (lat0 - lat1) / (SPAN * 3)
lats = lat1 + (np.arange(SPAN * 3) + 0.41) * vstep
nid = lambda i, j: 1000 + j * (len(lons) + 1) + i
# Short ways, as real OSM has: a way is stored in every block it touches and
# loaded whole, so a street running the length of the lattice would drag the
# whole lattice in and hide whether the window is doing its job.
CHUNK = 4
for j, y in enumerate(lats):
    row = [(nid(i, j), float(x), float(y)) for i, x in enumerate(lons)]
    for c in range(0, len(row) - 1, CHUNK):
        db.add(tags, row[c:c + CHUNK + 1])
for i, x in enumerate(lons):
    col = [(nid(i, j), float(x), float(y)) for j, y in enumerate(lats)]
    for c in range(0, len(col) - 1, CHUNK):
        db.add(tags, col[c:c + CHUNK + 1])
db.consolidate({"source": ["synthetic"]}, False)
t.TileGrid.from_tiles([(x, y) for x in range(X0, X0 + SPAN + 1)
                       for y in range(Y0, Y0 + SPAN + 1)]
                      ).save(os.path.join(DATA, "france.paved.npz"))
print(f"lattice over {SPAN}x{SPAN} tiles")


def run(*args, expect_fail=False):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, "--maps-dir", MAPS,
                        "--runs-dir", RUNS], capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout[-1800:]); print(r.stderr[-1200:])
        raise SystemExit(f"unexpected exit: {' '.join(args)}")
    return r.stdout + r.stderr


# four scattered tiles, nowhere near forming a rectangle
WANT = [(8124, 5654), (8131, 5657), (8127, 5663), (8133, 5652)]
START, END = (8122, 5652), (8135, 5665)
args = ["compTileSeekerRoute", "--db", "Database_OSM",
        "--start", f"{START[0]},{START[1]}", "--end", f"{END[0]},{END[1]}",
        "--seconds", "5", "--reopt-seconds", "3", "--cands", "3", "--no-dem"]
for w in WANT:
    args += ["--want", f"{w[0]},{w[1]}"]

out = run(*args, "--runname", "chasse")
for l in out.splitlines():
    if l.startswith(("wanted", "window", "graph", "route", "per tile",
                     "coverage", "MISSING")):
        print("   " + l.rstrip())

rj = json.load(open(os.path.join(RUNS, "TileSeeker", "chasse",
                                 "chasse.route.json")))
st = rj["stats"]
print(f"   mode in json: {rj['mode']}, wanted {len(rj.get('wanted') or [])}")

if rj["mode"] != "seeker":
    print("FAIL the json is not marked as a seeker run"); fails.append("mode")
if sorted(map(tuple, rj.get("wanted") or [])) != sorted(WANT):
    print("FAIL the wanted tiles did not reach the json"); fails.append("wanted")
if st["tiles"] != len(WANT) or st["covered"] != len(WANT):
    print(f"FAIL expected {len(WANT)} tiles covered, got "
          f"{st['covered']}/{st['tiles']}"); fails.append("cover")

# the track really does enter every wanted tile
coords = rj["coords"]
hit = set()
for (alon, alat), (blon, blat) in zip(coords, coords[1:]):
    d = t.haversine_m(alon, alat, blon, blat)
    n = max(2, int(d / 5) + 1)            # a sample every 5 m or so
    for k in range(n + 1):
        f = k / n
        tx, ty = t.lonlat_to_tilef(alon + (blon-alon)*f, alat + (blat-alat)*f)
        hit.add((int(tx), int(ty)))
missed = [w for w in WANT if w not in hit]
print(f"   track enters every wanted tile: {not missed}"
      + (f"  missed {missed}" if missed else ""))
if missed:
    fails.append("entered")

# only the bounding box plus the margin is loaded, not the whole lattice
win = [l for l in out.splitlines() if l.startswith("window")][0]
nodes = int([l for l in out.splitlines() if l.startswith("graph")
             ][-1].split(":")[1].split()[0].replace(",", ""))
full = (len(lons) * len(lats))
print(f"   {win.strip()}")
print(f"   loaded {nodes:,} of the lattice's {full:,} nodes "
      f"({100*nodes/full:.0f}%)")
if nodes >= full * 0.3:
    print("FAIL the window did not restrict what was loaded")
    fails.append("window")

# the square-only floor is not claimed for scattered tiles
print(f"   no ideal-floor claim: {'ideal floor' not in out}")
if "ideal floor" in out or "vs ideal" in out:
    print("FAIL a square's floor was quoted for scattered tiles")
    fails.append("floor")

# the run lands in its own kind of folder
got = sorted(os.listdir(os.path.join(RUNS, "TileSeeker", "chasse")))
print(f"   Runs/TileSeeker/chasse/ holds: {got}")
if got != ["chasse.gpx", "chasse.route.json"]:
    fails.append("runs")

# the map finds it by name and draws the wanted tiles
run("buildmap", "--tiles", "france.paved.npz", "--runname", "chasse")
run("buildmap", "--tiles", "france.paved.npz", "--out", "plain.html")
html = open(os.path.join(RUNS, "TileSeeker", "chasse", "chasse.html"),
            encoding="utf-8").read()
carried = '"wanted"' in html
print(f"   map built beside the route, wanted tiles carried: {carried}")
if not carried:
    fails.append("page")
node = shutil.which("node") or shutil.which("nodejs")
if node:
    r = subprocess.run([node, os.path.join(HERE, "test_draw.mjs"),
                        os.path.join(RUNS, "TileSeeker", "chasse",
                                     "chasse.html")],
                       capture_output=True, text=True)
    for line in (r.stdout or r.stderr).strip().splitlines():
        print("   " + line)
    if r.returncode != 0:
        fails.append("drawing")

# ---- a wanted tile with no road is skipped, the hunt goes on ------------- #
# punch a hole where one of the wanted tiles is, rebuild, and ask for it anyway
HOLE = (8129, 5659)
hw, hs, he, hn = t.tile_bounds(*HOLE)
db2 = t.DBWriter(os.path.join(DATA, "DB_hole"))


def outside_hole(pts):
    run_, out_ = [], []
    for q in pts:
        if hw <= q[1] <= he and hs <= q[2] <= hn:
            if len(run_) > 1:
                out_.append(run_)
            run_ = []
        else:
            run_.append(q)
    if len(run_) > 1:
        out_.append(run_)
    return out_


for j, y in enumerate(lats):
    row = [(nid(i, j), float(x), float(y)) for i, x in enumerate(lons)]
    for c in range(0, len(row) - 1, CHUNK):
        for seg in outside_hole(row[c:c + CHUNK + 1]):
            db2.add(tags, seg)
for i, x in enumerate(lons):
    col = [(nid(i, j), float(x), float(y)) for j, y in enumerate(lats)]
    for c in range(0, len(col) - 1, CHUNK):
        for seg in outside_hole(col[c:c + CHUNK + 1]):
            db2.add(tags, seg)
db2.consolidate({"source": ["synthetic"]}, False)

args2 = ["compTileSeekerRoute", "--db", "DB_hole",
         "--start", f"{START[0]},{START[1]}", "--end", f"{END[0]},{END[1]}",
         "--seconds", "4", "--reopt-seconds", "2", "--cands", "3", "--no-dem"]
for w in WANT + [HOLE]:
    args2 += ["--want", f"{w[0]},{w[1]}"]
out2 = run(*args2, "--out", "hole.route.json")
for l in out2.splitlines():
    if l.startswith(("WARNING", "              ")) or "no road" in l:
        print("   " + l.rstrip())
hj = json.load(open(os.path.join(DATA, "hole.route.json")))
hst = hj["stats"]
print(f"   asked for {len(WANT)+1}, required {hst['tiles']}, "
      f"covered {hst['covered']}, skipped {hj.get('skipped')}")
if "WARNING" not in out2:
    print("FAIL a red wanted tile was not reported"); fails.append("red-warn")
if hst["tiles"] != len(WANT) or hst["covered"] != len(WANT):
    print("FAIL the red tile was not dropped cleanly"); fails.append("red-drop")
if hj.get("skipped") != [list(HOLE)]:
    print("FAIL the skipped tile is not named"); fails.append("red-name")

# the planner panel itself: clicks driven through the page's real handlers
if node:
    r = subprocess.run([node, os.path.join(HERE, "test_plan.mjs"),
                        os.path.join(MAPS, "plain.html")],
                       capture_output=True, text=True)
    for line in (r.stdout or r.stderr).strip().splitlines():
        print("   " + line)
    if r.returncode != 0:
        fails.append("planner")

# square mode is untouched -- and its coverage is checked against the written
# geometry too, not only against what the solver reports about itself

out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", "8124,5654", "8127,5657",
          "--seconds", "3", "--cands", "3", "--no-dem", "--out", "sq.json")
sj = json.load(open(os.path.join(DATA, "sq.json")))
sq_want = [(x, y) for x in range(8124, 8128) for y in range(5654, 5658)]
sq_hit = set()
sc = sj["coords"]
for (al, aa), (bl, ba) in zip(sc, sc[1:]):
    d = t.haversine_m(al, aa, bl, ba)
    n = max(2, int(d / 5) + 1)
    for k in range(n + 1):
        f = k / n
        tx, ty = t.lonlat_to_tilef(al + (bl - al) * f, aa + (ba - aa) * f)
        sq_hit.add((int(tx), int(ty)))
sq_missed = [w for w in sq_want if w not in sq_hit]
print(f"   square mode still works: {sj['mode']}, {sj['stats']['covered']}/"
      f"{sj['stats']['tiles']}, floor quoted: {'ideal floor' in out}, "
      f"geometry misses {len(sq_missed)}")
if sj["mode"] != "square" or "ideal floor" not in out:
    fails.append("square")
if sq_missed:
    print(f"FAIL the square's own geometry misses {sq_missed}")
    fails.append("square-geometry")

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "tile seeker OK")
sys.exit(1 if fails else 0)
