"""The two-command flow: route json carries the profile, map renders it."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import json, math, os, shutil, subprocess, sys, tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin

import tilehunt as t

fails = []
ROOT = tempfile.mkdtemp()
DATA = os.path.join(ROOT, "Data"); os.makedirs(DATA)
MAPS = os.path.join(ROOT, "Maps"); os.makedirs(MAPS)
SRC = os.path.join(DATA, "SRC"); os.makedirs(SRC)
DEMB = os.path.join(DATA, "DEM_DB")

# the zone: a 4 x 4 square of explorer tiles near the user's own
X0, Y0, N = 8124, 5656, 4
lon0, lat1 = t.tilef_to_lonlat(X0 - 3, Y0 + N + 3)
lon1, lat0 = t.tilef_to_lonlat(X0 + N + 3, Y0 - 3)


def surf(lon, lat):
    return 180.0 + 120.0 * np.sin(2 * np.pi * lon * 6) * np.cos(2 * np.pi * lat * 6)


la, lo = math.floor(lat1), math.floor(lon0)
n, st = 2401, 1.0 / 2400
ln = lo + np.arange(n) * st
lt = (la + 1) - np.arange(n) * st
with rasterio.open(os.path.join(SRC, t.copernicus_name(la, lo) + ".tif"), "w",
                   driver="GTiff", height=n, width=n, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(lo, la + 1, st, st)) as ds:
    ds.write(surf(ln[None, :], lt[:, None]).astype("float32"), 1)
    ds.update_tags(AREA_OR_POINT="Point")
print(f"source tile {t.copernicus_name(la, lo)} written")

# a dense lattice of two-way streets over the zone, straight into the road
# database, so no PBF is needed to exercise the flow
db = t.DBWriter(os.path.join(DATA, "Database_OSM"))
STEPS = 5                             # lattice nodes per tile edge
tags = {"highway": "residential", "surface": "asphalt"}
lons = np.linspace(lon0, lon1, (N + 6) * STEPS + 1)
lats = np.linspace(lat1, lat0, (N + 6) * STEPS + 1)
nid = 1
ways = 0


def node_id(i, j):
    return 1000 + j * (len(lons) + 1) + i


for j, y in enumerate(lats):                      # every east-west street
    db.add(tags, [(node_id(i, j), float(x), float(y))
                  for i, x in enumerate(lons)])
    ways += 1
for i, x in enumerate(lons):                      # every north-south street
    db.add(tags, [(node_id(i, j), float(x), float(y))
                  for j, y in enumerate(lats)])
    ways += 1
info = db.consolidate({"source": ["synthetic"]}, False)
print(f"road database: {ways} ways, {info['blocks']} blocks")

# the coverage npz the map needs
tiles = [(x, y) for x in range(X0 - 3, X0 + N + 4)
         for y in range(Y0 - 3, Y0 + N + 4)]
t.TileGrid.from_tiles(tiles).save(os.path.join(DATA, "france.paved.npz"))


def run(*args):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, "--maps-dir", MAPS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(args)}")
    return r.stdout


# step 0 is gone on purpose: no elevation database exists yet, and `route`
# must build the blocks it needs by itself
assert not os.path.exists(DEMB), "the test must start with no DEM_DB"

# step 1: the route. No --dem given on purpose: it must find DEM_DB itself.
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
          "--seconds", "5", "--reopt-seconds", "3", "--cands", "3",
          "--dem-src", "SRC",
          "--gpx", "sq.gpx", "--out", "sq.route.json")
if "fetching" not in out:
    print("FAIL route did not build the missing heights itself")
    fails.append("ondemand")
else:
    print("   " + [l for l in out.splitlines() if "fetching" in l][0])
if not os.path.exists(os.path.join(DEMB, "index.json")):
    print("FAIL no database was created"); fails.append("created")
for key in ("elevation", "source", "covered"):
    for line in out.splitlines():
        if line.startswith(key):
            print("   " + line)

rj = json.load(open(os.path.join(DATA, "sq.route.json")))
print(f"route json keys: {sorted(rj)}")
if "profile" not in rj or not rj["profile"]:
    print("FAIL the route json carries no profile"); fails.append("json")
else:
    pr = rj["profile"]
    print(f"profile in json: {len(pr['elev'])} points, +{pr['up']} / "
          f"-{pr['down']} m, source {pr['src']}")
    if pr["covered"] < 0.999:
        print("FAIL the profile has gaps"); fails.append("covered")
    if not pr["src"]:
        print("FAIL the profile does not name its source"); fails.append("src")

if not os.path.exists(os.path.join(DATA, "sq.gpx")):
    print("FAIL no gpx"); fails.append("gpx")
else:
    gpx = open(os.path.join(DATA, "sq.gpx")).read()
    print(f"gpx: {gpx.count('<trkpt')} points, "
          f"{gpx.count('<ele>')} with elevation")
    if gpx.count("<ele>") < gpx.count("<trkpt") * 0.99:
        print("FAIL the gpx is missing elevations"); fails.append("gpxele")

# a second route run must find every block already there
out2 = run("compSquareRoute", "--db", "Database_OSM",
           "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
           "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
           "--seconds", "5", "--reopt-seconds", "3", "--cands", "3",
           "--dem-src", "SRC", "--out", "sq2.route.json")
print(f"second run refetched anything: {'fetching' in out2}")
if "fetching" in out2:
    print("FAIL the heights were fetched twice"); fails.append("refetch")

# with the database complete, a run must say so and open no source at all
out4 = run("compSquareRoute", "--db", "Database_OSM",
           "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
           "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
           "--seconds", "5", "--reopt-seconds", "3", "--cands", "3",
           "--no-dem-fetch", "--out", "off.route.json")
offline = json.load(open(os.path.join(DATA, "off.route.json")))
said = [l for l in out4.splitlines() if "nothing fetched" in l]
print(f"offline run: {said[0].strip() if said else 'NOT ANNOUNCED'}, "
      f"profile +{offline['profile']['up']} m")
if not said or not offline.get("profile"):
    print("FAIL the local-only run did not produce a profile")
    fails.append("offline")

# a height source that cannot be reached must cost the profile, never the route
out3 = run("compSquareRoute", "--db", "Database_OSM",
           "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
           "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
           "--seconds", "5", "--reopt-seconds", "3", "--cands", "3",
           "--dem", "NOPE_DB", "--dem-src", "NOWHERE",
           "--gpx", "bad.gpx", "--out", "bad.route.json")
bad = json.load(open(os.path.join(DATA, "bad.route.json")))
print(f"unreachable heights: route still written "
      f"({bad['stats'].get('km', bad['stats'].get('metres', '?'))}), "
      f"profile {bad.get('profile')}")
if not os.path.exists(os.path.join(DATA, "bad.gpx")) or not bad.get("coords"):
    print("FAIL a height problem destroyed the route"); fails.append("resilient")

# step 2: the map
out = run("buildmap", "--tiles", "france.paved.npz", "--route", "sq.route.json",
          "--out", "sq.html")
html = open(os.path.join(MAPS, "sq.html"), encoding="utf-8").read()
print(f"html: {len(html)/1024:.0f} kB")
if '"profile"' not in html:
    print("FAIL the html does not carry the profile"); fails.append("html")
else:
    print("the html carries the profile block")
if "plan-cmd" not in html:
    print("FAIL the plan panel is gone"); fails.append("panel")

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "end to end OK")
sys.exit(1 if fails else 0)
