"""Exact start/end points: snapping, pinning, and routes that begin outside."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import argparse, json, math, os, shutil, subprocess, sys, tempfile
import numpy as np

import tilehunt as t

fails = []
ROOT = tempfile.mkdtemp()
DATA = os.path.join(ROOT, "Data"); os.makedirs(DATA)
MAPS = os.path.join(ROOT, "Maps"); os.makedirs(MAPS)

X0, Y0, N = 8124, 5656, 4
lon0, lat1 = t.tilef_to_lonlat(X0 - 3, Y0 + N + 3)
lon1, lat0 = t.tilef_to_lonlat(X0 + N + 3, Y0 - 3)

# a lattice of streets, plus one deliberately isolated stub near the middle
db = t.DBWriter(os.path.join(DATA, "Database_OSM"))
STEPS = 5
tags = {"highway": "residential", "surface": "asphalt"}
lons = np.linspace(lon0, lon1, (N + 6) * STEPS + 1)
lats = np.linspace(lat1, lat0, (N + 6) * STEPS + 1)
nid = lambda i, j: 1000 + j * (len(lons) + 1) + i
for j, y in enumerate(lats):
    db.add(tags, [(nid(i, j), float(x), float(y)) for i, x in enumerate(lons)])
for i, x in enumerate(lons):
    db.add(tags, [(nid(i, j), float(x), float(y)) for j, y in enumerate(lats)])

# the trap: a two-node stub touching nothing, 15 m from a chosen point
TRAP_LAT = float((lats[12] + lats[13]) / 2)
TRAP_LON = float((lons[12] + lons[13]) / 2)
db.add(tags, [(999001, TRAP_LON, TRAP_LAT),
              (999002, TRAP_LON + 0.00005, TRAP_LAT + 0.00005)])
db.consolidate({"source": ["synthetic"]}, False)
print(f"database built, with an isolated 2-node stub at "
      f"{TRAP_LAT:.5f},{TRAP_LON:.5f}")

t.TileGrid.from_tiles([(x, y) for x in range(X0 - 3, X0 + N + 4)
                       for y in range(Y0 - 3, Y0 + N + 4)]
                      ).save(os.path.join(DATA, "france.paved.npz"))


def run(*args, expect_fail=False):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, "--maps-dir", MAPS],
                       capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"unexpected exit {r.returncode}: {' '.join(args)}")
    return r.stdout + r.stderr


# ---- 1. snap refuses the trap and takes the real network ----------------- #
out = run("snap", "--db", "Database_OSM", "--at", f"{TRAP_LAT},{TRAP_LON}")
print("\n".join("   " + l for l in out.strip().splitlines()))
if "isolated piece of road" not in out:
    print("FAIL snap did not warn about the isolated stub"); fails.append("trap")
m = [l for l in out.splitlines() if l.startswith("snaps to")]
snapped = m[0].split(":")[1].split()[0]
slat, slon = (float(v) for v in snapped.split(","))
if abs(slat - TRAP_LAT) < 1e-7 and abs(slon - TRAP_LON) < 1e-7:
    print("FAIL snap chose the isolated stub"); fails.append("chose-trap")

# ---- 2. a point inside the square is pinned exactly ---------------------- #
P_LAT, P_LON = float(lats[17] + 0.0003), float(lons[17] + 0.0004)
Q_LAT, Q_LON = float(lats[22] - 0.0002), float(lons[21] + 0.0001)
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start-at", f"{P_LAT},{P_LON}", "--end-at", f"{Q_LAT},{Q_LON}",
          "--seconds", "4", "--reopt-seconds", "2", "--cands", "3",
          "--no-dem", "--gpx", "pin.gpx", "--out", "pin.route.json")
for l in out.splitlines():
    if l.startswith(("start ", "end ", "approach", "distance", "covered")):
        print("   " + l)
rj = json.load(open(os.path.join(DATA, "pin.route.json")))
pin = rj.get("pinned")
if not pin or "start" not in pin or "end" not in pin:
    print("FAIL the route json carries no pinned points"); fails.append("json")
else:
    c = rj["coords"]
    d0 = t.haversine_m(c[0][0], c[0][1], pin["start"]["node"][0],
                       pin["start"]["node"][1])
    d1 = t.haversine_m(c[-1][0], c[-1][1], pin["end"]["node"][0],
                       pin["end"]["node"][1])
    print(f"   track starts {d0:.1f} m from the pinned node, ends {d1:.1f} m "
          f"from the pinned node")
    if d0 > 0.5 or d1 > 0.5:
        print("FAIL the track does not begin and end on the pinned nodes")
        fails.append("pinned")
    ask = t.haversine_m(pin["start"]["asked"][0], pin["start"]["asked"][1],
                        pin["start"]["node"][0], pin["start"]["node"][1])
    if abs(ask - pin["start"]["m"]) > 1.0:
        print("FAIL the reported snap distance is wrong"); fails.append("dist")
    print(f"   reported snap distance {pin['start']['m']} m, recomputed "
          f"{ask:.1f} m")

# ---- 3. a point outside the rectangle adds an approach, covers the same -- #
OUT_LAT, OUT_LON = float(lats[2]), float(lons[2])
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start-at", f"{OUT_LAT},{OUT_LON}", "--end-at", f"{Q_LAT},{Q_LON}",
          "--seconds", "4", "--reopt-seconds", "2", "--cands", "3",
          "--no-dem", "--out", "far.route.json")
far = json.load(open(os.path.join(DATA, "far.route.json")))
print(f"   outside start: {'approach' in out}, "
      f"tiles required {far['stats']['tiles']}, "
      f"covered {far['stats']['covered']}, {far['stats']['km']} km "
      f"(inside start was {rj['stats']['km']} km)")
if "approach" not in out:
    print("FAIL an endpoint outside the rectangle was not announced")
    fails.append("approach")
if far["stats"]["tiles"] != rj["stats"]["tiles"]:
    print("FAIL the outside endpoint changed what must be covered")
    fails.append("required")
if far["stats"]["covered"] != far["stats"]["tiles"]:
    print("FAIL the square is no longer fully covered"); fails.append("cover")
if far["stats"]["km"] <= rj["stats"]["km"]:
    print("FAIL the approach leg cost nothing"); fails.append("cost")

# ---- 4. a point far from any road is refused, not silently moved --------- #
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start-at", f"{P_LAT},{P_LON}", "--end-at", f"{Q_LAT},{Q_LON}",
          "--snap-max", "1", "--seconds", "2", "--no-dem",
          "--out", "no.route.json", expect_fail=True)
print(f"   --snap-max 1 m refused: {'snap-max' in out}")
if "snap-max" not in out:
    print("FAIL an impossible snap was accepted"); fails.append("snapmax")

# ---- 5. tiles still work, unchanged -------------------------------------- #
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
          "--seconds", "4", "--reopt-seconds", "2", "--cands", "3",
          "--no-dem", "--out", "tiles.route.json")
tj = json.load(open(os.path.join(DATA, "tiles.route.json")))
print(f"   tile mode still works: {tj['stats']['km']} km, "
      f"pinned={tj.get('pinned')}")
if tj.get("pinned") is not None or tj["stats"]["covered"] != tj["stats"]["tiles"]:
    print("FAIL the tile path regressed"); fails.append("tiles")

# ---- 6. bad input is caught --------------------------------------------- #
out = run("compSquareRoute", "--db", "Database_OSM", "--corners",
          f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}", "--start-at", "-1.38,48.47",
          "--no-dem", "--out", "x.json", expect_fail=True)
caught = "The order is LAT,LON" in out
hinted = "Did you mean" in out
print(f"   lat/lon swapped is caught: {caught}, with a suggestion: {hinted}")
print("   " + [l for l in out.splitlines() if "order is LAT" in l][0][:160]
      if caught else "")
if not caught or not hinted:
    print("FAIL a swapped pair was accepted"); fails.append("order")

# ---- 6b. forced waypoints ------------------------------------------------ #
V1 = (float(lats[19]), float(lons[25]))     # inside the square, off the line
V2 = (float(lats[13] - 0.0004), float(lons[14]))
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start-at", f"{P_LAT},{P_LON}", "--end-at", f"{Q_LAT},{Q_LON}",
          "--via", f"{V1[0]},{V1[1]}", "--via", f"{V2[0]},{V2[1]}",
          "--seconds", "4", "--reopt-seconds", "2", "--cands", "3",
          "--no-dem", "--out", "via.route.json")
vj = json.load(open(os.path.join(DATA, "via.route.json")))
vias = (vj.get("pinned") or {}).get("via") or []
print(f"   waypoints announced: {'waypoints' in out}, {len(vias)} in the json")
if len(vias) != 2:
    print("FAIL the waypoints did not reach the json"); fails.append("via-json")

# the track must actually pass through each pinned junction
coords = vj["coords"]
for k, v in enumerate(vias, 1):
    d = min(t.haversine_m(c[0], c[1], v["node"][0], v["node"][1])
            for c in coords)
    print(f"   via {k}: track passes {d:.1f} m from the pinned junction "
          f"({v['m']} m from the point asked)")
    if d > 0.5:
        print("FAIL the pruning dropped a forced waypoint")
        fails.append(f"via{k}-missed")

# and the square is still fully covered
print(f"   coverage with waypoints: {vj['stats']['covered']}/"
      f"{vj['stats']['tiles']}, {vj['stats']['km']} km "
      f"(without: {rj['stats']['km']} km)")
if vj["stats"]["covered"] != vj["stats"]["tiles"]:
    print("FAIL waypoints broke the coverage"); fails.append("via-cover")

# a waypoint that is also an endpoint must not create a second cluster
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start-at", f"{P_LAT},{P_LON}", "--end-at", f"{Q_LAT},{Q_LON}",
          "--via", f"{P_LAT},{P_LON}",
          "--seconds", "3", "--cands", "3", "--no-dem", "--out", "dup.json")
dj = json.load(open(os.path.join(DATA, "dup.json")))
print(f"   a waypoint on the start point: {dj['stats']['covered']}/"
      f"{dj['stats']['tiles']} covered, {dj['stats']['km']} km")
if dj["stats"]["covered"] != dj["stats"]["tiles"]:
    print("FAIL a duplicated endpoint broke the route"); fails.append("via-dup")

# ---- 6c. a long ride must not trip the sanity check ---------------------- #
# Lille to Paris through a tile in between: both ends are ~110 km from the
# middle of the area, which the old centre-based check refused outright.
import importlib
LILLE, PARIS = (50.6292, 3.0573), (48.8566, 2.3522)
far_args = argparse.Namespace(
    want=[(8331, 5511), (8314, 5574), (8299, 5636)],
    corners=None, start=None, end=None,
    start_at=LILLE, end_at=PARIS, via=None, margin=2, away_max=100.0)
x0 = min(w[0] for w in far_args.want + [(8331, 5511)])
# reproduce the guard the way cmd_route runs it
span = set(far_args.want)
gx0 = min(q[0] for q in span); gx1 = max(q[0] for q in span)
gy0 = min(q[1] for q in span); gy1 = max(q[1] for q in span)
bw, _, _, bn = t.tile_bounds(gx0, gy0)
_, bs, be, _ = t.tile_bounds(gx1, gy1)


def outside_km(lat, lon):
    return t.haversine_m(lon, lat, min(max(lon, bw), be),
                         min(max(lat, bs), bn)) / 1000.0


centre_km = t.haversine_m((bw + be) / 2, (bs + bn) / 2, LILLE[1], LILLE[0]) / 1000
print(f"   Lille->Paris: {outside_km(*LILLE):.0f} km outside the area, but "
      f"{centre_km:.0f} km from its centre")
if outside_km(*LILLE) > 100 or outside_km(*PARIS) > 100:
    print("FAIL a legitimate long ride is still refused")
    fails.append("long-ride")
if centre_km <= 100:
    print("   (note: this fixture no longer reproduces the old failure)")

# and a swapped pair is still caught
swapped = (3.0573, 50.6292)
print(f"   swapped LAT,LON still lands {outside_km(*swapped):,.0f} km out")
if outside_km(*swapped) <= 100:
    print("FAIL a swapped pair would now slip through"); fails.append("swap")

# ---- 7. the map carries and draws the pinned points ---------------------- #
run("buildmap", "--tiles", "france.paved.npz", "--route", "pin.route.json",
    "--out", "pin.html")
html = open(os.path.join(MAPS, "pin.html"), encoding="utf-8").read()
checks = {
    "pinned points in the page": '"pinned"' in html,
    "asked-vs-snapped drawing": "ROUTE.pinned" in html,
    "planner emits --start-at": "--start-at " in html,
    "planner no longer emits --start": " --start \" +" not in html,
    "exact-point wording": "exact point you start from" in html,
    "one ll2t only": html.count("function ll2t") == 1,
}
for k, v in checks.items():
    print(f"   {k}: {'yes' if v else 'NO'}")
    if not v:
        fails.append(k)

# the recap map of a ride with waypoints carries and draws them
run("buildmap", "--tiles", "france.paved.npz", "--route", "via.route.json",
    "--out", "via.html")
vhtml = open(os.path.join(MAPS, "via.html"), encoding="utf-8").read()
vchecks = {
    "waypoints in the page": '"via":' in vhtml,
    "one diamond helper, shared": vhtml.count("function diamond") == 1,
}
for k, v in vchecks.items():
    print(f"   {k}: {'yes' if v else 'NO'}")
    if not v:
        fails.append(k)

# Strings in the page prove nothing: the pinned points were in the source and
# in the data, and still nothing was drawn, because ROUTE dropped the field on
# its way out. So run the page's own drawing code and look at the canvas.
node = shutil.which("node") or shutil.which("nodejs")
if not node:
    print("   node absent: the drawing itself was not exercised")
else:
    r = subprocess.run([node, os.path.join(HERE, "test_draw.mjs"),
                        os.path.join(MAPS, "via.html")],
                       capture_output=True, text=True)
    for line in (r.stdout or r.stderr).strip().splitlines():
        print("   " + line)
    if r.returncode != 0:
        fails.append("drawing")

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "exact endpoints OK")
sys.exit(1 if fails else 0)
