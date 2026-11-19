"""--run: one folder per ride, every output sharing its name."""
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
RUNS = os.path.join(ROOT, "Runs")

X0, Y0, N = 8124, 5656, 4
lon0, lat1 = t.tilef_to_lonlat(X0 - 3, Y0 + N + 3)
lon1, lat0 = t.tilef_to_lonlat(X0 + N + 3, Y0 - 3)
db = t.DBWriter(os.path.join(DATA, "Database_OSM"))
tags = {"highway": "residential", "surface": "asphalt"}
lons = np.linspace(lon0, lon1, (N + 6) * 5 + 1)
lats = np.linspace(lat1, lat0, (N + 6) * 5 + 1)
nid = lambda i, j: 1000 + j * (len(lons) + 1) + i
for j, y in enumerate(lats):
    db.add(tags, [(nid(i, j), float(x), float(y)) for i, x in enumerate(lons)])
for i, x in enumerate(lons):
    db.add(tags, [(nid(i, j), float(x), float(y)) for j, y in enumerate(lats)])
db.consolidate({"source": ["synthetic"]}, False)
t.TileGrid.from_tiles([(x, y) for x in range(X0 - 3, X0 + N + 4)
                       for y in range(Y0 - 3, Y0 + N + 4)]
                      ).save(os.path.join(DATA, "france.paved.npz"))


def run(*args, expect_fail=False):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, "--maps-dir", MAPS,
                        "--runs-dir", RUNS], capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"unexpected exit: {' '.join(args)}")
    return r.stdout + r.stderr


NAME = "bretagne-nord"
out = run("compSquareRoute", "--db", "Database_OSM",
          "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
          "--start", f"{X0},{Y0}", "--end", f"{X0+N-1},{Y0+N-1}",
          "--seconds", "4", "--reopt-seconds", "2", "--cands", "3",
          "--no-dem", "--runname", NAME)
run("buildmap", "--tiles", "france.paved.npz", "--clip-to-mask", "--runname", NAME)

folder = os.path.join(RUNS, "Square", NAME)
got = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
print(f"Runs/Square/{NAME}/ holds: {got}")
want = [f"{NAME}.gpx", f"{NAME}.html", f"{NAME}.route.json"]
if got != want:
    print(f"FAIL expected {want}"); fails.append("layout")

# nothing leaked into Data/ or Maps/
stray = [f for f in os.listdir(DATA) if NAME in f] \
      + [f for f in os.listdir(MAPS) if NAME in f]
print(f"stray files outside the run folder: {stray or 'none'}")
if stray:
    fails.append("stray")

# the track inside the gpx carries the name too
gpx = open(os.path.join(folder, f"{NAME}.gpx"), encoding="utf-8").read()
print(f"gpx track named after the run: {('<name>' + NAME) in gpx}")
if ("<name>" + NAME) not in gpx:
    fails.append("trackname")

# the map found the route by name alone
html = open(os.path.join(folder, f"{NAME}.html"), encoding="utf-8").read()
rj = json.load(open(os.path.join(folder, f"{NAME}.route.json")))
print(f"map carries the route: {'\"d\":' in html}, {rj['stats']['km']} km")
if '"d":' not in html:
    fails.append("route-in-map")

# profile by name alone, no paths at all
out = run("profile", "--runname", NAME, "--dem", "NOPE", expect_fail=True)
print(f"profile --runname resolved the json: {'no heights available' in out}")
if "give --route" in out:
    print("FAIL --run did not supply the route"); fails.append("profile-run")

# an explicit path still wins over --run
run("compSquareRoute", "--db", "Database_OSM",
    "--corners", f"{X0},{Y0}", f"{X0+N-1},{Y0+N-1}",
    "--seconds", "2", "--cands", "3", "--no-dem",
    "--runname", "override", "--out", "elsewhere.route.json")
print(f"explicit --out wins: {os.path.exists(os.path.join(DATA, 'elsewhere.route.json'))}, "
      f"gpx still in the run folder: "
      f"{os.path.exists(os.path.join(RUNS, 'Square', 'override', 'override.gpx'))}")
if not os.path.exists(os.path.join(DATA, "elsewhere.route.json")):
    fails.append("override")

# a hostile name cannot escape the Runs folder
for bad, clean in [("../../etc/passwd", "etc-passwd"),
                   ("mon tracé: été 2026!", "mon-trace-ete-2026"),
                   ("Forêt d'Ânes", "Foret-d-Anes"),
                   ("  spaces  ", "spaces")]:
    got = t.run_name(bad)
    print(f"  {bad!r:<22} -> {got!r}")
    if os.sep in got or ".." in got:
        print("FAIL a name escaped the folder"); fails.append("escape")
    if got != clean:
        print(f"FAIL expected {clean!r}"); fails.append("clean")
try:
    t.run_name("///")
    print("FAIL an empty name was accepted"); fails.append("empty")
except SystemExit:
    print("  a name with nothing usable is refused")

# the page offers the field and emits --run
run("buildmap", "--tiles", "france.paved.npz", "--out", "plain.html")
page = open(os.path.join(MAPS, "plain.html"), encoding="utf-8").read()
for probe, label in [('id="run-name"', "name field"),
                     ("safeRunName", "same cleaning in the page"),
                     ('"TileSeeker" : "Square"',
                      "the hint names the kind's sub-folder"),
                     (' --runname " + name', "commands use --runname"),
                     ("Name the ride", "the fourth step")]:
    ok = probe in page
    print(f"   {label}: {'yes' if ok else 'NO'}")
    if not ok:
        fails.append(label)

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "named runs OK")
sys.exit(1 if fails else 0)
