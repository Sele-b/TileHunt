#!/usr/bin/env python3
"""End-to-end test of the .osm.pbf code path using a synthetic extract."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import math
import os
import shutil
import subprocess
import sys

import osmium
from Tilehunt import TileGrid, lonlat_to_tilef, tilef_to_lonlat

OUT = os.path.join(tempfile.gettempdir(), "tilehunt-pbftest")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
PBF = os.path.join(OUT, "synthetic.osm.pbf")

# Work near Orleans so the numbers are France-like.
BASE_X, BASE_Y = 8320, 5700


def tile_centre(tx, ty):
    lon, lat = tilef_to_lonlat(tx + 0.5, ty + 0.5)
    return lon, lat


nodes = []
ways = []
nid = itid = 0


def N(lon, lat):
    global nid
    nid += 1
    nodes.append((nid, lon, lat))
    return nid


def W(tags, pts):
    global itid
    itid += 1
    ways.append((itid, tags, [N(lo, la) for lo, la in pts]))


# 1. residential road running east across 4 tiles  -> 4 green tiles
a = tile_centre(BASE_X, BASE_Y)
b = tile_centre(BASE_X + 3, BASE_Y)
W({"highway": "residential", "name": "Rue Test"}, [a, b])
EXPECT = {(BASE_X + i, BASE_Y) for i in range(4)}

# 2. motorway crossing a fresh row -> must be ignored
c = tile_centre(BASE_X, BASE_Y + 5)
d = tile_centre(BASE_X + 3, BASE_Y + 5)
W({"highway": "motorway", "ref": "A10"}, [c, d])

# 3. unpaved track -> ignored
e = tile_centre(BASE_X, BASE_Y + 6)
f = tile_centre(BASE_X + 3, BASE_Y + 6)
W({"highway": "track", "surface": "dirt"}, [e, f])

# 4. asphalt track -> counted
g = tile_centre(BASE_X, BASE_Y + 7)
h = tile_centre(BASE_X + 1, BASE_Y + 7)
W({"highway": "track", "surface": "asphalt"}, [g, h])
EXPECT |= {(BASE_X, BASE_Y + 7), (BASE_X + 1, BASE_Y + 7)}

# 5. cycleway diagonal across 2x2 -> counted (and corner cells must appear)
i0 = tilef_to_lonlat(BASE_X + 10 + 0.5, BASE_Y + 10 + 0.5)
i1 = tilef_to_lonlat(BASE_X + 12 + 0.5, BASE_Y + 12 + 0.5)
W({"highway": "cycleway"}, [i0, i1])
EXPECT |= {(BASE_X + 10, BASE_Y + 10), (BASE_X + 11, BASE_Y + 11),
           (BASE_X + 12, BASE_Y + 12)}
# the diagonal through exact centres also clips the cells between
DIAG_EXTRA = {(BASE_X + 11, BASE_Y + 10), (BASE_X + 10, BASE_Y + 11),
              (BASE_X + 12, BASE_Y + 11), (BASE_X + 11, BASE_Y + 12)}

# 6. private service road -> ignored
j = tile_centre(BASE_X + 20, BASE_Y + 20)
k = tile_centre(BASE_X + 21, BASE_Y + 20)
W({"highway": "service", "access": "private"}, [j, k])

# 7. tiny 6 m stub straddling a tile boundary -> both tiles counted
lon_b, _ = tilef_to_lonlat(BASE_X + 30, BASE_Y + 30)
_, lat_m = tilef_to_lonlat(BASE_X + 30, BASE_Y + 30 + 0.5)
mpd = 111320 * math.cos(math.radians(lat_m))
W({"highway": "unclassified"},
  [(lon_b - 3.0 / mpd, lat_m), (lon_b + 3.0 / mpd, lat_m)])
EXPECT |= {(BASE_X + 29, BASE_Y + 30), (BASE_X + 30, BASE_Y + 30)}

# 8. a building (no highway) -> ignored
m = tile_centre(BASE_X + 40, BASE_Y + 40)
n = tile_centre(BASE_X + 41, BASE_Y + 40)
W({"building": "yes"}, [m, n])

# --- write the pbf ------------------------------------------------------- #
writer = osmium.SimpleWriter(PBF)
for i, lon, lat in nodes:
    writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), version=1))
for i, tags, nd in ways:
    writer.add_way(osmium.osm.mutable.Way(id=i, nodes=nd, tags=tags, version=1))
writer.close()
print(f"wrote {PBF} ({os.path.getsize(PBF)} bytes), "
      f"{len(nodes)} nodes, {len(ways)} ways")

# --- run the real CLI ---------------------------------------------------- #
res = subprocess.run(
    [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "genBDD",
     "--pbf", PBF, "--out", os.path.join(OUT, "t.npz"), "-v"],
    capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(1)

grid = TileGrid.load(os.path.join(OUT, "t.npz"))
got = {(grid.x0 + int(c), grid.y0 + int(r)) for r, c in zip(*grid.bits.nonzero())}

missing = EXPECT - got
extra = got - EXPECT - DIAG_EXTRA
ok = True
if missing:
    print(f"FAIL missing tiles: {sorted(missing)}"); ok = False
if extra:
    print(f"FAIL unexpected tiles: {sorted(extra)}"); ok = False

# the excluded ways must not have contributed anything on their rows
for row, label in ((BASE_Y + 5, "motorway"), (BASE_Y + 6, "unpaved track"),
                   (BASE_Y + 20, "private service"), (BASE_Y + 40, "building")):
    bad = {t for t in got if t[1] == row}
    if bad:
        print(f"FAIL {label} row {row} should be empty, got {sorted(bad)}"); ok = False


# --- the single-command build: roads and elevation in one pass ------------ #
import numpy as _np, rasterio as _rio
from rasterio.transform import from_origin as _fo

SRC = os.path.join(OUT, "SRC"); os.makedirs(SRC, exist_ok=True)
DBR = os.path.join(OUT, "Database_OSM")
DEMB = os.path.join(OUT, "DEM_DB")
import Tilehunt as _t

# a source tile covering wherever the synthetic ways landed
with _np.load(os.path.join(OUT, "t.npz")) as _f:
    pass
_g = _t.TileGrid.load(os.path.join(OUT, "t.npz"))
_yy, _xx = _np.nonzero(_g.bits)
_lon, _lat = _t.tilef_to_lonlat(int(_xx[0]) + _g.x0, int(_yy[0]) + _g.y0)
_la, _lo = math.floor(_lat), math.floor(_lon)
_n, _st = 1201, 1.0 / 1200
_ln = _lo + _np.arange(_n) * _st
_lt = (_la + 1) - _np.arange(_n) * _st
_z = (200.0 + 150.0 * _np.cos(2 * _np.pi * _ln[None, :] * 5)
      * _np.cos(2 * _np.pi * _lt[:, None] * 5)).astype("float32")
with _rio.open(os.path.join(SRC, _t.copernicus_name(_la, _lo) + ".tif"), "w",
               driver="GTiff", height=_n, width=_n, count=1, dtype="float32",
               crs="EPSG:4326", transform=_fo(_lo, _la + 1, _st, _st)) as _ds:
    _ds.write(_z, 1); _ds.update_tags(AREA_OR_POINT="Point")

r2 = subprocess.run(
    [sys.executable, "-I", os.path.join(REPO, "tilehunt.py"), "genBDD",
     "--pbf", PBF, "--out", os.path.join(OUT, "t2.npz"),
     "--db", DBR, "--dem-db", DEMB, "--dem-src", SRC],
    capture_output=True, text=True)
print(r2.stdout[-700:] if r2.returncode == 0 else r2.stderr[-1500:])
if r2.returncode != 0:
    print("FAIL the one-command build failed"); ok = False
else:
    road_blocks = [f for f in os.listdir(DBR) if f.endswith(".npz")]
    dem_blocks = [f for f in os.listdir(DEMB) if f.startswith("h")]
    print(f"one command produced {len(road_blocks)} road block(s) and "
          f"{len(dem_blocks)} elevation block(s)")
    if not road_blocks or not dem_blocks:
        print("FAIL one half of the database is missing"); ok = False
    db = _t.open_dem(DEMB)
    v = db.sample(_np.array([_lon]), _np.array([_lat]))[0]
    want = (200.0 + 150.0 * math.cos(2 * math.pi * _lon * 5)
            * math.cos(2 * math.pi * _lat * 5))
    print(f"elevation at the first green tile: {v:.1f} m "
          f"(source says {want:.1f} m)")
    if _np.isnan(v) or abs(v - want) > 2.0:
        print("FAIL elevation does not match the source"); ok = False

print(f"\nexpected {len(EXPECT)} core tiles, got {len(got)} total "
      f"({len(got & DIAG_EXTRA)} are diagonal corner-clips, which is correct)")
print("PBF path OK" if ok else "PBF path FAILED")
sys.exit(0 if ok else 1)
