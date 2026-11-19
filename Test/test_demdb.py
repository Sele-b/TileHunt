"""Elevation block database: accuracy, seams, straddling cells, chunking."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import os, shutil, sys, tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin

import Tilehunt as t

fails = []


def surf(lon, lat):
    return (300.0 + 200.0 * np.cos(2 * np.pi * lon * 3)
            * np.cos(2 * np.pi * lat * 3) + 60.0 * lat + 40.0 * lon)


ROOT = tempfile.mkdtemp()
SRC = os.path.join(ROOT, "SRC"); os.makedirs(SRC)
DB = os.path.join(ROOT, "DB")

# two adjacent 1-degree source tiles, so a block can straddle the seam
for la, lo in [(48, -2), (48, -1)]:
    n, st = 1801, 1.0 / 1800
    lon = lo + np.arange(n) * st
    lat = (la + 1) - np.arange(n) * st
    z = surf(lon[None, :], lat[:, None]).astype("float32")
    name = t.copernicus_name(la, lo) + ".tif"
    with rasterio.open(os.path.join(SRC, name), "w", driver="GTiff",
                       height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lo, la + 1, st, st),
                       compress="deflate", tiled=True) as ds:
        ds.write(z, 1); ds.update_tags(AREA_OR_POINT="Point")
print(f"two source tiles of {n}x{n} written")

# a zone wide enough to cross the 1-degree seam at 1W, built in tiny chunks so
# the chunking path is the one under test
xs, ys = np.meshgrid(np.arange(8120, 8190), np.arange(5650, 5680))
blocks = sorted(t.dem_blocks_for_tiles(xs.ravel(), ys.ravel()))
r = t.dem_build(DB, blocks, t.LocalSource(SRC), res=64)
print(f"{len(blocks)} blocks asked, {r['written']} written, "
      f"{r['cells']} cell reads, {r['bytes']/1e6:.1f} MB, "
      f"{r['holes']} void samples")
if r["written"] != len(blocks):
    print("FAIL not every block was written"); fails.append("written")
if r["holes"]:
    print("FAIL holes inside fully covered terrain"); fails.append("holes")

db = t.open_dem(DB)
if not isinstance(db, t.DEMDB):
    print("FAIL open_dem did not recognise the database"); fails.append("open")

# accuracy against the analytic surface, over the whole built area
rng = np.random.default_rng(11)
lo0, la1 = t.tilef_to_lonlat(8121, 5679)
lo1, la0 = t.tilef_to_lonlat(8189, 5651)
qlon = lo0 + rng.random(40000) * (lo1 - lo0)
qlat = la1 + rng.random(40000) * (la0 - la1)
got = db.sample(qlon, qlat)
err = np.abs(got - surf(qlon, qlat))
print(f"accuracy: mean {err.mean():.2f} m  p99 {np.percentile(err,99):.2f} m  "
      f"max {err.max():.2f} m  holes {int(np.isnan(got).sum())}")
if np.isnan(got).any() or err.max() > 2.0:
    print("FAIL database does not reproduce the source"); fails.append("accuracy")

# a block straddling the 1-degree seam must be whole, not half-filled
seam_lon = -1.0
xt, yt = t._lonlat_to_tile(np.array([seam_lon]), np.array([48.47]))
bx, by = int(xt[0]) >> 3, int(yt[0]) >> 3
ln = np.linspace(seam_lon - 0.01, seam_lon + 0.01, 4001)
v = db.sample(ln, np.full_like(ln, 48.47))
print(f"across the 1-degree seam in block {bx},{by}: "
      f"{int(np.isnan(v).sum())} holes, largest step {np.abs(np.diff(v)).max():.3f} m")
if np.isnan(v).any() or np.abs(np.diff(v)).max() > 1.0:
    print("FAIL seam between source tiles"); fails.append("seam")

# block-to-block seams must be invisible too
edge = 8128                                   # a multiple of 8: a block edge
lo_e, _ = t.tilef_to_lonlat(edge, 5660)
ln = np.linspace(lo_e - 0.004, lo_e + 0.004, 4001)
v = db.sample(ln, np.full_like(ln, 48.47))
step = float(np.abs(np.diff(v)).max())
print(f"across a block edge: largest step {step:.4f} m")
if step > 1.0:
    print("FAIL block seam discontinuity"); fails.append("blockseam")

# a point outside the build is reported, never invented
db2 = t.open_dem(DB)
out = db2.sample(np.array([3.0]), np.array([45.0]))
print(f"outside the build: value {out[0]}, missing {sorted(db2.missing)}")
if not np.isnan(out[0]) or not db2.missing:
    print("FAIL out-of-build point not reported"); fails.append("outside")

# only the blocks touched are ever read off disk
db3 = t.open_dem(DB)
db3.sample(np.array([-1.5]), np.array([48.47]))
loaded = len([k for k, v in db3._cache.items() if v is not None])
print(f"one point loaded {loaded} block file(s) out of {r['written']}")
if loaded != 1:
    print("FAIL a single point read more than its own block"); fails.append("lazy")

# rebuilding is a no-op, and a different --res is refused rather than mixed
before = sorted(os.listdir(DB))
t.dem_build(DB, blocks[:3], t.LocalSource(SRC), res=64)
if sorted(os.listdir(DB)) != before:
    print("FAIL rebuild changed the file set"); fails.append("rebuild")
try:
    t.dem_build(DB, blocks[:1], t.LocalSource(SRC), res=32)
    print("FAIL mixing resolutions was allowed"); fails.append("res")
except SystemExit:
    print("mixing resolutions refused, as it must be")

# a build stopped halfway and resumed must equal a build done in one go
HALF = os.path.join(ROOT, "HALF")
t.dem_build(HALF, blocks[:len(blocks)//2], t.LocalSource(SRC), res=64)
part = len([f for f in os.listdir(HALF) if f.startswith("h")])
t.dem_build(HALF, blocks, t.LocalSource(SRC), res=64)
whole = len([f for f in os.listdir(HALF) if f.startswith("h")])
same = True
for b in blocks:
    f1 = os.path.join(DB, t.dem_block_name(*b))
    f2 = os.path.join(HALF, t.dem_block_name(*b))
    with np.load(f1) as A, np.load(f2) as B:
        if not np.array_equal(A["z"], B["z"]):
            same = False
print(f"resumed build: {part} blocks after the first half, {whole} after the "
      f"second, identical to a one-shot build: {same}")
if whole != len(blocks) or not same:
    print("FAIL resuming did not reproduce a full build"); fails.append("resume")

# the second call must not re-read the source cells it has already finished
class Counting(t.LocalSource):
    reads = 0

    def cell(self, la, lo):
        Counting.reads += 1
        return super().cell(la, lo)

again = Counting(SRC)
t.dem_build(HALF, blocks, again, res=64)
print(f"re-running a finished build opened {Counting.reads} source cell(s)")
if Counting.reads:
    print("FAIL a finished build still read the source"); fails.append("idempotent")

# the profile machinery accepts a database in place of a tile folder
lon = np.linspace(-1.45, -1.25, 400)
lat = np.linspace(48.40, 48.55, 400)
prof = t.profile_from_coords(lon, lat, t.open_dem(DB), 25.0, 120.0)
print(f"profile through the database: {prof['km']} km, +{prof['up']} / "
      f"-{prof['down']} m, source {prof['src']}")
if prof is None or prof["covered"] < 0.999:
    print("FAIL profile did not come out of the database"); fails.append("profile")

st = db.stats()
print(f"stats: {st['blocks']} blocks, {st['bytes']/1e6:.1f} MB, "
      f"res {st['res']}, side {st['side']}")
print(f"per block: {st['bytes']/st['blocks']/1024:.0f} kB")

shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "dem database OK")
sys.exit(1 if fails else 0)
