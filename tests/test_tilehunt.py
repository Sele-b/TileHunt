#!/usr/bin/env python3
"""Verification for tilehunt's tile math, traversal and tag filter."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import math
import random
import sys

from tilehunt import (ZOOM, NTILES, lonlat_to_tilef, tilef_to_lonlat, tile_bounds,
                      tile_size_m, walk_segment, walk_polyline, classify_way)

fails = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {extra}")
        fails.append(name)


print("\n== tile math ==")
# Cross-check against an independent formulation of the Mercator y
# (via tan/sec instead of sin), and against the plain longitude formula.
x, y = lonlat_to_tilef(2.3499, 48.8530)
lat_r = math.radians(48.8530)
y_alt = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * 16384
x_alt = (2.3499 + 180.0) / 360.0 * 16384
check("x matches independent formula", abs(x - x_alt) < 1e-6, f"{x} vs {x_alt}")
check("y matches independent formula", abs(y - y_alt) < 1e-6, f"{y} vs {y_alt}")
PARIS = (math.floor(x), math.floor(y))
print(f"       (Paris 48.8530,2.3499 -> tile {PARIS[0]},{PARIS[1]})")

# round trip
for _ in range(2000):
    lon = random.uniform(-179.9, 179.9)
    lat = random.uniform(-84.0, 84.0)
    tx, ty = lonlat_to_tilef(lon, lat)
    lo2, la2 = tilef_to_lonlat(tx, ty)
    if abs(lo2 - lon) > 1e-7 or abs(la2 - lat) > 1e-7:
        check("roundtrip", False, f"{lon},{lat} -> {lo2},{la2}")
        break
else:
    check("lonlat<->tile roundtrip (2000 random)", True)

# tile 0,0 is the NW corner of the world
check("tile 0,0 NW is -180,+85.05", abs(tile_bounds(0, 0)[0] + 180) < 1e-9
      and abs(tile_bounds(0, 0)[3] - 85.0511287798066) < 1e-6)
check("grid is 16384 wide", NTILES == 16384 and ZOOM == 14)

w, h = tile_size_m(PARIS[1])
check("paris tile ~1.5 km", 1.4 < w / 1000 < 1.7 and 1.4 < h / 1000 < 1.7,
      f"got {w:.0f} x {h:.0f} m")

print("\n== segment traversal ==")
# 1. a segment inside one cell yields exactly that cell
check("same-cell segment", list(walk_segment(3.2, 4.3, 3.8, 4.9)) == [(3, 4)])

# 2. horizontal crossing
check("horizontal", list(walk_segment(3.5, 4.5, 6.5, 4.5)) ==
      [(3, 4), (4, 4), (5, 4), (6, 4)])

# 3. vertical, negative direction
check("vertical down->up", list(walk_segment(3.5, 7.5, 3.5, 5.5)) ==
      [(3, 7), (3, 6), (3, 5)])

# 4. exact diagonal through corners: supercover must include the side cells
d = list(walk_segment(0.5, 0.5, 2.5, 2.5))
check("diagonal endpoints", d[0] == (0, 0) and d[-1] == (2, 2), f"got {d}")
check("diagonal is connected 4-neighbour chain",
      all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(d, d[1:])), f"got {d}")

# 5. brute-force validation against a dense-sampling oracle.
#    Dense sampling can only UNDER-report, so: oracle must be a subset of walk.
bad = 0
for trial in range(4000):
    x0 = random.uniform(0, 12); y0 = random.uniform(0, 12)
    x1 = random.uniform(0, 12); y1 = random.uniform(0, 12)
    got = set(walk_segment(x0, y0, x1, y1))
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L * 4000))
    oracle = {(math.floor(x0 + (x1 - x0) * i / n), math.floor(y0 + (y1 - y0) * i / n))
              for i in range(n + 1)}
    if not oracle <= got:
        bad += 1
        if bad < 3:
            print(f"       missing {oracle - got} for ({x0:.4f},{y0:.4f})-({x1:.4f},{y1:.4f})")
check("4000 random segments cover a dense oracle", bad == 0, f"{bad} failures")

# 6. no spurious cells: every returned cell must actually be intersected.
#    Check each returned cell's box against the segment analytically.
def seg_hits_cell(x0, y0, x1, y1, cx, cy):
    """Liang-Barsky clip of the segment against [cx,cx+1]x[cy,cy+1]."""
    t0, t1 = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 - cx), (dx, cx + 1 - x0), (-dy, y0 - cy), (dy, cy + 1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1: return False
                if r > t0: t0 = r
            else:
                if r < t0: return False
                if r < t1: t1 = r
    return t1 >= t0

bad = 0
for trial in range(3000):
    x0 = random.uniform(0, 10); y0 = random.uniform(0, 10)
    x1 = random.uniform(0, 10); y1 = random.uniform(0, 10)
    for (cx, cy) in walk_segment(x0, y0, x1, y1):
        if not seg_hits_cell(x0, y0, x1, y1, cx, cy):
            bad += 1
            if bad < 3:
                print(f"       spurious ({cx},{cy}) for ({x0:.4f},{y0:.4f})-({x1:.4f},{y1:.4f})")
            break
check("3000 random segments return no spurious cells", bad == 0, f"{bad} failures")

# 7. long segment (crosses many cells) terminates and is complete
d = list(walk_segment(0.1, 0.1, 500.9, 300.3))
check("long segment terminates at endpoint", d[-1] == (500, 300), f"got {d[-1]}")
check("long segment chain connected",
      all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(d, d[1:])))

# 8. a 5 m stub poking into a neighbouring tile must register it.
#    Put a point 2 m west of a tile boundary and one 3 m east of it.
bx = PARIS[0] + 1               # a tile boundary just east of the Paris tile
lon_b, _ = tilef_to_lonlat(bx, PARIS[1])
mperdeg = 111320 * math.cos(math.radians(48.853))
p1 = (lon_b - 2.0 / mperdeg, 48.853)
p2 = (lon_b + 3.0 / mperdeg, 48.853)
s = set()
walk_polyline([p1, p2], s)
check("5 m stub across a boundary hits both tiles",
      (bx - 1, PARIS[1]) in s and (bx, PARIS[1]) in s, f"got {sorted(s)}")
check("5 m stub hits nothing else", len(s) == 2, f"got {sorted(s)}")

print("\n== segment length split ==")
from tilehunt import walk_segment_frac, walk_segment, walk_polyline, EARTH_CIRCUMFERENCE_M, NTILES

bad_sum = bad_cells = 0
for _ in range(4000):
    x0 = random.uniform(0, 12); y0 = random.uniform(0, 12)
    x1 = random.uniform(0, 12); y1 = random.uniform(0, 12)
    parts = list(walk_segment_frac(x0, y0, x1, y1))
    total = sum(f for _, _, f in parts)
    if abs(total - 1.0) > 1e-9:
        bad_sum += 1
        if bad_sum < 3:
            print(f"       sum {total} for ({x0:.4f},{y0:.4f})-({x1:.4f},{y1:.4f})")
    if any(f < -1e-12 for _, _, f in parts):
        bad_sum += 1
    cells_frac = {(c[0], c[1]) for c in parts}
    cells_walk = set(walk_segment(x0, y0, x1, y1))
    if cells_frac != cells_walk:
        bad_cells += 1
        if bad_cells < 3:
            print(f"       cells differ {cells_frac ^ cells_walk}")
check("4000 segments: fractions sum to 1", bad_sum == 0, f"{bad_sum} bad")
check("4000 segments: same cells as walk_segment", bad_cells == 0, f"{bad_cells} bad")

# a known length: one degree of longitude at the equator
acc = {}
s_ = set()
walk_polyline([(0.0, 0.0), (1.0, 0.0)], s_, acc)
got_km = sum(acc.values()) / 1000.0
want_km = EARTH_CIRCUMFERENCE_M / 360.0 / 1000.0
check("1 deg of longitude at equator ~111.3 km",
      abs(got_km - want_km) < 0.5, f"got {got_km:.2f}, want {want_km:.2f}")

# a 10 km east-west line at 45N should measure 10 km
import math as _m
lat = 45.0
dlon = 10000.0 / (EARTH_CIRCUMFERENCE_M * _m.cos(_m.radians(lat)) / 360.0)
acc = {}; s_ = set()
walk_polyline([(2.0, lat), (2.0 + dlon, lat)], s_, acc)
check("10 km line measures 10 km", abs(sum(acc.values()) - 10000.0) < 30,
      f"got {sum(acc.values()):.0f} m")
check("10 km line spans several tiles", len(acc) >= 5, f"got {len(acc)} tiles")

print("\n== tag filter ==")
cases = [
    ({"highway": "residential"}, True),
    ({"highway": "secondary"}, True),
    ({"highway": "tertiary", "surface": "asphalt"}, True),
    ({"highway": "cycleway"}, True),
    ({"highway": "cycleway", "surface": "gravel"}, False),
    ({"highway": "track"}, False),
    ({"highway": "track", "surface": "asphalt"}, True),
    ({"highway": "track", "surface": "compacted"}, False),
    ({"highway": "motorway"}, False),
    ({"highway": "motorway_link"}, False),
    ({"highway": "trunk"}, False),
    ({"highway": "trunk", "bicycle": "yes"}, True),
    ({"highway": "residential", "bicycle": "no"}, False),
    ({"highway": "service", "access": "private"}, False),
    ({"highway": "service", "access": "private", "bicycle": "yes"}, True),
    ({"highway": "footway"}, False),
    ({"highway": "footway", "bicycle": "designated", "surface": "asphalt"}, True),
    ({"highway": "footway", "bicycle": "designated", "surface": "ground"}, False),
    ({"highway": "path", "bicycle": "designated"}, False),          # unpaved by default
    ({"highway": "path", "bicycle": "designated", "surface": "paved"}, True),
    ({"highway": "pedestrian", "surface": "sett"}, False),          # needs bicycle=yes
    ({"highway": "pedestrian", "surface": "sett", "bicycle": "yes"}, True),
    ({"highway": "residential", "surface": "cobblestone"}, True),
    ({"highway": "residential", "surface": "sand"}, False),
    ({"highway": "steps"}, False),
    ({"highway": "construction"}, False),
    ({"highway": "primary", "bicycle": "use_sidepath"}, False),
    ({"highway": "unclassified"}, True),
    ({"highway": "living_street"}, True),
    ({"highway": "residential", "area": "yes"}, False),
    ({"building": "yes"}, False),
    ({"highway": "track", "tracktype": "grade1"}, False),
    ({"highway": "track", "tracktype": "grade3"}, False),
    # --- rules added after comparing against a stricter Overpass filter ---
    ({"highway": "service", "service": "driveway"}, False),
    ({"highway": "service", "service": "parking_aisle"}, False),
    ({"highway": "service", "service": "alley"}, True),
    ({"highway": "residential", "vehicle": "private"}, False),
    ({"highway": "residential", "vehicle": "private", "bicycle": "yes"}, True),
    ({"highway": "residential", "smoothness": "very_bad"}, False),
    ({"highway": "residential", "smoothness": "intermediate"}, True),
    ({"highway": "residential", "surface": "asphalt;concrete"}, True),
    ({"highway": "residential", "surface": "asphalt;gravel"}, False),
    ({"highway": "residential", "surface": "wibble"}, False),
    ({"highway": "residential", "access": "agricultural"}, False),
    # --- voie express: motorway restrictions without highway=motorway ---
    ({"highway": "trunk", "bicycle": "yes", "motorroad": "yes"}, False),
    ({"highway": "primary", "motorroad": "yes"}, False),
    ({"highway": "primary", "motorroad": "no"}, True),
    ({"highway": "trunk", "cycleway": "lane"}, True),
    ({"highway": "trunk", "cycleway": "lane", "motorroad": "yes"}, False),
]
for tags, want in cases:
    got, reason = classify_way(tags)
    check(f"{tags} -> {want}", got == want, f"got {got} ({reason})")

print("\n== all-roads preset ==")
from tilehunt import classify_all_roads as car
for tags, want in [({"highway": "motorway"}, False),
                   ({"highway": "trunk"}, True),
                   ({"highway": "trunk", "motorroad": "yes"}, False),
                   ({"highway": "primary", "motorroad": "yes"}, False)]:
    got, reason = car(tags)
    check(f"all-roads {tags} -> {want}", got == want, f"got {got} ({reason})")

print("\n== switches ==")
from tilehunt import classify_way as cw
check("cobbles on  -> sett accepted",
      cw({"highway": "residential", "surface": "sett"})[0] is True)
check("cobbles off -> sett rejected",
      cw({"highway": "residential", "surface": "sett"}, cobbles=False)[0] is False)
check("strict-surface rejects untagged residential",
      cw({"highway": "residential"}, strict_surface=True)[0] is False)
check("strict-surface keeps tagged asphalt",
      cw({"highway": "residential", "surface": "asphalt"}, strict_surface=True)[0] is True)
check("tracktype grade1 rejected by default",
      cw({"highway": "track", "tracktype": "grade1"})[0] is False)
check("tracktype grade1 accepted only when asked",
      cw({"highway": "track", "tracktype": "grade1"}, trust_tracktype=True)[0] is True)
check("surfaced track accepted whatever the switch",
      cw({"highway": "track", "tracktype": "grade1", "surface": "asphalt"})[0] is True)

print()
if fails:
    print(f"{len(fails)} FAILURES")
    sys.exit(1)
print("all checks passed")
