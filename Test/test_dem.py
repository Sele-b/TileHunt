#!/usr/bin/env python3
"""Elevation sampling and climb totals, against a DEM whose shape we know."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import math, os, shutil, sys, tempfile
import numpy as np
import Tilehunt
from Tilehunt import (DEM, hgt_name, hgt_url, resample_track, smooth,
                      ascent_descent, build_profile, haversine_m)

OUT = os.path.join(tempfile.gettempdir(), "tilehunt-demtest")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)

# a 1-arcsecond tile over N47E002 whose height is a pure cosine in longitude:
#   h(lon) = 500 + 200*cos(2*pi*(lon-2)*HILLS)
N, HILLS = 3601, 4.0
lon = 2.0 + np.arange(N) / (N - 1)
h = 500.0 + 200.0 * np.cos(2 * math.pi * (lon - 2.0) * HILLS)
grid = np.tile(h, (N, 1)).astype(">i2")
open(os.path.join(OUT, hgt_name(47, 2)), "wb").write(grid.tobytes())
print(f"wrote {hgt_name(47,2)}: cosine, 4 hills of 400 m peak-to-peak")

fails = []
dem = DEM(OUT)

# point sampling against the analytic surface
test_lons = np.array([2.05, 2.1, 2.37, 2.5, 2.9])
test_lats = np.full(5, 47.5)
got = dem.sample(test_lons, test_lats)
want = 500.0 + 200.0 * np.cos(2 * math.pi * (test_lons - 2.0) * HILLS)
err = np.abs(got - want).max()
print(f"bilinear sampling: worst error {err:.2f} m")
if err > 1.0:
    print("FAIL sampling does not match the analytic surface"); fails.append("sample")

# a west-east track across the whole tile must climb 4 hills up and 4 down
class _G:
    pass
g = _G()
steps = 2000
g.lon = np.linspace(2.001, 2.999, steps)
g.lat = np.full(steps, 47.5)
prof = build_profile(g, list(range(steps)), dem, 25.0, 120.0, 3.0)
# the track starts and ends near a crest, so expect ~4 full oscillations
expect = 400.0 * HILLS
print(f"track across the tile: +{prof['up']} m / -{prof['down']} m "
      f"(analytic ~{expect:.0f} m each)")
for name, v in (("up", prof["up"]), ("down", prof["down"])):
    if abs(v - expect) > 0.08 * expect:
        print(f"FAIL {name} is {v} m, expected about {expect:.0f}")
        fails.append(name)
if not (690 < prof["max"] <= 700 and 300 <= prof["min"] < 310):
    print(f"FAIL min/max {prof['min']}/{prof['max']} m, expected about 300/700")
    fails.append("range")

# smoothing must kill DEM noise rather than invent climbing
flat = np.full(4000, 100.0) + np.random.default_rng(0).normal(0, 1.5, 4000)
raw_up, _ = ascent_descent(flat)
sm_up, _ = ascent_descent(smooth(flat, 5), 3.0)
print(f"noisy flat road: {raw_up:.0f} m of ascent raw, {sm_up:.0f} m with "
      f"smoothing + 3 m threshold")
if sm_up > raw_up * 0.05:
    print("FAIL noise is still being counted as climbing"); fails.append("smooth")

# ... and the threshold must NOT eat real climbing, whichever way it starts
for label, prof_, wu, wd in (
        ("up then down", np.concatenate([np.linspace(100, 400, 500),
                                         np.linspace(400, 100, 500)]), 300, 300),
        ("down then up", np.concatenate([np.linspace(400, 100, 500),
                                         np.linspace(100, 400, 500)]), 300, 300),
        ("one climb only", np.linspace(100, 400, 500), 300, 0),
        ("one descent only", np.linspace(400, 100, 500), 0, 300)):
    hu, hd = ascent_descent(prof_, 3.0)
    ok = abs(hu - wu) <= 5 and abs(hd - wd) <= 5
    print(f"   {label:18s} +{hu:5.0f} / -{hd:5.0f} m   (want +{wu} / -{wd})"
          + ("" if ok else "   <-- FAIL"))
    if not ok:
        fails.append("threshold:" + label)

# a missing tile is reported with a usable URL
d2 = DEM(OUT)
d2.sample(np.array([9.5]), np.array([44.5]))
print(f"missing tile reported: {sorted(d2.missing)} -> {hgt_url(44, 9)}")
if (44, 9) not in d2.missing or "N44E009" not in hgt_url(44, 9):
    print("FAIL missing tile not reported properly"); fails.append("missing")



# --------------------------------------------------------------------------- #
# Copernicus-style GeoTIFF: same analytic surface, read through rasterio
# --------------------------------------------------------------------------- #
def test_geotiff():
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print("rasterio absent, GeoTIFF path not exercised")
        return

    root = tempfile.mkdtemp()

    def surface(lon, lat):
        return 800.0 + 800.0 * np.cos(2 * np.pi * (lon - 2.0)) \
                              * np.cos(2 * np.pi * (lat - 47.0))

    # 1. a 1-degree square tile, pixel-is-point, like Copernicus below 50N
    n = 1201
    step = 1.0 / (n - 1)
    lon = 2.0 + np.arange(n) * step
    lat = 48.0 - np.arange(n) * step
    z = surface(lon[None, :], lat[:, None]).astype(np.float32)
    with rasterio.open(
            os.path.join(root, "Copernicus_DSM_COG_10_N47_00_E002_00_DEM.tif"),
            "w", driver="GTiff", height=n, width=n, count=1,
            dtype="float32", crs="EPSG:4326",
            transform=from_origin(2.0, 48.0, step, step)) as ds:
        ds.write(z, 1)
        ds.update_tags(AREA_OR_POINT="Point")

    # 2. a tile with half the longitude resolution, as Copernicus ships above
    #    50N, and tagged Area so the samples sit at pixel centres
    nx, ny = 601, 1201
    lon2 = 3.0 + (np.arange(nx) + 0.5) / nx
    lat2 = 48.0 - (np.arange(ny) + 0.5) / ny
    z2 = surface(lon2[None, :], lat2[:, None]).astype(np.float32)
    with rasterio.open(os.path.join(root, "wide.tif"), "w", driver="GTiff",
                       height=ny, width=nx, count=1, dtype="float32",
                       crs="EPSG:4326",
                       transform=from_origin(3.0, 48.0, 1.0 / nx, 1.0 / ny)) as ds:
        ds.write(z2, 1)
        ds.update_tags(AREA_OR_POINT="Area")

    dem = Tilehunt.DEM(root)

    rng = np.random.default_rng(7)
    qlon = 2.02 + rng.random(4000) * 0.96
    qlat = 47.02 + rng.random(4000) * 0.96
    got = dem.sample(qlon, qlat)
    err = np.nanmax(np.abs(got - surface(qlon, qlat)))
    assert not np.isnan(got).any(), "square GeoTIFF left holes"
    assert err < 1.5, f"square GeoTIFF worst error {err:.2f} m"
    print(f"GeoTIFF 1201x1201 point-grid : worst error {err:.2f} m")

    qlon = 3.02 + rng.random(4000) * 0.96
    qlat = 47.02 + rng.random(4000) * 0.96
    got = dem.sample(qlon, qlat)
    err = np.nanmax(np.abs(got - surface(qlon, qlat)))
    assert not np.isnan(got).any(), "wide GeoTIFF left holes"
    assert err < 6.0, f"wide GeoTIFF worst error {err:.2f} m"
    print(f"GeoTIFF  601x1201 area-grid  : worst error {err:.2f} m")

    # the non-square tile must have been read through its geotransform, not
    # through an assumed n x n grid
    g = [t for t in dem._load_tifs() if t.nx == nx][0]
    assert g.ny == ny and abs(g.dlon - 1.0 / nx) < 1e-12
    assert abs(g.lon0 - (3.0 + 0.5 / nx)) < 1e-9, "Area half-pixel ignored"

    # outside both tiles: reported as missing, not silently extrapolated
    dem.sample(np.array([9.5]), np.array([44.5]))
    assert (44, 9) in dem.missing
    print("out-of-coverage point reported, not extrapolated")

    # the two formats coexist: drop an .hgt next to the tiffs
    m = 1201
    hl = 5.0 + np.arange(m) / (m - 1)
    ha = 51.0 - np.arange(m) / (m - 1)
    hz = np.round(surface(hl[None, :], ha[:, None])).astype(">i2")
    open(os.path.join(root, hgt_name(50, 5)), "wb").write(hz.tobytes())
    dem2 = Tilehunt.DEM(root)
    a = dem2.sample(np.array([5.5]), np.array([50.5]))[0]
    b = dem2.sample(np.array([2.5]), np.array([47.5]))[0]
    assert abs(a - surface(5.5, 50.5)) < 2.0
    assert abs(b - surface(2.5, 47.5)) < 2.0
    assert dem2.sources == {"SRTM .hgt", "GeoTIFF"}
    print("mixed folder: .hgt and GeoTIFF read side by side")

    print(f"copernicus url: {Tilehunt.copernicus_url(48, -2)}")


test_geotiff()

print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "elevation OK")
sys.exit(1 if fails else 0)
