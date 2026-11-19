"""Elevation profiles along a track, climbing totals, and the gpx writer."""
from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

import numpy as np

from .tiles import haversine_m
from .graph import RoadGraph
from .dem import DEM


def resample_track(lon: np.ndarray, lat: np.ndarray, step_m: float):
    """Points every step_m along the polyline, with their running distance."""
    seg = np.zeros(len(lon))
    for i in range(1, len(lon)):
        seg[i] = haversine_m(lon[i - 1], lat[i - 1], lon[i], lat[i])
    cum = np.cumsum(seg)
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 0:
        return np.array([0.0]), lon[:1].copy(), lat[:1].copy()
    want = np.arange(0.0, total + step_m, step_m)
    return want, np.interp(want, cum, lon), np.interp(want, cum, lat)


def smooth(values: np.ndarray, window_pts: int) -> np.ndarray:
    if window_pts < 2 or len(values) < 3:
        return values
    k = min(window_pts | 1, len(values) if len(values) % 2 else len(values) - 1)
    if k < 3:
        return values
    pad = k // 2
    padded = np.concatenate([np.full(pad, values[0]), values,
                             np.full(pad, values[-1])])
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")


def ascent_descent(elev: np.ndarray, threshold: float = 0.0
                   ) -> tuple[float, float]:
    """
    Metres climbed and descended, counting a reversal only once the profile
    has turned by `threshold` metres.

    Smoothing alone does not save you: summing every positive difference of a
    real DEM along a flat road still invents hundreds of metres. The
    hysteresis below is what trackers actually do -- a wobble smaller than the
    threshold is not a hill.
    """
    if len(elev) < 2:
        return 0.0, 0.0
    if threshold <= 0:
        d = np.diff(elev)
        return float(d[d > 0].sum()), float(-d[d < 0].sum())

    up = down = 0.0
    lo = hi = float(elev[0])
    trend = 0                                   # 0 unknown, 1 up, -1 down
    for e in map(float, elev[1:]):
        if trend >= 0:
            if e > hi:
                hi = e
            # an opening climb must establish the trend, or its ascent is
            # never credited when the profile finally turns over
            if trend == 0 and hi - lo >= threshold:
                trend = 1
            if hi - e >= threshold:
                if trend == 1:
                    up += hi - lo
                trend, lo = -1, e
        else:
            if e < lo:
                lo = e
            if e - lo >= threshold:
                down += hi - lo
                trend, hi = 1, e
    if trend == 1:
        up += hi - lo
    elif trend == -1:
        down += hi - lo
    return up, down


def build_profile(g: "RoadGraph", seq: Sequence[int], dem: DEM,
                  step_m: float, smooth_m: float, threshold_m: float = 3.0,
                  max_points: int = 1200):
    """Elevation profile of the ride, plus the climbing totals."""
    lon = np.array([g.lon[i] for i in seq], dtype=np.float64)
    lat = np.array([g.lat[i] for i in seq], dtype=np.float64)
    return profile_from_coords(lon, lat, dem, step_m, smooth_m,
                               threshold_m, max_points)


def profile_from_coords(lon: np.ndarray, lat: np.ndarray, dem: DEM,
                        step_m: float, smooth_m: float,
                        threshold_m: float = 3.0, max_points: int = 1200):
    dist, slon, slat = resample_track(lon, lat, step_m)
    raw = dem.sample(slon, slat)
    if np.isnan(raw).all():
        return None
    # fill the holes so a missing tile does not swallow the whole profile
    idx = np.arange(len(raw))
    good = ~np.isnan(raw)
    raw = np.interp(idx, idx[good], raw[good])
    sm = smooth(raw, int(round(smooth_m / step_m)))
    up, down = ascent_descent(sm, threshold_m)

    keep = max(1, len(sm) // max_points)
    thin = sm[::keep]
    return {
        "step_m": round(step_m * keep, 2),
        "elev": [int(round(v)) for v in thin],
        "up": round(up),
        "down": round(down),
        "min": int(round(float(sm.min()))),
        "max": int(round(float(sm.max()))),
        "km": round(float(dist[-1]) / 1000.0, 2),
        "covered": float(good.mean()),
        "src": sorted(dem.sources),
    }


# --------------------------------------------------------------------------- #
# routing: output                                                              #
# --------------------------------------------------------------------------- #

def write_gpx(path: str, g: "RoadGraph", seq: Sequence[int], name: str,
              dem=None) -> None:
    esc = (name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    ele = None
    if dem is not None:
        lon = np.array([g.lon[i] for i in seq])
        lat = np.array([g.lat[i] for i in seq])
        ele = dem.sample(lon, lat)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<gpx version="1.1" creator="tilehunt" '
                 'xmlns="http://www.topografix.com/GPX/1/1">\n'
                 f'  <metadata><name>{esc}</name></metadata>\n'
                 f'  <trk><name>{esc}</name><trkseg>\n')
        for k, i in enumerate(seq):
            if ele is not None and not math.isnan(ele[k]):
                fh.write(f'    <trkpt lat="{g.lat[i]:.7f}" lon="{g.lon[i]:.7f}">'
                         f'<ele>{ele[k]:.1f}</ele></trkpt>\n')
            else:
                fh.write(f'    <trkpt lat="{g.lat[i]:.7f}" '
                         f'lon="{g.lon[i]:.7f}"/>\n')
        fh.write('  </trkseg></trk>\n</gpx>\n')


def cmd_profile(a: argparse.Namespace) -> None:
    """Attach an elevation profile to a route that has already been solved."""
    from .dem import copernicus_url, default_dem, ensure_dem, hgt_url, open_dem
    if not a.route:
        raise SystemExit("give --route JSON, or --runname NAME to take "
                         "Runs/Square/NAME/NAME.route.json")
    with open(a.route, "r", encoding="utf-8") as fh:
        rj = json.load(fh)
    cs = rj.get("coords") or []
    if len(cs) < 2:
        raise SystemExit(f"{a.route} holds no track")
    lon = np.array([c[0] for c in cs], dtype=np.float64)
    lat = np.array([c[1] for c in cs], dtype=np.float64)

    a.dem = default_dem(a.dem)
    if not ensure_dem(a.dem, lon, lat, a.dem_res, a.dem_src, False,
                      getattr(a, "no_dem_fetch", False)):
        raise SystemExit("no heights available")
    dem = open_dem(a.dem)
    prof = profile_from_coords(lon, lat, dem, a.dem_step, a.dem_smooth,
                               a.dem_threshold)
    if dem.missing:
        print(f"missing     : {len(dem.missing)} tile(s) -- download either:")
        for la, lo in sorted(dem.missing):
            print(f"   copernicus: {copernicus_url(la, lo)}")
            print(f"   srtm      : {hgt_url(la, lo)}")
    if not prof:
        raise SystemExit("no tile covers this route")

    rj["profile"] = prof
    rj.setdefault("stats", {})["up"] = prof["up"]
    rj["stats"]["down"] = prof["down"]
    with open(a.route, "w", encoding="utf-8") as fh:
        json.dump(rj, fh, separators=(",", ":"))
    print(f"source      : {', '.join(prof['src']) or 'none'}")
    print(f"elevation   : +{prof['up']:,} m / -{prof['down']:,} m   "
          f"(min {prof['min']} m, max {prof['max']} m)")
    print(f"updated     : {a.route}")

    if a.gpx:
        ele = dem.sample(lon, lat)
        esc = (a.name or "tilehunt route").replace("&", "&amp;")
        with open(a.gpx, "w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<gpx version="1.1" creator="tilehunt" '
                     'xmlns="http://www.topografix.com/GPX/1/1">\n'
                     f'  <metadata><name>{esc}</name></metadata>\n'
                     f'  <trk><name>{esc}</name><trkseg>\n')
            for k in range(len(lon)):
                if not math.isnan(ele[k]):
                    fh.write(f'    <trkpt lat="{lat[k]:.7f}" lon="{lon[k]:.7f}">'
                             f'<ele>{ele[k]:.1f}</ele></trkpt>\n')
                else:
                    fh.write(f'    <trkpt lat="{lat[k]:.7f}" '
                             f'lon="{lon[k]:.7f}"/>\n')
            fh.write('  </trkseg></trk>\n</gpx>\n')
        print(f"gpx         : {a.gpx}")
