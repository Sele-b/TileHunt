#!/usr/bin/env python3
"""
Run every suite, from anywhere:   python3 tests/run_all.py

Each suite is a standalone script that exits non-zero on failure, so there is
no test framework to install. The slow ones come last, so a mistake in the
tile maths or the tag filters shows up in seconds rather than minutes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = [
    "test_tilehunt.py",     # tile maths and the OSM tag filters
    "test_dem.py",          # elevation tiles, .hgt and GeoTIFF
    "test_demdb.py",        # the elevation block database
    "test_demremote.py",    # range reads over HTTP
    "test_db.py",           # the road block database
    "test_visited.py",      # Statshunters, and the visited layer
    "test_runs.py",         # Runs/<kind>/<name>/ layout
    "test_gaps.py",         # tiles with no usable road
    "test_snap.py",         # exact endpoints and forced waypoints
    "test_seeker.py",       # mode 3, and the planner panel
    "test_endtoend.py",     # route json -> gpx -> map
    "test_pbf.py",          # the whole pipeline from a synthetic pbf
    "test_route.py",        # the solver against an analytic optimum
]


def main() -> int:
    only = sys.argv[1:]
    suites = [s for s in ORDER if not only or any(o in s for o in only)]
    missing = [s for s in suites if not os.path.exists(os.path.join(HERE, s))]
    for s in missing:
        print(f"{s}: not found")
    width = max(len(s) for s in suites)
    failed, t0 = [], time.time()
    for s in suites:
        if s in missing:
            continue
        print(f"{s:<{width}}  ", end="", flush=True)
        start = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                           capture_output=True, text=True)
        took = time.time() - start
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(f"{last:<28} {took:5.1f}s")
        if r.returncode != 0:
            failed.append(s)
            print(r.stdout[-2500:])
            print(r.stderr[-1500:])
    print()
    print(f"{len(suites) - len(failed)}/{len(suites)} suites passed "
          f"in {time.time() - t0:.0f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
