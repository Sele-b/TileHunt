"""The Statshunters client, against a stand-in serving the documented shape."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import http.server, json, os, shutil, socketserver, subprocess, sys
import tempfile, threading
import numpy as np

for v in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(v, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

import Tilehunt as t

fails = []
ROOT = tempfile.mkdtemp()
DATA = os.path.join(ROOT, "Data"); os.makedirs(DATA)

# A fake history: a solid 7x7 block of tiles ridden, plus a scattering around
# it, split across enough activities to force pagination.
BLOCK = [(8120 + i, 5650 + j) for i in range(7) for j in range(7)]
SCATTER = [(8130, 5660), (8131, 5660), (8140, 5670), (8100, 5640),
           (8107, 5637)]      # inside the mask, outside the roads: a red tile
RUNS_ONLY = [(8200, 5700), (8201, 5700)]
ALL_RIDE = BLOCK + SCATTER

acts = []
for k in range(0, len(ALL_RIDE), 3):
    acts.append({"id": 1000 + k, "date": f"2025-0{1 + k % 9}-15",
                 "type": "Ride", "distance": 42000,
                 "tiles": [{"x": x, "y": y} for x, y in ALL_RIDE[k:k + 3]]})
acts.append({"id": 9001, "date": "2024-03-02", "type": "Run",
             "distance": 10000,
             "tiles": [{"x": x, "y": y} for x, y in RUNS_ONLY]})
acts.append({"id": 9002, "date": "2019-01-01", "type": "Ride",
             "distance": 5000, "tiles": [{"x": 7000, "y": 5000}]})
PAGE = 7                                   # small, so paging is exercised
print(f"stand-in holds {len(acts)} activities over "
      f"{-(-len(acts) // PAGE)} pages")

CODE = "abcdef123456"
hits = {"pages": 0}


class H(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if not self.path.startswith(f"/share/{CODE}/api/activities"):
            self.send_error(404, "Not Found")
            return
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        page = int(dict(p.split("=") for p in q.split("&") if "=" in p)
                   .get("page", 1))
        hits["pages"] += 1
        chunk = acts[(page - 1) * PAGE: page * PAGE]
        body = json.dumps({"activities": chunk}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


srv = S(("127.0.0.1", 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
LINK = f"http://127.0.0.1:{port}/share/{CODE}"


def run(*args, expect_fail=False):
    extra = (["--share-base", f"http://127.0.0.1:{port}"]
             if args[0].startswith("genVisited") else [])
    r = subprocess.run([sys.executable, os.path.join(REPO, "tilehunt.py"),
                        *args, "--data-dir", DATA, *extra],
                       capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"unexpected exit: {' '.join(args)}")
    return r.stdout + r.stderr


# the share link is a local URL here, so bypass the shape check by calling the
# fetch directly the way the command does
import types
out = run("genVisited", "--share", LINK, "--out", "visited.npz", "-v")
print("\n".join("   " + l for l in out.strip().splitlines()[:9]))

g = t.TileGrid.load(os.path.join(DATA, "visited.npz"))
got = {(int(x) + g.x0, int(y) + g.y0)
       for y, x in zip(*np.nonzero(g.bits))}
want = set(ALL_RIDE) | set(RUNS_ONLY) | {(7000, 5000)}
print(f"   tiles: {len(got)} read, {len(want)} expected, identical: "
      f"{got == want}")
if got != want:
    print("FAIL the tile set does not match"); fails.append("tiles")
if hits["pages"] < 3:
    print(f"FAIL only {hits['pages']} page(s) fetched"); fails.append("paging")
else:
    print(f"   paging: {hits['pages']} requests, stopped on the empty page")

# the max square is the thing the whole game is played for
side = g.meta.get("max_square", [0])[0]
print(f"   max square reported: {side} (planted a solid 7x7)")
if side != 7:
    print("FAIL the max square is wrong"); fails.append("square")

# filters
out = run("genVisited", "--share", LINK, "--type", "Ride",
          "--since", "2025-01-01", "--out", "rides.npz")
g2 = t.TileGrid.load(os.path.join(DATA, "rides.npz"))
got2 = {(int(x) + g2.x0, int(y) + g2.y0)
        for y, x in zip(*np.nonzero(g2.bits))}
print(f"   with --type Ride --since 2025-01-01: {len(got2)} tiles "
      f"(runs and the 2019 ride dropped: {got2 == set(ALL_RIDE)})")
if got2 != set(ALL_RIDE):
    print("FAIL the filters did not apply"); fails.append("filters")

# the link is remembered, and reused when omitted
remembered = os.path.join(DATA, t.SHARE_FILE)
print(f"   link remembered in {os.path.basename(remembered)}: "
      f"{os.path.exists(remembered)}")
if not os.path.exists(remembered):
    fails.append("remember")
else:
    body = open(remembered, encoding="utf-8").read()
    if "keep this file out of git" not in body:
        print("FAIL the file does not warn what it holds"); fails.append("warn")
out = run("genVisited", "--out", "again.npz")
print(f"   second call without --share worked: "
      f"{os.path.exists(os.path.join(DATA, 'again.npz'))}")
if not os.path.exists(os.path.join(DATA, "again.npz")):
    fails.append("reuse")

# a revoked or mistyped link says so, rather than writing an empty grid
out = run("genVisited", "--share", f"http://127.0.0.1:{port}/share/nope",
          "--out", "x.npz", "--no-remember", expect_fail=True)
print(f"   a dead link is reported: {'404' in out}")
if "404" not in out:
    print("FAIL a dead link was not reported"); fails.append("dead")

def _accepts(v):
    try:
        t.share_url(v)
        return True
    except SystemExit:
        return False


# link shapes
print("   a non-Statshunters host is refused by default: "
      + str(not _accepts("http://evil.example/share/abc123")))
for good in ("https://www.statshunters.com/share/abc123",
             "https://statshunters.com/share/abc123/", "abc123"):
    if t.share_url(good) != "https://www.statshunters.com/share/abc123":
        print(f"FAIL {good!r} not understood"); fails.append("parse")
for bad in ("https://example.com/share/abc", "", "ab"):
    try:
        t.share_url(bad)
        print(f"FAIL {bad!r} accepted"); fails.append("parse-bad")
    except SystemExit:
        pass
print("   share link shapes understood, rubbish refused")

# ---- the map: a layer, the counts, and tiles drawn dimmer ---------------- #
MAPS = os.path.join(ROOT, "Maps"); os.makedirs(MAPS, exist_ok=True)
t.TileGrid.from_tiles([(x, y) for x in range(8110, 8171)
                       for y in range(5640, 5701)]
                      ).save(os.path.join(DATA, "france.paved.npz"))
# a mask wider than the roads, so red tiles really exist and the
# remove-red / remove-green switches have something to act on
t.TileGrid.from_tiles([(x, y) for x in range(8105, 8176)
                       for y in range(5635, 5706)]
                      ).save(os.path.join(DATA, "mask.npz"))
out = run("buildmap", "--tiles", "france.paved.npz", "--mask", "mask.npz",
          "--visited", "visited.npz", "--out", "vis.html", "--maps-dir", MAPS)
for l in out.splitlines():
    if l.startswith(("visited", "max square")):
        print("   " + l.rstrip())
html = open(os.path.join(MAPS, "vis.html"), encoding="utf-8").read()
for probe, label in [('"visitedStats"', "counts in the page"),
                     ('id="b-visited"', "layer toggle"),
                     ('id="vis-kv"', "panel rows"),
                     ('id="b-green"', "remove-green switch")]:
    ok = probe in html
    print(f"   {label}: {'yes' if ok else 'NO'}")
    if not ok:
        fails.append(label)

node = shutil.which("node") or shutil.which("nodejs")
if node:
    r = subprocess.run([node, os.path.join(HERE, "test_draw.mjs"),
                        os.path.join(MAPS, "vis.html")],
                       capture_output=True, text=True)
    for line in (r.stdout or r.stderr).strip().splitlines():
        if ("FAIL" in line or "drawing" in line or line.startswith("b-")
                or "held" in line or "layer off" in line
                or "only-my-tiles" in line):
            print("   " + line)
    if r.returncode != 0:
        fails.append("drawing")

srv.shutdown()
shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "statshunters OK")
sys.exit(1 if fails else 0)
