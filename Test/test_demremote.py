"""The range-read path, against a local HTTP server instead of the S3 bucket."""
import os
import sys

# The tool under test sits in the folder above this one, so a suite can be run
# from anywhere:  python3 tests/<name>.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tempfile
import http.server, os, shutil, socketserver, sys, tempfile, threading
import numpy as np, rasterio
from rasterio.transform import from_origin

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
for v in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(v, None)

import Tilehunt as t

fails = []


def surf(lon, lat):
    return 250.0 + 180.0 * np.cos(2 * np.pi * lon * 4) * np.cos(2 * np.pi * lat * 4)


ROOT = tempfile.mkdtemp()
SERVE = os.path.join(ROOT, "serve"); os.makedirs(SERVE)
DB = os.path.join(ROOT, "DB")

# one tiled, deflated COG, laid out the way the bucket lays them out
n, st = 3601, 1.0 / 3600
la, lo = 48, -2
lonv = lo + np.arange(n) * st
latv = (la + 1) - np.arange(n) * st
z = surf(lonv[None, :], latv[:, None]).astype("float32")
name = t.copernicus_name(la, lo) + ".tif"
with rasterio.open(os.path.join(SERVE, name), "w", driver="GTiff",
                   height=n, width=n, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(lo, la + 1, st, st),
                   compress="deflate", tiled=True, blockxsize=256,
                   blockysize=256) as ds:
    ds.write(z, 1); ds.update_tags(AREA_OR_POINT="Point")
full = os.path.getsize(os.path.join(SERVE, name))
print(f"serving one {n}x{n} tiled COG of {full/1e6:.1f} MB")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Stands in for the bucket, and counts only the body bytes it sends."""

    served = 0
    requests = 0
    ranged = 0
    # GDAL only attempts range reads against a server that keeps the
    # connection alive and advertises byte ranges
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=SERVE, **kw)

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/tiff")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "Not Found")       # the bucket's answer for sea
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        Handler.requests += 1
        if rng and rng.startswith("bytes="):
            Handler.ranged += 1
            lo_, _, hi_ = rng[6:].partition("-")
            lo_ = int(lo_)
            hi_ = min(int(hi_) if hi_ else size - 1, size - 1)
            with open(path, "rb") as fh:
                fh.seek(lo_)
                data = fh.read(hi_ - lo_ + 1)
            Handler.served += len(data)
            self.send_response(206)
            self.send_header("Content-Type", "image/tiff")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Range", f"bytes {lo_}-{hi_}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            return
        with open(path, "rb") as fh:
            data = fh.read()
        Handler.served += len(data)
        self.send_response(200)
        self.send_header("Content-Type", "image/tiff")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


srv = Server(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"

src = t.RemoteCopernicus(base=base)
xs, ys = np.meshgrid(np.arange(8124, 8137), np.arange(5656, 5669))
blocks = sorted(t.dem_blocks_for_tiles(xs.ravel(), ys.ravel()))
r = t.dem_build(DB, blocks, src, res=64)
print(f"{len(blocks)} blocks built from {r['cells']} cell read(s), "
      f"{r['bytes']/1e6:.2f} MB on disk, {r['holes']} void samples")
print(f"bytes pulled over HTTP: {Handler.served/1e6:.2f} MB of a "
      f"{full/1e6:.1f} MB tile ({100*Handler.served/full:.0f}%), in "
      f"{Handler.requests} GET(s), {Handler.ranged} of them ranged")
if r["written"] != len(blocks):
    print("FAIL not every block was built"); fails.append("written")
if Handler.ranged < Handler.requests or Handler.served > full * 0.6:
    print("FAIL the whole tile was pulled, so ranges are not being used")
    fails.append("ranges")

db = t.open_dem(DB)
rng = np.random.default_rng(5)
lo0, la1 = t.tilef_to_lonlat(8125, 5668)
lo1, la0 = t.tilef_to_lonlat(8136, 5657)
qlon = lo0 + rng.random(20000) * (lo1 - lo0)
qlat = la1 + rng.random(20000) * (la0 - la1)
got = db.sample(qlon, qlat)
err = np.abs(got - surf(qlon, qlat))
print(f"accuracy vs the analytic surface: mean {err.mean():.2f} m  "
      f"max {err.max():.2f} m  holes {int(np.isnan(got).sum())}")
if np.isnan(got).any() or err.max() > 2.0:
    print("FAIL range-read build does not match the source"); fails.append("accuracy")

# a cell the bucket does not hold (pure sea) must come back empty, not explode
missing = t.RemoteCopernicus(base=base)
missing.aim((5.0, 44.0, 5.5, 44.5))
if missing.cell(44, 5) is not None:
    print("FAIL an absent tile was not reported as absent"); fails.append("absent")
else:
    print("absent tile handled as no-data, not as an error")

srv.shutdown()
shutil.rmtree(ROOT)
print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "remote range reads OK")
sys.exit(1 if fails else 0)
