"""
tilehunt - VeloViewer / Statshunters "explorer tile" coverage from OpenStreetMap.

An explorer tile is a standard slippy-map tile at zoom 14 (16384 x 16384 tiles
covering the world). This tool answers, for every z14 tile in a region:

    "is there at least one paved road, on which cycling is allowed,
     passing through this tile?"

Green tiles are reachable. Red tiles are not (no paved cyclable road inside),
so no amount of route planning will ever let you claim them -- which is exactly
what you need to know before choosing where to attack a max-square.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

  # 0. install
  pip install numpy scipy shapely osmium rasterio

  Layout (created automatically, relative to this script):
      tilehunt.py
      Data/     .osm.pbf and .geojson downloads, plus the .npz datasets
      Maps/     the generated .html

  A bare input name is looked up in Data/. A bare output name goes to Maps/
  if it ends in .html, else to Data/. Any path containing a folder is used
  as given. Override with --data-dir / --maps-dir.

  # 1. put the downloads in Data/
  #    https://download.geofabrik.de/europe/france-latest.osm.pbf
  #    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson  -> countries.geojson
  #    https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson
  #    https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson

  # 2. the database: roads rasterised onto the z14 grid, the road graph for
  #    routing, and the heights.  (~25-30 min for France)
  python tilehunt.py genBDD --pbf france-latest.osm.pbf --out france.paved.npz          --db Database_OSM --dem-db DEM_DB -v

  # 3. the scope mask (countries.geojson is picked up automatically)
  python tilehunt.py mask --iso FRA --clip=-5.6,41.2,9.8,51.3          --min-part-area 0.004 --out france.mask.npz

  # 4. the map. mask, boundary, regions and departements are all found in
  #    Data/ by their conventional names, so nothing else need be named.
  python tilehunt.py buildmap --tiles france.paved.npz --clip-to-mask          --out france.html

  # 4b. from the map: "plan a square ride" hands you the two commands
  python tilehunt.py compSquareRoute --db Database_OSM          --corners 8314,5512 8329,5527 --start 8329,5512 --end 8329,5512          --cands 6 --seconds 120 --reopt-seconds 60          --gpx z8314_5512.gpx --out z8314_5512.route.json

  # 5. inspect a candidate square, or ask why one tile came out green
  python tilehunt.py zone --tiles france.paved.npz --corners 8290,5660 8301,5671
  python tilehunt.py why --pbf france-latest.osm.pbf --tile 8301,5708 --show-rejected

Sub-extracts are faster to iterate on, e.g.
  https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf
  https://download.geofabrik.de/europe/france/centre-latest.osm.pbf
`genBDD` accepts several --pbf flags and merges them.
The former names `tiles`, `map` and `route` still work.

--------------------------------------------------------------------------------
NOTES ON CORRECTNESS
--------------------------------------------------------------------------------
* Rasterisation is exact, not sampled. Every tile a way's polyline passes
  through is recorded, including tiles where it only clips a corner, via a
  grid-traversal (supercover DDA) walk in tile space. A 5 m stub poking into a
  tile counts -- which matches how the challenge is scored.
* Straight lines in Web Mercator are rhumb lines, not great circles. Over OSM
  node spacing (metres to a few hundred metres) the difference is far below one
  tile, so segments are walked as straight lines in tile space.
* The paved / cyclable decision is tag-driven and documented in
  `classify_way()`. Defaults are tuned for France. Use --dump-rejects to audit.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Sequence

from .layout import apply_layout
from .tiles import bbox_arg, cmd_info, cmd_mask, cmd_zone, parse_tile
from .osm import cmd_tiles, cmd_why
from .graph import cmd_snap, parse_latlon
from .db import cmd_dbinfo
from .dem import DEM_RES, cmd_dem, cmd_demdb
from .profile import cmd_profile
from .statshunters import SHARE_FILE, cmd_visited
from .route import cmd_route
from .viewer import cmd_map


_NEGATIVE_VALUE = re.compile(r"^-\d+(\.\d+)?([,\s].*)?$")


def glue_negative_values(argv: Sequence[str]) -> list:
    """
    Let `--clip -5.3,41.2,9.7,51.2` work.

    argparse refuses a value starting with '-' unless it parses as a plain
    negative number, and a comma-separated bbox does not. Rather than make
    every user remember the `--clip=...` form, the value is glued onto the
    option here. Only values that really look like numbers are glued, so a
    genuine missing-argument mistake is still caught by argparse.
    """
    out, i = [], 0
    argv = list(argv)
    while i < len(argv):
        tok = argv[i]
        if (tok.startswith("--") and "=" not in tok and i + 1 < len(argv)
                and _NEGATIVE_VALUE.match(argv[i + 1])):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="tilehunt.py",
        description="VeloViewer z14 explorer-tile coverage from OpenStreetMap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1].split("NOTES")[0] if __doc__ else None,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", metavar="DIR",
                        help="folder holding downloads and .npz datasets "
                             "(default: Data/ next to this script)")
    common.add_argument("--maps-dir", metavar="DIR",
                        help="folder for generated .html "
                             "(default: Maps/ next to this script)")
    common.add_argument("--runs-dir", metavar="DIR",
                        help="folder holding one sub-folder per named ride "
                             "(default: Runs/ next to this script)")

    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("genBDD", parents=[common], aliases=["tiles"],
                       help="build the database: rasterise paved cyclable "
                            "roads onto the z14 grid, and optionally the "
                            "road graph and the heights alongside")
    t.add_argument("--pbf", action="append", help="input .osm.pbf (repeatable)")
    t.add_argument("--geojson", action="append", help="input line GeoJSON (repeatable)")
    t.add_argument("--out", required=True)
    t.add_argument("--idx", default="flex_mem",
                   help="pyosmium node-location index; use "
                        "'sparse_file_array,nodes.idx' if short on RAM")
    t.add_argument("--bbox", type=bbox_arg, help="clip result to W,S,E,N")
    t.add_argument("--pad", type=int, default=0, help="pad the output frame by N tiles")
    t.add_argument("--no-tag-filter", action="store_true",
                   help="for --geojson: accept every line without tag checks")
    t.add_argument("--accept",
                   choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle",
                   help="which ways count. 'paved-cycle' (default): paved roads "
                        "where cycling is allowed. 'all-roads': every road except "
                        "motorways, ignoring surface and access. 'any-highway': "
                        "every highway at all, including tracks and footpaths - "
                        "the right choice for building a scope mask.")
    t.add_argument("--db", metavar="DIR",
                   help="also build the road database used by `route`, so the "
                        "PBF never needs reading again (try Database_OSM)")
    t.add_argument("--no-lengths", action="store_true",
                   help="skip measuring road length per tile (slightly faster, "
                        "but the map then has no density mode)")
    t.add_argument("--no-cobbles", action="store_true",
                   help="treat sett/cobblestone (pave) as unpaved")
    t.add_argument("--trust-tracktype", action="store_true",
                   help="accept track + tracktype=grade1 with no surface tag; "
                        "off by default because grade1 also covers compacted "
                        "hardcore, not just tarmac")
    t.add_argument("--strict-surface", action="store_true",
                   help="require an explicit paved surface tag; ways with no "
                        "surface at all are rejected instead of assumed paved")
    t.add_argument("--dump-rejects", metavar="FILE",
               help="write every rejected way and the rule that rejected it")
    t.add_argument("-v", "--verbose", action="store_true")
    t.add_argument("--dem-db", metavar="DIR",
                   help="build the elevation half of the database in the same "
                        "run, covering the tiles that hold a road: one npz "
                        "per 8x8-tile block, read on demand afterwards")
    t.add_argument("--dem-src", metavar="DIR",
                   help="build the elevation from 1-degree tiles already on "
                        "disk; without it they are read straight out of the "
                        "Copernicus bucket, fetching only the bytes needed")
    t.add_argument("--dem-res", type=int, default=DEM_RES,
                   help=f"elevation samples per explorer tile edge "
                        f"(default {DEM_RES}, about 25 m at 48N)")
    t.add_argument("--dem-force", action="store_true",
                   help="rebuild elevation blocks that are already complete")
    t.set_defaults(func=cmd_tiles)

    m = sub.add_parser("mask", parents=[common], help="which tiles are in scope (land / country)")
    m.add_argument("--boundary",
                   help="polygon GeoJSON (default: countries.geojson in the Data folder)")
    m.add_argument("--iso", help="keep only features with this ISO_A3 (e.g. FRA)")
    m.add_argument("--clip", type=bbox_arg, help="clip to W,S,E,N (drops overseas parts)")
    m.add_argument("--min-part-area", type=float, default=0.0,
                   help="drop polygon parts smaller than this, in square degrees")
    m.add_argument("--out", required=True)
    m.add_argument("-v", "--verbose", action="store_true")
    m.set_defaults(func=cmd_mask)

    g = sub.add_parser("buildmap", parents=[common], aliases=["map"],
                       help="build the self-contained interactive HTML")
    g.add_argument("--tiles", required=True)
    g.add_argument("--mask")
    g.add_argument("--visited", metavar="NPZ",
                   help="the tiles you have already ridden, from genVisited: "
                        "adds a layer, the count still to get, and your max "
                        "square")
    g.add_argument("--out")
    g.add_argument("--boundary", help="polygon GeoJSON for the vector basemap")
    g.add_argument("--iso", default="FRA", help="ISO_A3 to highlight in the basemap")
    g.add_argument("--simplify", type=float, default=0.012,
                   help="basemap simplification tolerance, degrees")
    g.add_argument("--basemap",
                   choices=["none", "osm", "esri-gray", "esri-dark-gray"],
                   default="none",
                   help="raster tiles under the grid. 'none' (default) keeps "
                        "the page fully offline. 'osm' is OpenStreetMap's own "
                        "(rate-limited) servers; add --basemap-grey to "
                        "desaturate them. 'esri-gray' / 'esri-dark-gray' are "
                        "ready-made grey canvases needing no key.")
    g.add_argument("--basemap-url", metavar="TEMPLATE",
                   help="any XYZ tile template, e.g. "
                        "'https://{s}.example.com/{z}/{x}/{y}.png'. Supports "
                        "{s} {z} {x} {y} and {r} (which becomes @2x on retina). "
                        "Use this for a provider needing a key.")
    g.add_argument("--basemap-sub", metavar="LETTERS",
                   help="subdomain letters for {s}, e.g. abc")
    g.add_argument("--basemap-attr", metavar="HTML",
                   help="attribution line for a custom basemap (required by "
                        "most providers' terms)")
    g.add_argument("--basemap-dark", action="store_true",
                   help="tell the viewer the basemap is dark, so labels stay light")
    g.add_argument("--basemap-budget", type=int, default=400,
                   help="most tiles to request for one view (400)")
    g.add_argument("--basemap-everywhere", action="store_true",
                   help="draw the basemap over the whole world; by default it "
                        "is clipped to the mapped country so the surroundings "
                        "keep the plain dark background")
    g.add_argument("--basemap-grey", action="store_true",
                   help="desaturate the raster basemap in the browser; use it "
                        "to grey out --basemap osm")
    g.add_argument("--basemap-opacity", type=float, default=1.0,
                   help="0..1, how strongly the raster basemap shows (1.0)")
    g.add_argument("--basemap-detail", type=float, default=0.6,
                   help="0..1, how strongly the basemap is re-drawn on top of "
                        "the tiles in multiply blending, so roads and town "
                        "shapes stay readable through the colour (0.6; 0 off)")
    g.add_argument("--tile-opacity", type=float, default=None,
                   help="0..1, how opaque the green/red tiles are. Defaults to "
                        "0.8 with no basemap and 0.45 with one, so the map "
                        "underneath stays readable.")
    g.add_argument("--regions", metavar="GEOJSON",
                   help="polygon GeoJSON of regions, drawn as bold outlines")
    g.add_argument("--departments", "--departements", metavar="GEOJSON",
                   dest="departments",
                   help="polygon GeoJSON of departements, drawn as fine outlines")
    g.add_argument("--admin-simplify", type=float, default=0.0022,
                   help="simplification tolerance for the admin outlines, in "
                        "degrees (0.0022 is about 200 m); lower is sharper and "
                        "bigger")
    g.add_argument("--dem-name", default="DEM_DB", metavar="DIR",
                   help="elevation folder named in the commands the plan "
                        "panel hands you (default DEM_DB, the block "
                        "database)")
    g.add_argument("--route-args", default="", metavar="FLAGS",
                   help="extra flags written into the command the plan panel "
                        "hands you. Empty by default, since --cands 6, "
                        "--seconds 80 and --reopt-seconds 30 are now what the "
                        "solver does anyway.")
    g.add_argument("--python-cmd", default="python", metavar="NAME",
                   help="how the commands in the plan panel should invoke "
                        "Python (python)")
    g.add_argument("--db-name", metavar="DIR",
                   help="name of the road database, used only to write the "
                        "ready-made command in the map's plan panel "
                        "(Database_OSM)")
    g.add_argument("--route", metavar="JSON",
                   help="a route written by the `route` command, drawn on top")
    g.add_argument("--clip-to-mask", action="store_true",
                   help="discard road tiles outside --mask and crop the window "
                        "to the mask (keeps the map about one country)")
    g.add_argument("--title", default="Explorer tile coverage")
    g.add_argument("--note", default="")
    g.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="build the map of a named ride: reads "
                        "Runs/<kind>/NAME/NAME.route.json and writes "
                        "NAME.html beside it")
    g.set_defaults(func=cmd_map, run_kind="Square",
                   run_files=(("route", ".route.json"), ("out", ".html")))

    z = sub.add_parser("zone", parents=[common], help="report on a rectangle of tiles")
    z.add_argument("--corners", nargs=2, type=parse_tile, required=True,
                   metavar="X,Y", help="two opposite corner tiles")
    z.add_argument("--tiles", help="dataset, to report green/red")
    z.add_argument("--list-limit", type=int, default=40)
    z.set_defaults(func=cmd_zone)

    y = sub.add_parser("why", parents=[common], help="explain a tile: which ways made it green")
    y.add_argument("--pbf", action="append", required=True)
    y.add_argument("--tile", action="append", type=parse_tile, required=True,
                   metavar="X,Y", help="tile to explain (repeatable)")
    y.add_argument("--accept",
                   choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle")
    y.add_argument("--no-cobbles", action="store_true")
    y.add_argument("--strict-surface", action="store_true")
    y.add_argument("--trust-tracktype", action="store_true")
    y.add_argument("--show-rejected", action="store_true",
                   help="also list the ways that were thrown out, and why")
    y.add_argument("--limit", type=int, default=25)
    y.add_argument("--idx", default="flex_mem")
    y.set_defaults(func=cmd_why)

    # Both routing modes take the same options but for how the area is
    # described, so the options are declared once and shared.
    route_opts = argparse.ArgumentParser(add_help=False)
    route_opts.add_argument("--db", metavar="DIR",
                   help="the road database built by `genBDD --db`; with this, "
                        "no .osm.pbf is read at all")
    route_opts.add_argument("--pbf", action="append",
                   help="OSM extract to build the road graph from, when there "
                        "is no database")
    route_opts.add_argument("--graph-cache", metavar="NPZ",
                   help="reuse (or create) a cached graph for this area; makes "
                        "repeated runs on the same zone instant")
    route_opts.add_argument("--start", type=parse_tile, metavar="X,Y",
                   help="tile to start in (default: the NW corner)")
    route_opts.add_argument("--end", type=parse_tile, metavar="X,Y",
                   help="tile to finish in (default: the SE corner)")
    route_opts.add_argument("--start-at", type=parse_latlon, metavar="LAT,LON",
                   help="exact point to start from, pinned to the nearest "
                        "junction of the network. May sit outside the "
                        "rectangle. Overrides --start.")
    route_opts.add_argument("--end-at", type=parse_latlon, metavar="LAT,LON",
                   help="exact point to finish at. Overrides --end.")
    route_opts.add_argument("--away-max", type=float, default=100.0,
                   metavar="KM",
                   help="refuse an exact endpoint lying more than this far "
                        "OUTSIDE the area covered (100 km). A point inside is "
                        "never refused, however long the ride; the usual "
                        "cause is LAT,LON given the wrong way round")
    route_opts.add_argument("--via", action="append", type=parse_latlon,
                   metavar="LAT,LON",
                   help="a point the ride must go through, pinned like the "
                        "endpoints. Repeat for several. The solver chooses "
                        "the order it rides them in.")
    route_opts.add_argument("--snap-max", type=float, default=150.0, metavar="M",
                   help="refuse to pin an endpoint further than this from the "
                        "point you gave (150 m)")
    route_opts.add_argument("--ignore-oneway", action="store_true",
                   help="treat every road as two-way. Off by default: one-way "
                        "streets are respected, except where OSM says bicycles "
                        "may take the contraflow (oneway:bicycle=no, "
                        "cycleway=opposite*), which is how French double-sens "
                        "cyclable is tagged.")
    route_opts.add_argument("--margin", type=int, default=2,
                   help="tiles of road to load around the rectangle, so the "
                        "ride may leave it when that is shorter (2)")
    route_opts.add_argument("--cands", type=int, default=6,
                   help="candidate junctions per tile (6). More is slower but "
                        "lets the route enter a tile at a better point.")
    route_opts.add_argument("--seconds", type=float, default=80.0,
                   help="time budget for improving the visit order (80)")
    route_opts.add_argument("--prune-seconds", type=float, default=60.0,
                   help="extra budget for dropping redundant visits (60)")
    route_opts.add_argument("--seed", type=int, default=12345,
                   help="random seed for the restarts, so a run is repeatable")
    route_opts.add_argument("--reopt-seconds", type=float, default=30.0,
                   help="budget for re-optimising the visit order once the "
                        "redundant visits have been dropped (30)")
    route_opts.add_argument("--no-progress", action="store_true",
                   help="do not print the running best distance")
    route_opts.add_argument("--no-prune", action="store_true",
                   help="keep one explicit visit per tile; much worse, but "
                        "useful to see what the pruning is worth")
    route_opts.add_argument("--pin-ends", action="store_true",
                   help="force the exact start and end junctions rather than "
                        "letting the solver choose inside those tiles")
    route_opts.add_argument("--require-all", action="store_true",
                   help="stop rather than skip, if any tile of the rectangle "
                        "has no usable road. By default such tiles are "
                        "reported and left out, and the rest is ridden.")
    route_opts.add_argument("--allow-gaps", action="store_true",
                   help=argparse.SUPPRESS)      # now the default; kept so old
                                                # command lines still run
    route_opts.add_argument("--batch", type=int, default=48,
                   help="Dijkstra sources per batch (memory vs speed)")
    route_opts.add_argument("--accept", choices=["paved-cycle", "all-roads", "any-highway"],
                   default="paved-cycle")
    route_opts.add_argument("--no-cobbles", action="store_true")
    route_opts.add_argument("--strict-surface", action="store_true")
    route_opts.add_argument("--trust-tracktype", action="store_true")
    route_opts.add_argument("--idx", default="flex_mem")
    route_opts.add_argument("--dem", metavar="DIR",
                   help="where the heights come from: the block database "
                        "built by `demdb` (or by `genBDD --dem-db`), or a "
                        "folder of 1-degree Copernicus .tif / SRTM .hgt "
                        "tiles. Adds the profile and the climbing totals to "
                        "the json, the gpx and the map. Left out, Data/DEM_DB "
                        "is used when it exists.")
    route_opts.add_argument("--no-dem-fetch", action="store_true",
                   help="never touch the network for heights: use only what "
                        "the database already holds")
    route_opts.add_argument("--no-dem", action="store_true",
                   help="skip the elevation profile entirely")
    route_opts.add_argument("--dem-res", type=int, default=DEM_RES,
                   help=f"samples per explorer tile edge when blocks have to "
                        f"be built (default {DEM_RES})")
    route_opts.add_argument("--dem-src", metavar="DIR",
                   help="build any missing heights from 1-degree tiles on "
                        "disk instead of over the network")
    route_opts.add_argument("--dem-step", type=float, default=25.0,
                   help="spacing in metres at which the track is sampled (25)")
    route_opts.add_argument("--dem-smooth", type=float, default=120.0,
                   help="smoothing window in metres before counting the "
                        "climbs (120). Raw DEM noise invents hundreds of "
                        "metres of ascent on a flat road.")
    route_opts.add_argument("--dem-threshold", type=float, default=3.0,
                   help="metres the profile must turn by before a climb is "
                        "counted (3). Lower it and DEM noise becomes ascent.")
    route_opts.add_argument("--gpx", help="write the ride as a .gpx track")
    route_opts.add_argument("--out", help="write the route as .json for `map --route`")
    route_opts.add_argument("--name", help="track name inside the gpx")
    route_opts.add_argument("-v", "--verbose", action="store_true")
    route_opts.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="name this ride: the gpx and the json are written to "
                        "Runs/Square/NAME/NAME.* and the track is named NAME")

    r = sub.add_parser("compSquareRoute", parents=[common, route_opts],
                       aliases=["route"],
                       help="mode 1 (square): shortest ride covering a "
                            "rectangle of tiles, from a start point to an "
                            "end point")
    r.add_argument("--corners", nargs=2, type=parse_tile, required=True,
                   metavar="X,Y", help="two opposite corner tiles of the rectangle")
    r.set_defaults(func=cmd_route, run_kind="Square", want=None,
                   run_files=(("gpx", ".gpx"), ("out", ".route.json")))

    k = sub.add_parser("compTileSeekerRoute", parents=[common, route_opts],
                       aliases=["seek"],
                       help="mode 3 (seeker): shortest ride collecting a "
                            "chosen set of tiles, from a start point to an "
                            "end point")
    k.add_argument("--want", action="append", type=parse_tile, metavar="X,Y",
                   required=True,
                   help="a tile the ride must go through. Repeat for each "
                        "one. Only the bounding box of these and the two "
                        "ends, widened by --margin, is ever loaded.")
    k.set_defaults(func=cmd_route, run_kind="TileSeeker", corners=None,
                   run_files=(("gpx", ".gpx"), ("out", ".route.json")))

    f = sub.add_parser("profile", parents=[common],
                       help="add the elevation profile to a route already "
                            "solved, without redoing the optimisation")
    f.add_argument("--route", metavar="JSON")
    f.add_argument("--dem", metavar="DIR",
                   help="elevation source (default Data/DEM_DB, built on "
                        "demand)")
    f.add_argument("--dem-res", type=int, default=DEM_RES)
    f.add_argument("--dem-src", metavar="DIR")
    f.add_argument("--no-dem-fetch", action="store_true")
    f.add_argument("--gpx", help="also rewrite the gpx with elevations")
    f.add_argument("--name")
    f.add_argument("--dem-step", type=float, default=25.0)
    f.add_argument("--dem-smooth", type=float, default=120.0)
    f.add_argument("--dem-threshold", type=float, default=3.0)
    f.add_argument("--runname", "--run", dest="runname", metavar="NAME",
                   help="work on a named ride under Runs/<kind>/NAME/")
    f.set_defaults(func=cmd_profile, run_kind="Square",
                   run_files=(("route", ".route.json"), ("gpx", ".gpx")))

    e = sub.add_parser("dem", parents=[common],
                       help="which elevation tiles a zone needs, and a script "
                            "to fetch the missing ones")
    e.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y",
                   help="the two opposite tiles of a square")
    e.add_argument("--margin", type=int, default=2,
                   help="tiles of slack around the square (default 2, the "
                        "same slack the solver is allowed)")
    e.add_argument("--tiles", metavar="NPZ",
                   help="instead of a square, every tile holding a road in a "
                        "coverage npz -- this is the whole-country answer")
    e.add_argument("--dem", metavar="DIR",
                   help="folder already holding elevation tiles; only what is "
                        "missing is listed")
    e.add_argument("--source", choices=("copernicus", "srtm"),
                   default="copernicus",
                   help="which dataset to fetch (default copernicus)")
    e.add_argument("--out", metavar="FILE",
                   help="write a download script (.ps1 for PowerShell, "
                        "anything else for sh) instead of printing URLs")
    e.set_defaults(func=cmd_dem)

    q = sub.add_parser("demdb", parents=[common],
                       help="cut elevation into per-block files, so a ride "
                            "only ever loads what it crosses")
    q.add_argument("--out", required=True, metavar="DIR",
                   help="the database folder, e.g. DEM_DB")
    q.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y",
                   help="build just this square")
    q.add_argument("--margin", type=int, default=2,
                   help="tiles of slack around the square (default 2)")
    q.add_argument("--tiles", metavar="NPZ",
                   help="instead of a square, every tile holding a road in a "
                        "coverage npz -- the whole-country build")
    q.add_argument("--src", metavar="DIR",
                   help="build from 1-degree tiles already on disk; without "
                        "it the heights are read straight out of the "
                        "Copernicus bucket, fetching only the bytes needed")
    q.add_argument("--res", type=int, default=DEM_RES,
                   help=f"samples per explorer tile edge (default {DEM_RES}, "
                        f"about 25 m at 48N; 32 halves the size)")
    q.add_argument("--force", action="store_true",
                   help="rebuild blocks that are already there")
    q.set_defaults(func=cmd_demdb)

    sn = sub.add_parser("snap", parents=[common],
                        help="where a point lands on the road network")
    sn.add_argument("--at", required=True, type=parse_latlon, metavar="LAT,LON")
    sn.add_argument("--db", metavar="DIR")
    sn.add_argument("--pbf")
    sn.add_argument("--pad", type=float, default=0.02, metavar="DEG",
                    help="half-width of the area loaded around the point "
                         "(0.02 degrees, about 2 km)")
    sn.add_argument("--accept", choices=("paved-cycle", "all-roads",
                                         "any-highway"), default="paved-cycle")
    sn.add_argument("--no-cobbles", action="store_true")
    sn.add_argument("--strict-surface", action="store_true")
    sn.add_argument("--trust-tracktype", action="store_true")
    sn.add_argument("--ignore-oneway", action="store_true")
    sn.add_argument("--idx")
    sn.add_argument("-v", "--verbose", action="store_true")
    sn.set_defaults(func=cmd_snap)

    vi = sub.add_parser("genVisited", parents=[common], aliases=["visited"],
                        help="the tiles you have already ridden, from your "
                             "Statshunters share link")
    vi.add_argument("--share", metavar="LINK",
                    help="your share link from https://www.statshunters.com/"
                         "share (or just its code). Remembered afterwards, so "
                         "you only pass it once.")
    vi.add_argument("--out", default="visited.npz",
                    help="where to write the grid (visited.npz)")
    vi.add_argument("--type", metavar="LIST",
                    help="keep only these activity types, comma separated, "
                         "e.g. Ride or Ride,VirtualRide. Default: all.")
    vi.add_argument("--since", metavar="YYYY-MM-DD",
                    help="ignore activities before this date")
    vi.add_argument("--until", metavar="YYYY-MM-DD",
                    help="ignore activities after this date")
    vi.add_argument("--timeout", type=float, default=60.0)
    vi.add_argument("--no-remember", dest="remember", action="store_false",
                    help="do not write the share link to Data/"
                         + SHARE_FILE)
    vi.add_argument("--share-base", help=argparse.SUPPRESS)
    vi.add_argument("-v", "--verbose", action="store_true")
    vi.set_defaults(func=cmd_visited)

    d = sub.add_parser("dbinfo", parents=[common],
                       help="what the road database holds for given tiles")
    d.add_argument("--db", required=True, metavar="DIR")
    d.add_argument("--tile", action="append", type=parse_tile, metavar="X,Y")
    d.add_argument("--corners", nargs=2, type=parse_tile, metavar="X,Y")
    d.set_defaults(func=cmd_dbinfo)

    i = sub.add_parser("info", parents=[common], help="describe a dataset")
    i.add_argument("--top", type=int, default=10,
                   help="how many of the busiest tiles to list (10)")
    i.add_argument("--tiles", required=True)
    i.set_defaults(func=cmd_info)

    a = p.parse_args(glue_negative_values(
        sys.argv[1:] if argv is None else argv))
    apply_layout(a)
    a.func(a)
