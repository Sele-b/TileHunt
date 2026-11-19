"""
tilehunt -- explorer-tile routing for the VeloViewer / Statshunters
square challenge.

The usage guide, with every command, is the docstring of Tilehunt.cli
(and what `python -m Tilehunt --help` prints).
"""
from __future__ import annotations

# Everything the single-file tilehunt.py exposed is still reachable as
# Tilehunt.<name>, so `import Tilehunt as t` keeps working unchanged.
from .layout import (  # noqa: F401
    ROOT,
    SAFE_NAME,
    apply_layout,
    default_input,
    find_input,
    find_run,
    place_output,
    run_dir,
    run_file,
    run_name,
)
from .tiles import (  # noqa: F401
    EARTH_CIRCUMFERENCE_M,
    MAX_LAT,
    NTILES,
    TileGrid,
    ZOOM,
    _lonlat_to_tile,
    _tile_to_lonlat,
    bbox_arg,
    cmd_info,
    cmd_mask,
    cmd_zone,
    haversine_m,
    largest_square,
    lonlat_to_tilef,
    parse_tile,
    tile_bounds,
    tile_size_m,
    tilef_to_lonlat,
    walk_polyline,
    walk_segment,
    walk_segment_frac,
)
from .osm import (  # noqa: F401
    ACCESS_NEGATIVE,
    ASSUME_PAVED,
    ASSUME_UNPAVED,
    BAD_SMOOTHNESS,
    BIKE_NEGATIVE,
    BIKE_POSITIVE,
    COBBLE_SURFACES,
    HIGHWAY_EXCLUDED,
    HIGHWAY_NEEDS_BIKE_YES,
    PAVED_SURFACES,
    ROAD_CLASSES,
    SERVICE_EXCLUDED,
    UNPAVED_SURFACES,
    classify_all_roads,
    classify_any_highway,
    classify_way,
    cmd_tiles,
    cmd_why,
    pick_classifier,
    read_ways_geojson,
    read_ways_pbf,
)
from .graph import (  # noqa: F401
    PathCache,
    RoadGraph,
    bike_directions,
    build_graph,
    cmd_snap,
    components,
    csr_min_duplicates,
    largest_component,
    many_to_many,
    nodes_in_tile,
    parse_latlon,
    pick_candidates,
    ridable_mask,
    seed_in_tile,
    snap_to_graph,
)
from .db import (  # noqa: F401
    BLOCK_ZOOM,
    DBWriter,
    FLAG_DOC,
    F_BWD,
    F_COBBLE,
    F_FWD,
    F_PAVED,
    F_ROAD,
    F_SURFACE,
    F_TRACK1,
    SHARD_ZOOM,
    block_of,
    blocks_for_bbox,
    cmd_dbinfo,
    db_flags,
    flag_mask,
    load_db_graph,
)
from .solver import (  # noqa: F401
    IDEAL_PER_TILE,
    Progress,
    choose_candidates,
    cluster_matrix,
    covered_tiles,
    double_bridge,
    iterated_local_search,
    nn_path,
    prune_redundant,
    route_length_m,
    theoretical_min_m,
    tile_side_m,
    trace_route,
    two_opt,
)
from .dem import (  # noqa: F401
    DEM,
    DEMDB,
    DEM_INDEX,
    DEM_RES,
    DEM_VOID16,
    LocalSource,
    RemoteCopernicus,
    SRTM_VOID,
    _Grid,
    _dem_block_cells,
    _dem_cells_of_block,
    _dem_complete,
    _dem_read_block,
    cmd_dem,
    cmd_demdb,
    copernicus_name,
    copernicus_url,
    default_dem,
    dem_block_bbox,
    dem_block_name,
    dem_block_of,
    dem_blocks_for_tiles,
    dem_build,
    dem_cells_for_tiles,
    ensure_dem,
    hgt_name,
    hgt_url,
    open_dem,
)
from .profile import (  # noqa: F401
    ascent_descent,
    build_profile,
    cmd_profile,
    profile_from_coords,
    resample_track,
    smooth,
    write_gpx,
)
from .statshunters import (  # noqa: F401
    SHARE_FILE,
    SHARE_RE,
    cmd_visited,
    fetch_activities,
    remember_share,
    remembered_share,
    share_url,
)
from .route import (  # noqa: F401
    cmd_route,
)
from .viewer import (  # noqa: F401
    BASEMAPS,
    ESRI_ATTR,
    FRENCH_CITIES,
    OSM_ATTR,
    VIEWER_HTML,
    basemap_spec,
    build_admin,
    build_basemap,
    cmd_map,
    delta_ints,
    douglas_peucker,
)
from .cli import main, glue_negative_values  # noqa: F401


def __getattr__(name):
    # Data/, Maps/ and Runs/ can be moved at run time by --data-dir and its
    # siblings, so they are looked up live rather than copied here once.
    if name in ("DATA_DIR", "MAPS_DIR", "RUNS_DIR"):
        from . import layout
        return getattr(layout, name)
    raise AttributeError(f"module 'Tilehunt' has no attribute {name!r}")
