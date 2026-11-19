"""Where things live: Data/, Maps/, Runs/<kind>/<name>/, and how bare file names are resolved against them."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


# the folder holding the tilehunt/ package, not the package itself:
# Data/, Maps/ and Runs/ belong next to the code, not inside it
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
MAPS_DIR = ROOT / "Maps"
RUNS_DIR = ROOT / "Runs"      # Runs/<kind>/<name>/, named by --runname


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def run_name(name: str) -> str:
    """
    A --runname value fit to be a folder name, on any filesystem.

    Accents are transliterated rather than struck out, so "tracé d'été" comes
    back as "trace-d-ete" and not as a row of dashes.
    """
    import unicodedata
    folded = unicodedata.normalize("NFKD", str(name).strip())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    clean = SAFE_NAME.sub("-", folded).strip("-._")
    if not clean:
        raise SystemExit(f"--runname {name!r} leaves nothing usable as a folder "
                         f"name; use letters, digits, dash or underscore")
    if len(clean) > 64:
        clean = clean[:64].rstrip("-._")
    return clean


def find_run(name: str) -> Path | None:
    """An existing Runs/<kind>/<name>/ folder, whatever kind made it."""
    n = run_name(name)
    if not RUNS_DIR.is_dir():
        return None
    for kind in sorted(RUNS_DIR.iterdir()):
        if kind.is_dir() and (kind / n).is_dir():
            return kind / n
    return None


def run_dir(name: str, kind: str = "Square") -> Path:
    """
    Runs/<kind>/<name>/, created on demand.

    The kind groups rides by how they were produced -- Square today, other
    modes later -- so the folder does not become a flat pile. A ride already
    filed under some kind is found there rather than duplicated: that is what
    lets `buildmap --runname X` work without being told which mode made X.
    """
    d = find_run(name) or (RUNS_DIR / kind / run_name(name))
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_file(name: str, suffix: str, kind: str = "Square") -> str:
    """Runs/<kind>/<name>/<name><suffix>."""
    n = run_name(name)
    return str(run_dir(n, kind) / f"{n}{suffix}")


def find_input(p: str) -> str:
    """Resolve an input path, falling back to the Data folder."""
    if not p:
        return p
    cand = Path(p).expanduser()
    if cand.exists():
        return str(cand)
    alt = DATA_DIR / cand.name
    if alt.exists():
        return str(alt)
    return str(cand)          # let the caller raise a clear error


def place_output(p: str) -> str:
    """Resolve an output path: bare .html -> Maps/, bare anything else -> Data/."""
    if not p:
        return p
    cand = Path(p).expanduser()
    if cand.parent != Path("."):
        cand.parent.mkdir(parents=True, exist_ok=True)
        return str(cand)
    target = MAPS_DIR if cand.suffix.lower() in (".html", ".htm") else DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return str(target / cand.name)


def default_input(*names: str) -> str | None:
    """First of these conventional filenames that exists in the Data folder."""
    for n in names:
        p = DATA_DIR / n
        if p.exists():
            return str(p)
    return None


def apply_layout(a: argparse.Namespace) -> None:
    """Rewrite the namespace's path fields according to the layout above."""
    global DATA_DIR, MAPS_DIR, RUNS_DIR
    if getattr(a, "data_dir", None):
        DATA_DIR = Path(a.data_dir).expanduser().resolve()
    if getattr(a, "maps_dir", None):
        MAPS_DIR = Path(a.maps_dir).expanduser().resolve()
    if getattr(a, "runs_dir", None):
        RUNS_DIR = Path(a.runs_dir).expanduser().resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # --runname gathers one ride's outputs under Runs/<name>/, all sharing
    # the name. Anything named explicitly still wins, so it is a shorthand and
    # never a constraint.
    name = getattr(a, "runname", None)
    if name:
        a.runname = run_name(name)
        kind = getattr(a, "run_kind", "Square")
        for field, suffix in (getattr(a, "run_files", ()) or ()):
            if not getattr(a, field, None):
                setattr(a, field, run_file(a.runname, suffix, kind))
        if hasattr(a, "name") and getattr(a, "name", None) is None:
            a.name = a.runname

    for field in ("pbf", "geojson"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, [find_input(x) for x in v])
    for field in ("boundary", "regions", "departments", "tiles", "mask",
                  "visited"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, find_input(v))
    for field in ("db", "dem", "src", "dem_db", "dem_src"):
        v = getattr(a, field, None)
        if v and not os.path.isabs(v) and os.sep not in v:
            setattr(a, field, str(DATA_DIR / v))
    for field in ("graph_cache", "gpx", "route"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, find_input(v) if field == "route"
                    else place_output(v))
    for field in ("out", "dump_rejects"):
        v = getattr(a, field, None)
        if v:
            setattr(a, field, place_output(v))
