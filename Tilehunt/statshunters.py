"""The tiles you have already ridden, from a Statshunters share link."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time

from . import layout


# --------------------------------------------------------------------------- #
# Statshunters: the tiles you have already ridden
#
# There is no OAuth for third parties, so there is no "sign in with
# Statshunters" to build. What Statshunters offers instead is a share link the
# user creates themselves, which exposes their activities as JSON. Every tool
# in this ecosystem works that way. The link is a secret -- whoever holds it
# can read the whole history -- so it is kept in a file of its own rather than
# typed into a command line that lands in a shell history.
# --------------------------------------------------------------------------- #

SHARE_RE = re.compile(r"^https?://(?:www\.)?statshunters\.com/share/(\w+)/?$")
SHARE_FILE = "statshunters.txt"


def share_url(value: str, base: str | None = None) -> str:
    """
    Accept a full share link or just its code, and return the link.

    `base` exists so the client can be pointed at a local stand-in in the
    tests; in normal use it is Statshunters itself. Refusing anything else by
    default is deliberate: a share link is a secret, and a typo that sent it
    to the wrong host would hand over the whole history.
    """
    v = str(value).strip().strip('"').strip("'")
    host = (base or "https://www.statshunters.com").rstrip("/")
    if base:
        m = re.match(r"^https?://[^/]+/share/(\w+)/?$", v)
        if m:
            return f"{host}/share/{m.group(1)}"
    m = SHARE_RE.match(v)
    if m:
        return f"{host}/share/{m.group(1)}"
    if re.fullmatch(r"\w{6,}", v):
        return f"{host}/share/{v}"
    raise SystemExit(
        f"{value!r} is not a Statshunters share link. Create one at "
        f"https://www.statshunters.com/share -- it looks like "
        f"https://www.statshunters.com/share/abcdef123456")


def remembered_share() -> str | None:
    path = layout.DATA_DIR / SHARE_FILE
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def remember_share(link: str) -> None:
    path = layout.DATA_DIR / SHARE_FILE
    path.write_text(
        "# Your Statshunters share link. Anyone holding it can read your\n"
        "# whole activity history, so keep this file out of git. Revoke or\n"
        "# replace it at https://www.statshunters.com/share\n"
        f"{link}\n", encoding="utf-8")
    print(f"remembered  : {path}  (keep it out of git)")


def fetch_activities(link: str, timeout: float = 60.0,
                     verbose: bool = False):
    """
    Every activity behind a share link, page by page.

    The endpoint is the share link with /api/activities appended and a page
    number; it answers with {"activities": [...]} and an empty list once there
    is nothing left. Pages hold 500 activities.
    """
    import urllib.error
    import urllib.request

    base = link.rstrip("/") + "/api/activities"
    seen = 0
    for page in itertools.count(1):
        url = f"{base}?page={page}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "tilehunt/1.0 (explorer tile routing)",
                          "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SystemExit(
                    f"Statshunters answered 404 for {url}\n"
                    f"The share link is wrong, or it has been revoked. "
                    f"Create a new one at https://www.statshunters.com/share")
            raise SystemExit(f"Statshunters answered {exc.code} for {url}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"could not reach Statshunters: {exc.reason}")
        except json.JSONDecodeError:
            raise SystemExit(
                f"{url} did not answer with JSON. If you pasted the link in a "
                f"browser and saw your page, the link is fine and the API "
                f"path may have changed.")

        acts = payload.get("activities")
        if acts is None:
            raise SystemExit(f"unexpected answer from {url}: no 'activities'")
        if not acts:
            return
        seen += len(acts)
        if verbose:
            print(f"   page {page}: {len(acts)} activities "
                  f"({seen:,} so far)", flush=True)
        yield from acts


def cmd_visited(a: argparse.Namespace) -> None:
    """Pull the tiles you have already ridden out of Statshunters."""
    from .tiles import TileGrid, largest_square
    link = (share_url(a.share, a.share_base) if a.share
            else remembered_share())
    if not link:
        raise SystemExit(
            "no share link given and none remembered.\n"
            "Create one at https://www.statshunters.com/share, then:\n"
            "  python tilehunt.py genVisited --share "
            "https://www.statshunters.com/share/YOURCODE")
    print(f"share link  : {link[:38]}...  (kept out of this output)")

    want_types = {t.strip().lower() for t in a.type.split(",")} if a.type else None
    tiles: set[tuple[int, int]] = set()
    kept = skipped = 0
    oldest = newest = None
    for act in fetch_activities(link, a.timeout, a.verbose):
        when = str(act.get("date") or act.get("start_date") or "")[:10]
        if a.since and when and when < a.since:
            skipped += 1
            continue
        if a.until and when and when > a.until:
            skipped += 1
            continue
        if want_types and str(act.get("type", "")).lower() not in want_types:
            skipped += 1
            continue
        kept += 1
        if when:
            oldest = when if oldest is None else min(oldest, when)
            newest = when if newest is None else max(newest, when)
        for t in act.get("tiles") or ():
            try:
                tiles.add((int(t["x"]), int(t["y"])))
            except (KeyError, TypeError, ValueError):
                continue

    if not tiles:
        raise SystemExit(
            "no tiles came back. If you filtered by --type or by date, try "
            "without; otherwise the account may have no processed activities "
            "yet (Statshunters imports in the background).")

    print(f"activities  : {kept:,} kept"
          + (f", {skipped:,} filtered out" if skipped else ""))
    if oldest:
        print(f"period      : {oldest} to {newest}")
    grid = TileGrid.from_tiles(sorted(tiles), meta={
        "source": "statshunters", "activities": kept,
        "from": oldest, "to": newest,
        "type": a.type or "all",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S")})
    side, sx_, sy_ = largest_square(grid.bits)
    print(f"tiles       : {len(tiles):,} visited, "
          f"spanning x {grid.x0}..{grid.x0+grid.w-1}  "
          f"y {grid.y0}..{grid.y0+grid.h-1}")
    print(f"max square  : {side} x {side}  at "
          f"{grid.x0+sx_},{grid.y0+sy_} .. "
          f"{grid.x0+sx_+side-1},{grid.y0+sy_+side-1}")
    grid.meta["max_square"] = [int(side), int(grid.x0 + sx_),
                               int(grid.y0 + sy_)]
    grid.save(a.out)
    print(f"written     : {a.out}")
    if a.share and a.remember:
        remember_share(link)
    print(f"use it with : --visited {os.path.basename(a.out)}")
