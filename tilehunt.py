#!/usr/bin/env python3
"""
Kept so that `python tilehunt.py <command> ...` works exactly as before.

The code itself now lives in the tilehunt/ package beside this file; this is
the same as running `python -m tilehunt <command> ...`.
"""
import os
import sys

# `python -I tilehunt.py` (isolated mode) leaves this folder off the import
# path; the single-file version did not care, so the shim must not either.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tilehunt.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
