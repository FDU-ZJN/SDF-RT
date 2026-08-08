#!/usr/bin/env python3
"""Compatibility entry point for the modular SDF-RT PYNQ runtime."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# This entry point has the same name as the installed PYNQ package. Load the
# vendor package before restoring the deployment directory for local imports.
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
__import__("pynq")
sys.path.insert(0, str(SCRIPT_DIR))

from sdf_rt.cli import main


if __name__ == "__main__":
    main()
