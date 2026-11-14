#!/usr/bin/env python
"""Wrapper kept for existing instructions. The code is in the installed package:
`pcjigsaw-teach vamp2 OUTDIR` (or `python -m pcjigsaw.education vamp2 OUTDIR`)."""
import os
import sys

try:
    from pcjigsaw.education.cli import main
except ImportError:                      # checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from pcjigsaw.education.cli import main

if __name__ == "__main__":
    main(["vamp2"] + sys.argv[1:])
