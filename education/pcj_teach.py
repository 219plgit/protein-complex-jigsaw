#!/usr/bin/env python
"""Wrapper kept for existing instructions. The code is in the installed package:
`pcjigsaw-teach pack MODEL OUTDIR` (or `python -m pcjigsaw.education pack ...`)."""
import os
import sys

try:
    from pcjigsaw.education.cli import main
except ImportError:                      # checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from pcjigsaw.education.cli import main

if __name__ == "__main__":
    main(["pack"] + sys.argv[1:])
