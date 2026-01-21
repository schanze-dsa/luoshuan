#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML-only viewer wrapper (Plotly/WebGL).

This entrypoint forwards to visualize_contact_tightening_plotly.py and no longer
produces PNG/GIF outputs.
"""
from __future__ import annotations

import os
import sys

TOOLS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from visualize_contact_tightening_plotly import main


if __name__ == "__main__":
    print("[info] Using Plotly HTML viewer (PNG/GIF outputs disabled).")
    main()
