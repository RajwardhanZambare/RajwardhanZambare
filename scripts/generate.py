#!/usr/bin/env python3
"""
Generates dark.svg and light.svg for the profile.sh --live banner (Phase 1).

Usage:
    python3 generate.py

Runs, in order:
  1. portrait.py   - crop/dither the profile photo for both themes (data/*.npy)
  2. logos.py      - trace + sample the React/Node/Python reference SVGs
  3. grouping.py   - build + verify the intro-shimmer / drift-band groupings
  4. svg_build.py  - assemble dark.svg and light.svg

The .npy files under data/ are the source of truth for the dot positions and
groupings -- re-running svg_build.build_all() alone (without re-running the
earlier steps) will reuse them as-is.
"""
import os
import time

import portrait
import logos
import grouping
import svg_build

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    print("== Phase 1: profile.sh --live banner ==\n")

    print("-- step 1/4: portrait dithering --")
    portrait.build_portrait_data()

    print("\n-- step 2/4: logo tracing --")
    logos.build_logo_data()

    print("\n-- step 3/4: grouping + verification metrics --")
    grouping.build_grouping_data()

    print("\n-- step 4/4: SVG assembly --")
    paths = svg_build.build_all(outdir=HERE)

    print(f"\ndone in {time.time()-t0:.1f}s")
    for theme, path in paths.items():
        print(f"  {theme}: {path}  ({os.path.getsize(path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
