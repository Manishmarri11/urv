"""Generates assets/terrain_heightmap.png, the cliff-path heightmap used by
cartpole_world_dual_camera.xml. Reuses generate_cliff_path.py from the inspo
codebase (URV-Summer-2026/cartpole_video/event, expected as a sibling of this
project's parent folder), run dead-straight and transposed so the path runs
along world x instead of the inspo's winding x/y path.

The output PNG is already committed under assets/, so you only need to
re-run this if you want to change the terrain's shape/scale:
    python generate_terrain.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "..", "URV-Summer-2026", "cartpole_video", "event"))
from generate_cliff_path import generate_cliff_heatmap, save_heatmap_to_png  # noqa: E402

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "assets", "terrain_heightmap.png")


def main():
    # width/height here are in the *pre-transpose* image, where the path travels
    # along the "height" dimension. cliff-width/radius are in pixels; the physical
    # size is set separately by the hfield geom's `size` attribute in the XML.
    heatmap = generate_cliff_heatmap(
        width=300,
        height=1200,
        radius=120.0,
        cliff_width=60.0,
        min_length=200.0,
        max_length=200.0,
        min_angle=0.0,
        max_angle=0.0,  # dead straight along x, the cart's long travel axis (slider_x range +/-2.4).
        # The path's fixed width in y (flat plateau ~+/-1.6) already covers slider_y's
        # smaller +/-1.2 range with margin, so a winding path isn't needed for that axis.
        seed=0,
    )
    heatmap = heatmap.T  # path now runs along columns (world x) instead of rows (world y)
    save_heatmap_to_png(heatmap, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} ({heatmap.shape[1]}x{heatmap.shape[0]}, ncol x nrow)")


if __name__ == "__main__":
    main()
