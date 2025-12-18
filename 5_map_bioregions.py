# map_bioregions_quick.py
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# ----------------------
# CONFIG (edit as needed)
# ----------------------
OUT_DIR = "/home/eric/GitHub/analysis_OUT/"
LABEL_FILE = "/home/eric/GitHub/analysis_OUT/bioregion_labels_TUNED.parquet"  # produced by cluster_bioregions.py
PNG_OUT = "/home/eric/GitHub/analysis_OUT/bioregions_quickmap.png"

# IMPORTANT: must match the grid used in your Hellinger build
GRID_SIZE_M = 10_000
# ----------------------

def _rect_from_gxgy(gx, gy, cell):
    """
    Return a matplotlib Rectangle for the grid square (gx,gy),
    where gx,gy are integer cell indices in SWEREF 99 TM / meters.
    Lower-left corner is (gx*cell, gy*cell).
    """
    x0 = gx * cell
    y0 = gy * cell
    return Rectangle((x0, y0), cell, cell)

def _auto_extent(gx, gy, cell):
    xmin = (gx.min()) * cell
    xmax = (gx.max() + 1) * cell
    ymin = (gy.min()) * cell
    ymax = (gy.max() + 1) * cell
    return xmin, xmax, ymin, ymax

def main():
    labels_path = os.path.join(OUT_DIR, LABEL_FILE)
    df = pd.read_parquet(labels_path)

    # We expect columns: gx, gy, SWEREF_*_centroid, bioregion
    needed = {"gx", "gy", "bioregion"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {labels_path}: {missing}")

    # Build rectangles for each cell
    rects = [ _rect_from_gxgy(int(gx), int(gy), GRID_SIZE_M) for gx,gy in zip(df["gx"], df["gy"]) ]
    patches = PatchCollection(rects, match_original=True, linewidths=0.1)

    # Color by bioregion
    # We map discrete clusters to a categorical colormap
    clusters = df["bioregion"].to_numpy()
    unique = np.unique(clusters)
    # Build a compact index 0..K-1 for colormap
    remap = {c:i for i,c in enumerate(unique)}
    idx = np.array([remap[c] for c in clusters], dtype=int)

    cmap = plt.get_cmap("tab20")  # categorical
    colors = cmap(idx % cmap.N)
    patches.set_facecolor(colors)
    patches.set_edgecolor("k")
    patches.set_linewidth(0.15)
    patches.set_alpha(0.9)

    # Make figure
    fig_w = 8
    # Aspect according to data extent
    gx = df["gx"].to_numpy()
    gy = df["gy"].to_numpy()
    xmin, xmax, ymin, ymax = _auto_extent(gx, gy, GRID_SIZE_M)
    aspect = (xmax - xmin) / max(1.0, (ymax - ymin))
    fig_h = max(6.0, fig_w / max(0.1, aspect))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.add_collection(patches)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    # Clean look
    ax.set_xlabel("SWEREF 99 TM Easting (m)")
    ax.set_ylabel("SWEREF 99 TM Northing (m)")
    ax.set_title("Proposed Grassland Bioregions (cluster labels)")

    # Legend: show up to 20 cluster samples (tab20 cycles colors)
    # Build a simple proxy legend using small squares
    import matplotlib.patches as mpatches
    handles = []
    for c in unique[:20]:
        i = remap[c]
        color = cmap(i % cmap.N)
        handles.append(mpatches.Patch(color=color, label=f"Cluster {c}"))
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8, title="Bioregions")

    # Gridlines every 100 km to aid orientation (optional)
    step = 100_000
    x_ticks = np.arange(math.floor(xmin/step)*step, math.ceil(xmax/step)*step + 1, step)
    y_ticks = np.arange(math.floor(ymin/step)*step, math.ceil(ymax/step)*step + 1, step)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(alpha=0.2, linewidth=0.5)

    out_path = os.path.join(OUT_DIR, PNG_OUT)
    plt.tight_layout()
    fig.savefig(out_path)
    print(f"[SAVE] Map -> {out_path}")

if __name__ == "__main__":
    main()
