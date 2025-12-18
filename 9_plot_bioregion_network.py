#!/usr/bin/env python3
"""
Plot a Leiden bioregion network:
- Nodes: grid cells
- Edges: kNN-based similarity between cells (Hellinger / Euclidean)
- Node color: bioregion (Leiden cluster label)

Requires:
- scipy, numpy, pandas, matplotlib
- igraph (python-igraph)

Assumes you have already run:
  - build_hellinger_euclidean.py
  - cluster_bioregions_hellinger.py
for the same OUT_DIR.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# =========================
# CONFIG — EDIT AS NEEDED
# =========================

# Same OUT_DIR as in your grasslands Hellinger / Leiden scripts
OUT_DIR = "/home/eric/GitHub/analysis_OUT/"

# Files produced by your existing pipeline
KNN_GLOB_PATTERN = "knn_k*_c*.npz"          # from build_hellinger_euclidean.py
LABELS_FILENAME  = "bioregion_labels_TUNED.parquet"  # from cluster_bioregions_hellinger.py

# Similarity transform: should match cluster_bioregions_hellinger.py
SIMILARITY_MODE = "adaptive_rbf"   # "adaptive_rbf", "reciprocal", or "exp"
EXP_BETA        = 1.0

# Graph sparsification for plotting
EDGE_MIN_SIM       = 0.0        # drop edges with similarity below this (0 keeps everything)
MAX_EDGES_FOR_PLOT = 100_000    # cap number of edges to draw (largest weights kept)

# Layout and figure
FR_NITER    = 500       # iterations for Fruchterman-Reingold layout
FIGSIZE     = (10, 10)
DPI         = 300
POINT_SIZE  = 12
EDGE_ALPHA  = 0.08
EDGE_WIDTH  = 0.3
OUT_PNG     = "bioregion_network.png"


# =========================
# HELPERS (borrowed logic)
# =========================

def _glob_one(dirpath, pattern):
    paths = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(dirpath, pattern)}")
    # pick the last (usually the one with largest cN etc.)
    return paths[-1]


def _distance_knn_to_similarity(
    A_dist: csr_matrix,
    mode: str = "adaptive_rbf",
    beta: float = 1.0,
    eps_floor: float = 1e-6,
    two_hop_alpha: float = 0.02,
    sym: str = "max",
) -> csr_matrix:
    """
    Convert a kNN distance matrix to a symmetric similarity graph.

    This mirrors the logic in cluster_bioregions_hellinger.py to keep
    clustering and plotting consistent.
    """
    from scipy.sparse import coo_matrix

    A = A_dist.tocoo(copy=True)

    if mode == "adaptive_rbf":
        A_csr = A_dist.tocsr()
        data = A_csr.data

        # local scale s_i = median neighbor distance (fallback to global median)
        row_meds = np.zeros(A_csr.shape[0], dtype=np.float64)
        for i in range(A_csr.shape[0]):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            if end > start:
                row_meds[i] = np.median(data[start:end])
        global_med = np.median(data) if data.size else 1.0
        row_meds[row_meds == 0.0] = global_med

        si = row_meds[A.row]
        sj = row_meds[A.col]
        denom = 2.0 * si * sj
        denom[denom == 0.0] = global_med

        A.data = np.exp(-(A.data ** 2) / denom) + eps_floor

    elif mode == "reciprocal":
        A.data = 1.0 / (1.0 + A.data) + eps_floor

    elif mode == "exp":
        A.data = np.exp(-beta * A.data) + eps_floor

    else:
        raise ValueError(f"Unknown SIMILARITY mode: {mode}")

    A = A.tocsr()

    # Symmetrize
    if sym == "max":
        A = A.maximum(A.T)
    elif sym == "mean":
        A = (A + A.T) * 0.5
    else:
        A = A.minimum(A.T)
    A.eliminate_zeros()

    # Optional: light 2-hop reinforcement to reduce fragmentation
    if two_hop_alpha and two_hop_alpha > 0.0:
        A2 = A @ A
        A2.setdiag(0.0)
        A2.eliminate_zeros()
        A = A + (A2.multiply(two_hop_alpha))
        A = A.maximum(A.T)
        A.eliminate_zeros()

    return A


def _build_layout_and_edges(A_sim: csr_matrix, niter: int = 500):
    """
    Build a Fruchterman–Reingold layout and return:
      - coords: (n_nodes, 2) array of node coordinates
      - edge_i, edge_j, edge_w: arrays of edges (i<j) and weights
    """
    from scipy.sparse import coo_matrix
    import igraph as ig

    coo = A_sim.tocoo()
    n = A_sim.shape[0]

    # Keep only one direction for plotting (i < j)
    mask = coo.row < coo.col
    row = coo.row[mask]
    col = coo.col[mask]
    data = coo.data[mask]

    # Build igraph graph
    g = ig.Graph(n=n)
    edges = list(zip(row.tolist(), col.tolist()))
    g.add_edges(edges)
    g.es["weight"] = data.tolist()

    # Layout – weighted FR
    layout = g.layout_fruchterman_reingold(weights="weight", niter=niter)
    coords = np.array(layout.coords, dtype=float)

    return coords, row, col, data


# =========================
# MAIN
# =========================

def main():
    # Load labels (includes cell index + bioregion column)
    labels_path = os.path.join(OUT_DIR, LABELS_FILENAME)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_parquet(labels_path)
    if "bioregion" not in df.columns:
        raise ValueError("Expected a 'bioregion' column in labels file.")

    bioregions = df["bioregion"].to_numpy(dtype=int)
    n_nodes = len(df)

    # Load kNN distances
    knn_path = _glob_one(OUT_DIR, KNN_GLOB_PATTERN)
    print(f"[LOAD] kNN distances: {knn_path}")
    A_dist = load_npz(knn_path).tocsr()

    if A_dist.shape[0] != n_nodes:
        raise ValueError(
            f"Shape mismatch: kNN matrix has {A_dist.shape[0]} nodes, "
            f"but labels file has {n_nodes} rows."
        )

    # Convert distances to similarities
    print("[GRAPH] Converting kNN distances to similarities…")
    A_sim = _distance_knn_to_similarity(
        A_dist,
        mode=SIMILARITY_MODE,
        beta=EXP_BETA,
        eps_floor=1e-6,
        two_hop_alpha=0.02,
        sym="max",
    )

    # Optional: drop very weak edges
    if EDGE_MIN_SIM > 0.0:
        mask = A_sim.data >= EDGE_MIN_SIM
        A_sim.data[~mask] = 0.0
        A_sim.eliminate_zeros()

    # Possibly downsample edges for plotting
    from scipy.sparse import coo_matrix

    coo = A_sim.tocoo()
    nnz = coo.nnz
    print(f"[GRAPH] Similarity edges before filtering: {nnz}")

    if nnz > MAX_EDGES_FOR_PLOT:
        # Keep the strongest edges
        idx_sort = np.argsort(coo.data)
        keep_idx = idx_sort[-MAX_EDGES_FOR_PLOT:]
        row = coo.row[keep_idx]
        col = coo.col[keep_idx]
        data = coo.data[keep_idx]
        A_sim = coo_matrix((data, (row, col)), shape=A_sim.shape).tocsr()
        A_sim = A_sim.maximum(A_sim.T)
        A_sim.eliminate_zeros()
        print(f"[GRAPH] Edges trimmed to top {MAX_EDGES_FOR_PLOT} by similarity.")

    # Compute layout and extract edges for plotting
    print("[LAYOUT] Computing Fruchterman–Reingold layout…")
    coords, e_i, e_j, e_w = _build_layout_and_edges(A_sim, niter=FR_NITER)
    print(f"[INFO] Nodes: {n_nodes}, plotted edges: {len(e_i)}")

    # ---------------------------------------
    # EXPAND HAIRBALL TO FILL PLOT AREA
    # ---------------------------------------

    # Optionally: clamp extreme outliers
    low_x, high_x = np.quantile(coords[:, 0], [0.005, 0.995])
    low_y, high_y = np.quantile(coords[:, 1], [0.005, 0.995])

    coords[:, 0] = np.clip(coords[:, 0], low_x, high_x)
    coords[:, 1] = np.clip(coords[:, 1], low_y, high_y)

    # Normalize to 0–1 range
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

    coords[:, 0] = (coords[:, 0] - min_x) / (max_x - min_x)
    coords[:, 1] = (coords[:, 1] - min_y) / (max_y - min_y)


    # =========================
    # PLOT
    # =========================
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Build edge segments
    segments = np.stack(
        [np.column_stack([coords[e_i, 0], coords[e_j, 0]]),
         np.column_stack([coords[e_i, 1], coords[e_j, 1]])],
        axis=-1,
    )
    # segments shape: (n_edges, 2, 2) -> [[(x_i, x_j), (y_i, y_j)], ...]

    lc = LineCollection(
        segments,
        linewidths=EDGE_WIDTH,
        alpha=EDGE_ALPHA,
        color="#cccccc",
        zorder=1,
    )
    ax.add_collection(lc)

    # Node colors by bioregion – match map_bioregions_quick.py
    clusters = df["bioregion"].to_numpy()
    unique = np.unique(clusters)
    # compact index 0..K-1
    remap = {c: i for i, c in enumerate(unique)}
    idx = np.array([remap[c] for c in clusters], dtype=int)

    # use the same categorical map as the Sweden map
    cmap = plt.get_cmap("tab20")
    node_colors = cmap(idx % cmap.N)

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=POINT_SIZE,
        c=node_colors,
        edgecolors="none",
        zorder=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] Network plot saved to: {out_path}")


if __name__ == "__main__":
    main()
