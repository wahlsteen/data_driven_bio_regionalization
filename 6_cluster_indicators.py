#!/usr/bin/env python3

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, coo_matrix
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# =========================
# CONFIG — EDIT HERE
# =========================
@dataclass
class Config:
    # Inputs from your two pipelines
    HELL_DIR: str = "/home/eric/GitHub/analysis_OUT/"      # from build_hellinger_euclidean.py
    LABELS_PATH: str = "/home/eric/GitHub/analysis_OUT/bioregion_labels_TUNED.parquet"  # from cluster_bioregions_hellinger.py

    HELL_FEATS_GLOB: str = "hellinger_features_c*_f*.npz"              # CSR (cells × species)
    CELL_INDEX_GLOB: str = "cell_index_*.parquet"                      # rows aligned to HELL CSR

    # Optional: path to species names (one column `species`), or a Parquet/CSV
    # with column `species` in the same order as the matrix columns.
    SPECIES_NAMES_PATH = "/home/eric/GitHub/OUT/locality_species_presence_species_index_f9105.parquet"

    MIN_GLOBAL_PREVALENCE: float = 0.05   # skip species present in <Z% of cells
    MIN_DELTA_EARLY: float = 0.00001           # skip M-W test if delta <= this

    # Filtering / stats
    Q_FDR_THRESH: float = 0.05
    TOP_N: int = 40
    MIN_DELTA: float = 0.0        # require positive delta; keep 0.0 for inclusive

    # Output
    OUT_DIR: str = "/home/eric/GitHub/analysis_OUT/"

    # Plots
    MAKE_PLOTS: bool = True
    MAX_STRIP_PLOTS_PER_CLUSTER: int = 12

CFG = Config()
# =========================


# ---------- Utilities ----------

def _glob_one(dirpath: str, pattern: str) -> str:
    import glob
    paths = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(dirpath, pattern)}")
    return paths[-1]


def _read_species_names(path: Optional[str], p: int) -> List[str]:
    if path is None:
        return [f"sp_{i:05d}" for i in range(p)]
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        s = pd.read_parquet(path)
    elif ext in (".csv", ".tsv", ".txt"):
        sep = "," if ext == ".csv" else None
        s = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported species name file: {path}")
    if "species" not in s.columns:
        # use first column if unnamed
        col0 = s.columns[0]
        s = s.rename(columns={col0: "species"})
    vals = s["species"].astype(str).tolist()
    if len(vals) != p:
        print(f"[WARN] species name count ({len(vals)}) != matrix columns ({p}); using synthesized names.")
        return [f"sp_{i:05d}" for i in range(p)]
    return vals


def to_dense_col(X: csr_matrix, j: int) -> np.ndarray:
    col = X.getcol(j)
    arr = np.zeros(X.shape[0], dtype=np.float32)
    arr[col.indices] = col.data
    return arr


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    q_vals = np.empty_like(q)
    q_vals[order] = q
    return np.clip(q_vals, 0.0, 1.0)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if sp2 <= 0:
        return np.nan
    return (mx - my) / math.sqrt(sp2)


# ---------- Sparse summaries (fast) ----------

def cluster_summaries(X: csr_matrix, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return per-cluster means and prevalence for all species.
    Uses a sparse assignment matrix S so it's vectorized over species.
    """
    labs = labels.astype(int)
    valid = labs >= 0
    labs = labs[valid]
    uniq = np.unique(labs)
    k = uniq.size

    # Build S (k × N)
    row_idx = np.fromiter((np.where(uniq == c)[0][0] for c in labs), dtype=np.int64, count=labs.size)
    col_idx = np.arange(labels.size, dtype=np.int64)[valid]
    data = np.ones(labs.size, dtype=np.float32)
    S = coo_matrix((data, (row_idx, col_idx)), shape=(k, labels.size)).tocsr()

    # Means: (S @ X) / counts
    counts = np.asarray(S.sum(axis=1)).ravel().astype(np.float32)
    counts[counts == 0] = 1.0
    sums = (S @ X).tocsr()          # k × p
    # divide rows by counts (in place, preserving sparsity)
    for i in range(k):
        start, end = sums.indptr[i], sums.indptr[i+1]
        if start < end:
            sums.data[start:end] /= counts[i]
    means = sums  # k × p

    # Prevalence: binarize X then average
    X_bin = X.copy().tocsr()
    X_bin.data = np.ones_like(X_bin.data, dtype=np.float32)
    prev_counts = (S @ X_bin).toarray().astype(np.float32)  # k × p dense small-ish
    prevalence = prev_counts / counts.reshape(-1, 1)

    # Compute overall means per species as well (for mean_out later)
    overall_mean = np.asarray(X.mean(axis=0)).ravel()

    meta = pd.DataFrame({"cluster": uniq, "n_cells": (labels[labels >= 0].size / k)})
    return meta, means, prevalence


# ---------- Main marker scan ----------

def scan_markers(X: csr_matrix, labels: np.ndarray, species_names: List[str], outdir: str,
                 q_thresh: float, top_n: int, min_delta: float, make_plots: bool,
                 max_strip: int) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:

    os.makedirs(outdir, exist_ok=True)

    clusters = np.unique(labels[labels >= 0])
    k = clusters.size
    n_cells, p = X.shape

    # ---- Precompute per-cluster means & prevalence via sparse algebra (fast, vectorized)
    _, means_sparse, prevalence = cluster_summaries(X, labels)
    means = means_sparse.toarray()  # shape k × p

    # ---- Global prevalence (to skip ultra-rare spp quickly)
    X_bin = X.copy().tocsr()
    X_bin.data = np.ones_like(X_bin.data, dtype=np.float32)
    global_prev = np.asarray(X_bin.mean(axis=0)).ravel()  # shape p

    # ---- Build masks once
    masks = [(labels == cl) for cl in clusters]
    mask_outs = [~m for m in masks]
    n_in_list = [int(m.sum()) for m in masks]
    n_out_list = [int(mo.sum()) for mo in mask_outs]

    # ---- Allocate result holders
    all_p = np.ones((k, p), dtype=float)
    all_d = np.full((k, p), np.nan, dtype=float)
    mean_out_mat = np.zeros((k, p), dtype=float)
    prev_out_mat = np.zeros((k, p), dtype=float)

    # ---- Compute mean_out and prev_out per cluster (vectorized)
    for i, mo in enumerate(mask_outs):
        mean_out_mat[i, :] = np.asarray(X[mo].mean(axis=0)).ravel()
        prev_out_mat[i, :] = np.asarray(X_bin[mo].mean(axis=0)).ravel()

    # ---- Optional fast skip by global prevalence
    valid_species = np.where(global_prev >= CFG.MIN_GLOBAL_PREVALENCE)[0]
    # Species with lower prevalence will keep default p=1, d=nan

    # ---- Main loop: extract each column ONCE and reuse across clusters
    for j in valid_species:
        v = to_dense_col(X, j)   # full column (length = n_cells)
        for i in range(k):
            delta = means[i, j] - mean_out_mat[i, j]
            if delta <= CFG.MIN_DELTA_EARLY:
                all_p[i, j] = 1.0
                all_d[i, j] = np.nan
                continue
            xi = v[masks[i]]
            xo = v[mask_outs[i]]
            # two-sided then fold to one-sided (robust across SciPy versions)
            try:
                _, p2 = mannwhitneyu(xi, xo, alternative="two-sided")
                p = p2/2.0 if delta > 0 else 1.0 - p2/2.0
            except Exception:
                p = np.nan
            all_p[i, j] = p
            # Cohen's d
            nx, ny = xi.size, xo.size
            if nx >= 2 and ny >= 2:
                mx, my = xi.mean(), xo.mean()
                vx, vy = xi.var(ddof=1), xo.var(ddof=1)
                sp2 = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
                all_d[i, j] = (mx - my) / math.sqrt(sp2) if sp2 > 0 else np.nan
            else:
                all_d[i, j] = np.nan

    # ---- Build output frames per cluster
    all_rows = []
    per_cluster_top = []
    for i, cl in enumerate(clusters):
        qvals = benjamini_hochberg(np.nan_to_num(all_p[i], nan=1.0))
        df = pd.DataFrame({
            "cluster": cl,
            "species": species_names,
            "mean_in": means[i],
            "mean_out": mean_out_mat[i],
            "delta": means[i] - mean_out_mat[i],
            "cohen_d": all_d[i],
            "p": all_p[i],
            "q": qvals,
            "prevalence_in": prevalence[i],
            "prevalence_out": prev_out_mat[i],
            "n_in": n_in_list[i],
            "n_out": n_out_list[i],
        }).sort_values(["q", "delta"], ascending=[True, False])
        df_pos = df[(df["delta"] > min_delta) & (df["q"] <= q_thresh)].copy()
        if top_n:
            df_pos = df_pos.head(top_n)
        all_rows.append(df)
        per_cluster_top.append(df_pos)

        if make_plots:
            try:
                plot_volcano(df, cl, outdir)
                top_names = df_pos["species"].tolist()[:max_strip]
                plot_top_strips(X[masks[i]], X[mask_outs[i]], top_names, species_names, cl, outdir)
            except Exception as e:
                print(f"[WARN] plotting failed for cluster {cl}: {e}")

    all_df = pd.concat(all_rows, ignore_index=True)
    out_all = os.path.join(outdir, "markers_all_clusters.csv")
    all_df.to_csv(out_all, index=False)
    for df in per_cluster_top:
        cl = df["cluster"].iloc[0] if len(df) else "NA"
        df.to_csv(os.path.join(outdir, f"markers_cluster_{cl}.csv"), index=False)
    print(f"[DONE] wrote {out_all} and per-cluster top lists in {outdir}")
    return all_df, per_cluster_top


# ---------- Plotting ----------

def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-","_")) else "_" for c in name)[:120]


def plot_volcano(df: pd.DataFrame, cl, outdir: str):
    x = df["delta"].to_numpy()
    y = -np.log10(np.clip(df["q"].to_numpy(), 1e-300, 1.0))
    sig = (df["q"] <= CFG.Q_FDR_THRESH) & (df["delta"] > CFG.MIN_DELTA)

    plt.figure(figsize=(6,5))
    plt.scatter(x[~sig], y[~sig], s=8, alpha=0.5)
    plt.scatter(x[sig], y[sig], s=12, alpha=0.9)
    plt.axhline(-np.log10(CFG.Q_FDR_THRESH), linestyle=":")
    plt.axvline(CFG.MIN_DELTA, linestyle=":")
    plt.xlabel("Delta (mean_in - mean_out)")
    plt.ylabel("-log10(q)")
    plt.title(f"Cluster {cl}: marker scan")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"volcano_cluster_{cl}.png"), dpi=200)
    plt.close()


def plot_top_strips(X_in: csr_matrix, X_out: csr_matrix, top_species: List[str], species_names: List[str], cl, outdir: str):
    name_to_idx = {n:i for i,n in enumerate(species_names)}
    for sp in top_species:
        j = name_to_idx.get(sp)
        if j is None:
            continue
        xi = to_dense_col(X_in, j)
        xo = to_dense_col(X_out, j)
        plt.figure(figsize=(5,3))
        jitter_i = np.random.uniform(-0.05, 0.05, size=xi.size)
        jitter_o = np.random.uniform(-0.05, 0.05, size=xo.size)
        plt.scatter(np.full_like(xi, 0.9)+jitter_i, xi, s=10, alpha=0.6)
        plt.scatter(np.full_like(xo, 1.1)+jitter_o, xo, s=10, alpha=0.6)
        plt.xticks([1.0, 1.2], [f"in {cl}", "out"])
        plt.ylabel("Hellinger abundance")
        plt.title(sp)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"strip_cluster_{cl}_{_sanitize(sp)}.png"), dpi=200)
        plt.close()


# ---------- IO + orchestration ----------

def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    # Locate inputs
    hell_path = _glob_one(CFG.HELL_DIR, CFG.HELL_FEATS_GLOB)
    cell_index_path = _glob_one(CFG.HELL_DIR, CFG.CELL_INDEX_GLOB)
    labels_df = pd.read_parquet(CFG.LABELS_PATH)

    print(f"[LOAD] Hellinger CSR: {hell_path}")
    X = load_npz(hell_path).tocsr()

    print(f"[LOAD] cell_index: {cell_index_path}")
    cell_index = pd.read_parquet(cell_index_path)

    # Validate shapes
    n_cells = X.shape[0]
    if len(cell_index) != n_cells:
        raise SystemExit(f"cell_index rows ({len(cell_index)}) != Hellinger rows ({n_cells})")

    # Align labels to cell_index order
    if "cell_id" in labels_df.columns and "cell_id" in cell_index.columns:
        lab = labels_df.set_index("cell_id").loc[cell_index["cell_id"].values]
        labels = lab["bioregion"].to_numpy(dtype=int)
    else:
        # assume identical order as in cluster script
        labels = labels_df["bioregion"].to_numpy(dtype=int)
        if labels.shape[0] != n_cells:
            raise SystemExit("Labels length does not match cells, and no cell_id to align on.")

    # Species names
    p = X.shape[1]
    species_names = _read_species_names(CFG.SPECIES_NAMES_PATH, p)
    src = "explicit file" if CFG.SPECIES_NAMES_PATH else "synthetic (no file provided)"
    print(f"[NAMES] Loaded {len(species_names)} names from {src}. First3={species_names[:3]}")


    print(f"[INFO] cells={n_cells} species={p} clusters={np.unique(labels).size}")

    scan_markers(
        X=X,
        labels=labels,
        species_names=species_names,
        outdir=CFG.OUT_DIR,
        q_thresh=CFG.Q_FDR_THRESH,
        top_n=CFG.TOP_N,
        min_delta=CFG.MIN_DELTA,
        make_plots=CFG.MAKE_PLOTS,
        max_strip=CFG.MAX_STRIP_PLOTS_PER_CLUSTER,
    )


if __name__ == "__main__":
    main()
