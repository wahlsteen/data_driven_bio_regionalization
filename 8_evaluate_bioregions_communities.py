#!/usr/bin/env python3

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================

CONFIG = {
    # Paths
    "COMM_MATRIX_PATH": "/home/eric/GitHub/analysis_OUT/hellinger_features_c4255_f9105.npz",
    "CELL_INDEX_PATH": "/home/eric/GitHub/analysis_OUT/cell_index_4255.parquet",
    "CLUSTERS_PATH": "/home/eric/GitHub/analysis_OUT/bioregion_labels_TUNED.parquet", 
    "SPECIES_INDEX_PATH": "/home/eric/GitHub/OUT/locality_species_presence_species_index_f9105.parquet",
    "MARKERS_PATH": "/home/eric/GitHub/analysis_OUT/markers_all_clusters.csv",  # optional
    "OUTPUT_DIR": "/home/eric/GitHub/analysis_OUT/",


    # Column names
    "CELL_ID_COL": "cell_id",          # column in cell index + clusters
    "CLUSTER_COL": "bioregion",          # cluster label column
    "SPECIES_ID_COL": "species_id",    # only needed if you later use species_index/markers

    # For LCBD & beta
    "ASSUME_MATRIX_IS_HELLINGER": True,  # set False if raw abundances, then Hellinger will be applied
    "MAX_PAIRS_PER_CLUSTER": 50000,      # limit pairwise Sørensen computations per cluster
    "RANDOM_SEED": 12345,
}


# ========================
# DATA STRUCTURES
# ========================

@dataclass
class CommunityData:
    X: sparse.csr_matrix         # cells × species matrix
    cell_df: pd.DataFrame        # index aligned with rows of X
    clusters: pd.Series          # cluster label per row (aligned with X rows)
    species_index: Optional[pd.DataFrame] = None


# ========================
# LOADING FUNCTIONS
# ========================
from typing import Tuple

# ========================
# PLOTTING
# ========================

def plot_lcbd_by_cluster(cell_df: pd.DataFrame, clusters: pd.Series, cfg: Dict) -> None:
    """
    Boxplot of LCBD per cluster, saved as PNG.
    Assumes cell_df has a 'LCBD' column already.
    """
    if "LCBD" not in cell_df.columns:
        print("[PLOT] LCBD column not found in cell_df – skipping LCBD-by-cluster plot.")
        return

    df = cell_df.copy()
    df["cluster"] = clusters.values

    cluster_ids = sorted(df["cluster"].unique())
    data = [df.loc[df["cluster"] == cl, "LCBD"].values for cl in cluster_ids]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=cluster_ids, showfliers=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("LCBD")
    ax.set_title("LCBD distribution per cluster")
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(cfg["OUTPUT_DIR"], "lcbd_by_cluster_boxplot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Wrote LCBD-by-cluster boxplot → {out_path}")


def plot_cluster_radar(
    lcbd_summary: pd.DataFrame,
    ms_indval_summary: Optional[pd.DataFrame],
    beta_df: pd.DataFrame,
    cfg: Dict
) -> None:
    """
    Radar plot comparing clusters across scaled community metrics.

    Uses:
      - LCBD_mean
      - beta_turnover_mean
      - beta_nestedness_mean
      - prop_indicator_species (if available)
      - mean_effect (if available)

    All metrics are min-max scaled to [0,1] for visualization.
    """
    # Merge per-cluster tables
    merged = lcbd_summary.merge(beta_df, on="cluster", how="left")

    if ms_indval_summary is not None:
        merged = merged.merge(ms_indval_summary, on="cluster", how="left")

    # Choose which columns to include
    metric_cols = ["LCBD_mean", "beta_turnover_mean", "beta_nestedness_mean"]
    if "prop_indicator_species" in merged.columns:
        metric_cols.append("prop_indicator_species")
    if "mean_effect" in merged.columns:
        metric_cols.append("mean_effect")

    if len(metric_cols) < 3:
        print("[PLOT] Too few metrics for a meaningful radar plot – skipping.")
        return

    # Min-max scale each metric to [0,1]
    scaled = merged[["cluster"]].copy()
    for col in metric_cols:
        col_min = merged[col].min()
        col_max = merged[col].max()
        if col_max > col_min:
            scaled[col] = (merged[col] - col_min) / (col_max - col_min)
        else:
            # constant column: all 0.5
            scaled[col] = 0.5

    clusters_sorted = scaled["cluster"].tolist()
    values = scaled[metric_cols].values

    # Radar geometry
    n_vars = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    # close the loop: repeat first angle at end
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each cluster
    for row, cl in zip(values, clusters_sorted):
        vals = np.concatenate([row, row[:1]])  # close loop
        ax.plot(angles, vals, label=f"Cluster {cl}")
        ax.fill(angles, vals, alpha=0.1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_yticklabels([])  # hide radial labels, they're arbitrary [0,1]

    ax.set_title("Cluster community profiles (scaled metrics)", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    out_path = os.path.join(cfg["OUTPUT_DIR"], "cluster_radar_plot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Wrote cluster radar plot → {out_path}")

def load_clusters(path: str) -> pd.DataFrame:
    print(f"[LOAD] Cluster labels: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def align_cells_with_clusters(
    cell_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    cell_id_col: str,
    cluster_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Try to align cells with cluster labels.

    Strategy:
      1) If both dataframes share cell_id_col → merge on that.
      2) Otherwise, if lengths match → align by row index and copy cluster_col.
    """

    print("[ALIGN] Matching cells with cluster labels…")
    print(f"  cell_df columns:     {list(cell_df.columns)}")
    print(f"  clusters_df columns: {list(clusters_df.columns)}")

    # Case 1: both files have a common ID column -> merge
    if (cell_id_col in cell_df.columns) and (cell_id_col in clusters_df.columns):
        print(f"[ALIGN] Using '{cell_id_col}' as key for merge.")
        merged = cell_df.merge(
            clusters_df[[cell_id_col, cluster_col]],
            on=cell_id_col,
            how="left",
        )

        if merged[cluster_col].isna().any():
            n_missing = merged[cluster_col].isna().sum()
            print(f"[WARN] {n_missing} cells have no cluster label (NaN); dropping those rows.")
            merged = merged.loc[~merged[cluster_col].isna()].reset_index(drop=True)

        clusters = merged[cluster_col].astype(int)
        merged = merged.drop(columns=[cluster_col])
        return merged, clusters

    # Case 2: no shared ID column, but row order and length match
    if len(cell_df) == len(clusters_df) and cluster_col in clusters_df.columns:
        print("[ALIGN] Falling back to index-based alignment (same order, same length).")
        cell_df = cell_df.reset_index(drop=True)
        clusters = clusters_df[cluster_col].astype(int).reset_index(drop=True)
        return cell_df, clusters

    # If we get here: something is truly inconsistent
    raise ValueError(
        "Cannot align cells with clusters.\n"
        f"- cell_id_col='{cell_id_col}' not present in both frames, and\n"
        f"- either lengths differ (cell_df={len(cell_df)}, clusters_df={len(clusters_df)}) "
        "or cluster column is missing.\n"
        "Check that you configured the correct CLUSTERS_PATH, CELL_ID_COL and CLUSTER_COL, "
        "and that cluster labels were produced from the same cell_index as the community matrix."
    )

def load_sparse_csr(path: str) -> sparse.csr_matrix:
    print(f"[LOAD] Community matrix: {path}")
    loader = np.load(path)
    return sparse.csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])


def load_cell_index(path: str) -> pd.DataFrame:
    print(f"[LOAD] Cell index: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def load_species_index(path: str) -> pd.DataFrame:
    print(f"[LOAD] Species index (optional): {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def load_community_data(cfg: Dict) -> CommunityData:
    X = load_sparse_csr(cfg["COMM_MATRIX_PATH"])
    cell_df = load_cell_index(cfg["CELL_INDEX_PATH"])
    clusters_df = load_clusters(cfg["CLUSTERS_PATH"])

    if len(cell_df) != X.shape[0]:
        raise ValueError(f"Cell index rows ({len(cell_df)}) != X rows ({X.shape[0]}). Check alignment.")

    cell_df_aligned, clusters = align_cells_with_clusters(
        cell_df,
        clusters_df,
        cfg["CELL_ID_COL"],
        cfg["CLUSTER_COL"],
    )

    # at this point sizes should match, so X doesn’t need subsetting
    species_index = None
    sp_path = cfg.get("SPECIES_INDEX_PATH")
    if sp_path and os.path.exists(sp_path):
        species_index = load_species_index(sp_path)
    else:
        print("[INFO] No species index provided or file not found – species-level reporting will be limited.")

    return CommunityData(X=X, cell_df=cell_df_aligned, clusters=clusters, species_index=species_index)


# ========================
# HELLINGER & LCBD
# ========================

def hellinger_transform(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Apply Hellinger transform to a sparse abundance matrix.
    """
    print("[HELLINGER] Applying Hellinger transformation…")
    X = X.tocsr()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = 1.0 / row_sums
    X_norm = X.multiply(inv_row_sums[:, None])

    # sqrt of non-zero data
    X_norm.data = np.sqrt(X_norm.data)
    return X_norm


def compute_lcbd(X_hellinger: sparse.csr_matrix) -> np.ndarray:
    """
    Compute LCBD (Local Contribution to Beta Diversity) following Legendre & De Cáceres (2013).

    This implementation converts to dense, so it's memory hungry but simple.
    For ~4–5k cells × ~40k species it's borderline but should be OK on a decent machine.
    """
    print("[LCBD] Computing LCBD values (dense approximation)…")
    Y = X_hellinger.toarray()
    col_means = Y.mean(axis=0, keepdims=True)
    Y_centered = Y - col_means
    SS_per_row = np.square(Y_centered).sum(axis=1)
    SS_total = SS_per_row.sum()
    lcbd = SS_per_row / SS_total
    return lcbd


# ========================
# MULTI-SPECIES INDICATOR METRICS
# ========================

def load_markers(markers_path: str) -> Optional[pd.DataFrame]:
    if not markers_path or not os.path.exists(markers_path):
        print("[INFO] No markers_all_clusters file provided – skipping multi-species IndVal metrics.")
        return None
    print(f"[LOAD] markers_all_clusters: {markers_path}")
    df = pd.read_csv(markers_path)
    return df


def summarize_multispecies_indval(markers_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    cluster_col = "cluster"
    df = markers_df.copy()

    # Choose an effect column (delta or cohen_d)
    indval_col = None
    for c in ["indval", "IndVal", "delta", "cohen_d"]:
        if c in df.columns:
            indval_col = c
            break

    if indval_col is None:
        raise ValueError("No effect-size / IndVal-like column found.")

    # Define indicator as "positive effect"
    df["is_indicator"] = df[indval_col] > 0

    rows = []
    for cl, g in df.groupby(cluster_col):
        sub = g[g["is_indicator"]]
        row = {
            "cluster": cl,
            "n_candidate_species": len(g),
            "n_indicator_species": len(sub),
            "prop_indicator_species": len(sub) / len(g) if len(g) > 0 else np.nan,
            "sum_effect": sub[indval_col].sum(),
            "mean_effect": sub[indval_col].mean(),
            "median_effect": sub[indval_col].median(),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ========================
# SØRENSEN β PARTITION
# ========================

def row_nz_as_set(X: sparse.csr_matrix, i: int) -> set:
    row = X.getrow(i)
    return set(row.indices)


def sorensen_partition_for_pair(a: int, b: int, c: int) -> Tuple[float, float, float]:
    """
    Baselga's partition for pairwise Sørensen:

    a = shared species
    b = only in site 1
    c = only in site 2

    β_sor = (b + c) / (2a + b + c)
    β_sim (turnover) = min(b, c) / (a + min(b, c))
    β_sne (nestedness) = β_sor - β_sim
    """
    if (2 * a + b + c) == 0:
        return (0.0, 0.0, 0.0)

    beta_sor = (b + c) / (2.0 * a + b + c)

    if (a + min(b, c)) == 0:
        beta_sim = 0.0
    else:
        beta_sim = min(b, c) / (a + min(b, c))

    beta_sne = beta_sor - beta_sim
    return beta_sor, beta_sim, beta_sne


def compute_sorensen_partition_per_cluster(
    X: sparse.csr_matrix,
    clusters: pd.Series,
    cfg: Dict
) -> pd.DataFrame:
    """
    Compute within-cluster Sørensen β and its partition into turnover and nestedness,
    using a random sample of pairs per cluster (to avoid O(n^2) blowup).
    """

    print("[BETA] Computing within-cluster Sørensen β partition (sampled pairs)…")
    random.seed(cfg["RANDOM_SEED"])
    clusters_array = clusters.values
    unique_clusters = np.unique(clusters_array)

    rows = []
    for cl in unique_clusters:
        idx = np.where(clusters_array == cl)[0]
        n = len(idx)
        if n < 2:
            continue

        possible_pairs = n * (n - 1) // 2
        max_pairs = cfg["MAX_PAIRS_PER_CLUSTER"]
        if possible_pairs <= max_pairs:
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((idx[i], idx[j]))
        else:
            pairs = set()
            while len(pairs) < max_pairs:
                i, j = random.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                pairs.add((idx[i], idx[j]))
            pairs = list(pairs)

        print(f"[BETA] Cluster {cl}: n={n}, pairs used={len(pairs)} (of {possible_pairs})")

        # Pre-cache non-zero sets for this cluster
        nz_sets: Dict[int, set] = {}
        for rid in idx:
            nz_sets[rid] = row_nz_as_set(X, rid)

        sor_list: List[float] = []
        sim_list: List[float] = []
        sne_list: List[float] = []

        for i, j in pairs:
            si = nz_sets[i]
            sj = nz_sets[j]
            a = len(si & sj)
            b = len(si - sj)
            c = len(sj - si)
            beta_sor, beta_sim, beta_sne = sorensen_partition_for_pair(a, b, c)
            sor_list.append(beta_sor)
            sim_list.append(beta_sim)
            sne_list.append(beta_sne)

        row = {
            "cluster": cl,
            "n_cells": n,
            "n_pairs_used": len(pairs),
            "beta_sor_mean": float(np.mean(sor_list)),
            "beta_sor_sd": float(np.std(sor_list, ddof=1)),
            "beta_turnover_mean": float(np.mean(sim_list)),
            "beta_turnover_sd": float(np.std(sim_list, ddof=1)),
            "beta_nestedness_mean": float(np.mean(sne_list)),
            "beta_nestedness_sd": float(np.std(sne_list, ddof=1)),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ========================
# MAIN PIPELINE
# ========================

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    cfg = CONFIG
    ensure_output_dir(cfg["OUTPUT_DIR"])

    # Save config snapshot
    with open(os.path.join(cfg["OUTPUT_DIR"], "community_eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Load core data
    data = load_community_data(cfg)

    # Hellinger transform if needed
    if cfg["ASSUME_MATRIX_IS_HELLINGER"]:
        print("[INFO] Assuming X is already Hellinger-transformed.")
        X_hell = data.X
    else:
        X_hell = hellinger_transform(data.X)

    # ======================
    # LCBD
    # ======================
    lcbd = compute_lcbd(X_hell)
    data.cell_df["LCBD"] = lcbd

    lcbd_path = os.path.join(cfg["OUTPUT_DIR"], "lcbd_per_cell.csv")
    cols_to_save = [cfg["CELL_ID_COL"], "LCBD"]
    data.cell_df[cols_to_save].to_csv(lcbd_path, index=False)
    print(f"[LCBD] Wrote LCBD per cell → {lcbd_path}")

    lcbd_summary = (
        data.cell_df
        .assign(cluster=data.clusters.values, LCBD=lcbd)
        .groupby("cluster")["LCBD"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    lcbd_summary.columns = ["cluster", "LCBD_mean", "LCBD_median", "LCBD_sd", "LCBD_min", "LCBD_max"]

    # ======================
    # Multi-species indicator metrics
    # ======================
    markers_df = load_markers(cfg.get("MARKERS_PATH", ""))
    if markers_df is not None:
        ms_indval_summary = summarize_multispecies_indval(markers_df, cfg)
        ms_indval_path = os.path.join(cfg["OUTPUT_DIR"], "multispecies_indval_per_cluster.csv")
        ms_indval_summary.to_csv(ms_indval_path, index=False)
        print(f"[MULTISPECIES] Wrote multi-species indicator summaries → {ms_indval_path}")
    else:
        ms_indval_summary = None

    # ======================
    # Sørensen β partition
    # ======================
    X_pa = data.X.copy().tocsr()
    X_pa.data[:] = 1.0
    beta_df = compute_sorensen_partition_per_cluster(X_pa, data.clusters, cfg)
    beta_path = os.path.join(cfg["OUTPUT_DIR"], "beta_sorensen_partition_per_cluster.csv")
    beta_df.to_csv(beta_path, index=False)
    print(f"[BETA] Wrote β-diversity partition per cluster → {beta_path}")

    # ======================
    # PLOTS
    # ======================
    #plot_lcbd_by_cluster(data.cell_df, data.clusters, cfg) #currently not very helpful 
    #plot_cluster_radar(lcbd_summary, ms_indval_summary, beta_df, cfg) #currently not very helpful 


    # ======================
    # SHORT TEXT REPORT
    # ======================
    report_path = os.path.join(cfg["OUTPUT_DIR"], "community_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Community-level evaluation of bioregions (no traits)\n")
        f.write("===============================================\n\n")
        f.write(f"Cells: {data.X.shape[0]}\n")
        f.write(f"Species: {data.X.shape[1]}\n")
        clusters_list = sorted(int(c) for c in set(data.clusters.values))
        f.write(f"Clusters: {clusters_list}\n\n")


        f.write("LCBD (Local Contribution to Beta Diversity):\n")
        f.write(lcbd_summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        f.write("\n\n")

        if ms_indval_summary is not None:
            f.write("Multi-species indicator metrics per cluster:\n")
            f.write(ms_indval_summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
            f.write("\n\n")

        f.write("Within-cluster Sørensen β partition (Baselga):\n")
        f.write(beta_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        f.write("\n")

    print(f"[REPORT] Wrote community evaluation report → {report_path}")
    print("[DONE] Community-level evaluation complete.")


if __name__ == "__main__":
    main()