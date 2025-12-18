import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
import skbio

# =====================
# CONFIG
# =====================
HELL_PATH = "/home/eric/GitHub/analysis_OUT/hellinger_features_c4255_f9105.npz"
LABELS_PATH = "/home/eric/GitHub/analysis_OUT/bioregion_labels_TUNED.parquet"

DIST_METRIC = "euclidean"   # in Hellinger space this is correct

# =====================
# LOAD DATA
# =====================
print("Loading Hellinger matrix...")
X = load_npz(HELL_PATH).tocsr()
X_dense = X.toarray()   # silhouette and PERMANOVA need dense

print("Loading labels...")
lab = pd.read_parquet(LABELS_PATH)
labels = lab["bioregion"].to_numpy()

clusters = np.unique(labels[labels >= 0])
k = len(clusters)
print(f"Clusters found: {clusters}")

# Helper: cluster -> indices
cluster_idx = {c: np.where(labels == c)[0] for c in clusters}

# =====================
# 1. WITHIN-CLUSTER DISPERSION
# =====================
print("\nComputing within-cluster dispersion (Hellinger–Euclidean distance, unitless)...")

def cluster_centroid(X, ids):
    return X[ids].mean(axis=0)

within_stats = {}
centroids = {}

for c in clusters:
    idx = cluster_idx[c]
    cent = cluster_centroid(X_dense, idx)
    centroids[c] = cent
    dists = np.linalg.norm(X_dense[idx] - cent, axis=1)
    n_c = len(dists)
    mean = float(dists.mean())
    sd = float(dists.std(ddof=1)) if n_c > 1 else np.nan
    se = sd / np.sqrt(n_c) if n_c > 1 else np.nan
    if np.isfinite(se):
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        ci_low = np.nan
        ci_high = np.nan
    within_stats[c] = {
        "n": n_c,
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

# =====================
# 2. BETWEEN-CLUSTER SEPARATION
# =====================
print("Computing between-cluster separations (cell-to-cell Hellinger–Euclidean distances, unitless)...")

# Precompute full distance matrix once
D = pairwise_distances(X_dense, metric=DIST_METRIC)

between_stats = {}
mean_mat = np.full((k, k), np.nan, dtype=float)
cluster_to_pos = {c: i for i, c in enumerate(clusters)}

for ci in clusters:
    idx_i = cluster_idx[ci]
    pi = cluster_to_pos[ci]
    for cj in clusters:
        pj = cluster_to_pos[cj]
        if cj <= ci:
            continue
        idx_j = cluster_idx[cj]
        # All pairwise distances between cells in ci and cj
        block = D[np.ix_(idx_i, idx_j)].ravel()
        n_ij = len(block)
        mean = float(block.mean())
        sd = float(block.std(ddof=1)) if n_ij > 1 else np.nan
        se = sd / np.sqrt(n_ij) if n_ij > 1 else np.nan
        if np.isfinite(se):
            ci_low = mean - 1.96 * se
            ci_high = mean + 1.96 * se
        else:
            ci_low = np.nan
            ci_high = np.nan

        between_stats[(ci, cj)] = {
            "n": n_ij,
            "mean": mean,
            "sd": sd,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        # Fill symmetric mean matrix
        mean_mat[pi, pj] = mean
        mean_mat[pj, pi] = mean

# =====================
# 3. SILHOUETTE SCORE
# =====================
print("Computing silhouette scores...")

sil_overall = silhouette_score(
    X_dense,
    labels,
    metric=DIST_METRIC
)

print("\nComputing per-cluster silhouette scores...")
sil_samples = silhouette_samples(X_dense, labels, metric=DIST_METRIC)

cluster_sil = {}
for c in clusters:
    idx = cluster_idx[c]
    vals = sil_samples[idx]
    cluster_sil[c] = {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": int(len(vals)),
        "prop_negative": float(np.mean(vals < 0)),
    }

# =====================
# 4. PERMANOVA
# =====================
print("Running PERMANOVA...")

n = X_dense.shape[0]
ids = [str(i) for i in range(n)]
dm = skbio.DistanceMatrix(D, ids=ids)
grouping = pd.Series(labels, index=ids, name="cluster")

res = skbio.stats.distance.permanova(
    dm,
    grouping,
    permutations=999
)

# Derive R^2 from pseudo-F, N, k
N = int(res["sample size"])
k_groups = int(res["number of groups"])
F = float(res["test statistic"])

ratio = F * (k_groups - 1) / (N - k_groups)   # = SS_effect / SS_resid
R2 = ratio / (1.0 + ratio)                    # = SS_effect / (SS_effect + SS_resid)

# =====================
# OUTPUT
# =====================
print("\n====== RESULTS ======")

# Within-cluster table
print("\nWithin-cluster dispersion (Hellinger–Euclidean distance, unitless):")
within_df = pd.DataFrame.from_dict(within_stats, orient="index")
within_df.index.name = "cluster"
print(within_df.to_string(float_format=lambda v: f"{v:.5f}"))

# Between-cluster mean matrix
print("\nBetween-cluster separation: mean cell-to-cell Hellinger–Euclidean distance (unitless):")
mean_df = pd.DataFrame(
    mean_mat,
    index=[f"C{c}" for c in clusters],
    columns=[f"C{c}" for c in clusters],
)
def fmt_mean(v):
    if np.isnan(v):
        return "   -   "
    return f"{v:.5f}"
print(mean_df.to_string(formatters={col: fmt_mean for col in mean_df.columns}))

# Detailed between-cluster stats
print("\nBetween-cluster distance statistics (cell-to-cell distances, unitless):")
for (ci, cj), info in between_stats.items():
    print(
        f"  C{ci} vs C{cj}: "
        f"mean={info['mean']:.5f}, "
        f"sd={info['sd']:.5f}, "
        f"95% CI=[{info['ci_low']:.5f}, {info['ci_high']:.5f}], "
        f"n={info['n']}"
    )

# Silhouette
print(f"\nSilhouette score (overall): {sil_overall:.4f}")

print("\nSilhouette by cluster:")
for c, info in cluster_sil.items():
    print(
        f"  Cluster {c}: "
        f"mean={info['mean']:.4f}, "
        f"median={info['median']:.4f}, "
        f"min={info['min']:.4f}, "
        f"max={info['max']:.4f}, "
        f"n={info['n']}, "
        f"negative={info['prop_negative']*100:.1f}%"
    )

# PERMANOVA
print("\nPERMANOVA:")
print(res)
print(f"\nDerived R^2 (variance explained by clusters): {R2:.4f}")
print("\n(All distances are Hellinger–Euclidean in species space; units are dimensionless.)")

# =====================
# OUTPUT (screen + save)
# =====================

# Prepare path for saving report
import os
report_path = os.path.join(os.path.dirname(HELL_PATH), "evaluation_report.txt")

with open(report_path, "w", encoding="utf-8") as f:

    def w(x=""):
        f.write(str(x) + "\n")

    # Header
    w("===== BIOREGION VALIDATION REPORT =====")
    w(f"Clusters found: {clusters.tolist()}")
    w("")

    # Within-cluster dispersion
    w("Within-cluster dispersion (Hellinger–Euclidean distance, unitless):")
    w(within_df.to_string(float_format=lambda v: f"{v:.5f}"))
    w("")

    # Between-cluster matrix
    w("Between-cluster separation: mean cell-to-cell Hellinger–Euclidean distance (unitless):")
    w(mean_df.to_string(formatters={col: fmt_mean for col in mean_df.columns}))
    w("")

    # Detailed between-cluster statistics
    w("Between-cluster distance statistics (cell-to-cell distances, unitless):")
    for (ci, cj), info in between_stats.items():
        w(
            f"  C{ci} vs C{cj}: "
            f"mean={info['mean']:.5f}, "
            f"sd={info['sd']:.5f}, "
            f"95% CI=[{info['ci_low']:.5f}, {info['ci_high']:.5f}], "
            f"n={info['n']}"
        )
    w("")

    # Silhouette results
    w(f"Silhouette score (overall): {sil_overall:.4f}")
    w("")
    w("Silhouette by cluster:")
    for c, info in cluster_sil.items():
        w(
            f"  Cluster {c}: "
            f"mean={info['mean']:.4f}, "
            f"median={info['median']:.4f}, "
            f"min={info['min']:.4f}, "
            f"max={info['max']:.4f}, "
            f"n={info['n']}, "
            f"negative={info['prop_negative']*100:.1f}%"
        )
    w("")

    # PERMANOVA
    w("PERMANOVA results:")
    w(str(res))
    w(f"\nDerived R^2 (variance explained by clusters): {R2:.4f}")
    w("")
    w("(All distances are Hellinger–Euclidean in species space; units are dimensionless.)")

print(f"\n[REPORT] Saved evaluation report → {report_path}")