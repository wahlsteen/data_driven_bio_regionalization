import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, coo_matrix
from sklearn.metrics import pairwise_distances

# =========================
# CONFIG
# =========================
OUT_DIR = "/home/eric/GitHub/analysis_OUT/"

# Inputs produced earlier
KNN_GLOB = "knn_k*_c*.npz"             # kNN graph with distances
HELL_FEATS_GLOB = "hellinger_features_c*_f*.npz"   # Hellinger (row-normalized then sqrt), CSR
CELL_INDEX_GLOB = "cell_index_*.parquet"

# Target number of clusters
TARGET_K = 4
MIN_CLUSTER_SIZE = 25     # orig = 10. merge clusters smaller than this, if any

# Resolution sweep for Leiden/Louvain
RES_GRID = np.geomspace(0.3004, 0.36461, 3)  # try smaller -> fewer clusters
SEED = 123

# Similarity transform for distances in kNN
# w = 1 / (1 + d) gives [0,1]; you can try 'exp' for stronger locality.
SIMILARITY = "adaptive_rbf" # "reciprocal" or "exp"
EXP_BETA = 1.0

LABELS_OUT = "bioregion_labels_TUNED.parquet"
REPORT_OUT = "clustering_tuned_report.txt"
# =========================
def _score_run(labels: np.ndarray, info: dict, target_k: int) -> tuple:
    k = np.unique(labels).size
    # proxy for fragmentation: min cluster size
    _, counts = np.unique(labels, return_counts=True)
    min_ct = counts.min() if counts.size else 0
    quality = info.get("quality", info.get("modularity", -1.0))
    # sort key: (distance to target K, -quality, -min_cluster_size)
    return (abs(k - target_k), -quality, -min_ct)


def _glob_one(dirpath, pattern):
    paths = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(dirpath, pattern)}")
    return paths[-1]


def _distance_knn_to_similarity(A_dist: csr_matrix, mode="adaptive_rbf", beta=1.0,
                                eps_floor=1e-6, two_hop_alpha=0.02, sym="max") -> csr_matrix:
    """Convert kNN distances to a robust similarity graph.

    mode:
      - 'adaptive_rbf': w_ij = exp(-d_ij^2 / (2 * s_i * s_j)) + eps_floor,
                        where s_i is the median distance of i's nonzero neighbors.
      - 'reciprocal' or 'exp' keep your original behavior.
    sym: 'max' | 'mean' | 'min'
    two_hop_alpha: add alpha * A^2 (light 2-hop reinforcement) to reduce fragmentation.
    """
    A = A_dist.tocoo(copy=True)

    if mode == "adaptive_rbf":
        A_csr = A_dist.tocsr()
        # local scale s_i = median neighbor distance (fallback to global median)
        row_meds = np.zeros(A_csr.shape[0], dtype=np.float64)
        data = A_csr.data
        for i in range(A_csr.shape[0]):
            start, end = A_csr.indptr[i], A_csr.indptr[i+1]
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
        raise ValueError("Unknown SIMILARITY mode")

    A = A.tocsr()
    # Symmetrize
    if sym == "max":
        A = A.maximum(A.T)
    elif sym == "mean":
        A = (A + A.T) * 0.5
    else:
        A = A.minimum(A.T)
    A.eliminate_zeros()

    # Optional: 2-hop reinforcement
    if two_hop_alpha and two_hop_alpha > 0.0:
        A2 = A @ A
        A2.setdiag(0.0)
        A2.eliminate_zeros()
        A = A + (A2.multiply(two_hop_alpha))
        A = A.maximum(A.T)  # keep undirected
        A.eliminate_zeros()
    return A


def _run_leiden(A_sim: csr_matrix, resolution: float, seed=123):
    import igraph as ig
    import leidenalg as la
    coo = A_sim.tocoo()
    g = ig.Graph(n=A_sim.shape[0])
    g.add_edges(list(zip(coo.row.tolist(), coo.col.tolist())))
    g.es["weight"] = coo.data.tolist()
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(part.membership, dtype=np.int32)
    info = {"algorithm": "leiden", "resolution": resolution, "quality": float(part.quality())}
    return labels, info


def _run_louvain(A_sim: csr_matrix, resolution: float, seed=123):
    import networkx as nx
    try:
        import community as community_louvain
    except Exception:
        import community.community_louvain as community_louvain
    G = nx.from_scipy_sparse_array(A_sim, edge_attribute="weight")
    labels_dict = community_louvain.best_partition(G, weight="weight", random_state=seed, resolution=resolution)
    n = A_sim.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        labels[i] = labels_dict.get(i, -1)
    mod = community_louvain.modularity(labels_dict, G, weight="weight")
    info = {"algorithm": "louvain", "resolution": resolution, "modularity": float(mod)}
    return labels, info


def _sweep_resolutions(A_sim, res_grid, seed=123):
    runs = []
    algo_used = None
    for res in res_grid:
        try:
            labels, info = _run_leiden(A_sim, resolution=float(res), seed=seed)
            algo_used = "leiden"
        except Exception as e:
            if algo_used == "leiden":
                warnings.warn(f"Leiden failed at res={res}: {e}")
            labels, info = _run_louvain(A_sim, resolution=float(res), seed=seed)
            algo_used = "louvain"
        k = int(np.unique(labels).size)
        runs.append((labels, info, k))
        print(f"[SWEEP] res={res:.4g} -> k={k}")
    return runs


def _merge_small_clusters_to_neighbors(A_sim: csr_matrix, labels: np.ndarray, min_size: int) -> tuple[np.ndarray, int]:
    labs = labels.copy()
    unique, counts = np.unique(labs, return_counts=True)
    small = [c for c, ct in zip(unique, counts) if ct < min_size]
    n_small = len(small)
    if n_small == 0:
        return labs, 0

    A = A_sim.tocsr()
    for c in small:
        members = np.where(labs == c)[0]
        if members.size == 0:
            continue
        neigh_clusters = {}
        for i in members:
            start, end = A.indptr[i], A.indptr[i+1]
            nbrs = A.indices[start:end]
            w = A.data[start:end]
            for j, wij in zip(nbrs, w):
                cj = labs[j]
                if cj == c:
                    continue
                neigh_clusters.setdefault(cj, []).append(wij)
        if not neigh_clusters:
            cand = [cl for cl in unique if cl != c]
            if not cand:
                continue
            sizes = {cl: np.sum(labs == cl) for cl in cand}
            best = max(sizes, key=sizes.get)
        else:
            best = max(neigh_clusters, key=lambda cl: np.mean(neigh_clusters[cl]))
        labs[members] = best

    # compact labels
    uniq = np.unique(labs)
    remap = {c: i for i, c in enumerate(uniq)}
    for c in uniq:
        labs[labs == c] = remap[c]
    return labs, n_small


def _cluster_centroids_sparse(X: csr_matrix, labels: np.ndarray):
    # Build an assignment matrix S (K x N)
    labs = labels.copy()
    valid = labs >= 0
    labs = labs[valid]
    idx = np.arange(labels.size, dtype=np.int64)[valid]
    uniq = np.unique(labs)
    k = uniq.size
    remap = {c: i for i, c in enumerate(uniq)}
    row_idx = np.fromiter((remap[c] for c in labs), dtype=np.int64, count=labs.size)
    col_idx = idx
    data = np.ones(labs.size, dtype=np.float32)
    S = coo_matrix((data, (row_idx, col_idx)), shape=(k, X.shape[0])).tocsr()
    sums = S @ X            # k x p
    counts = np.asarray(S.sum(axis=1)).ravel().astype(np.float32)
    counts[counts == 0.0] = 1.0
    # centroid = mean of member rows (still sparse)
    for i in range(k):
        start, end = sums.indptr[i], sums.indptr[i+1]
        if start < end:
            sums.data[start:end] /= counts[i]
    return sums.tocsr(), uniq  # centroids, label values corresponding to rows


def _greedy_merge_to_K(X: csr_matrix, labels: np.ndarray, target_k: int):
    labels = labels.copy()
    while True:
        uniq = np.unique(labels[labels >= 0])
        k = uniq.size
        if k <= target_k:
            break
        # Centroids in Hellinger space
        C, uniq_vals = _cluster_centroids_sparse(X, labels)
        # Pairwise distances between centroids
        D = pairwise_distances(C, metric="euclidean")
        # Pick closest pair to merge
        np.fill_diagonal(D, np.inf)
        a, b = np.unravel_index(np.argmin(D), D.shape)
        la = uniq_vals[a]
        lb = uniq_vals[b]
        # Merge cluster lb into la (choose the larger as receiver to be stable)
        size_a = np.sum(labels == la)
        size_b = np.sum(labels == lb)
        keep, drop = (la, lb) if size_a >= size_b else (lb, la)
        labels[labels == drop] = keep
    # Compact labels to 0..K-1
    uniq = np.unique(labels[labels >= 0])
    remap = {c:i for i,c in enumerate(uniq)}
    for c in uniq:
        labels[labels == c] = remap[c]
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[LOAD] Inputs…")
    knn_path = _glob_one(OUT_DIR, KNN_GLOB)
    feats_path = _glob_one(OUT_DIR, HELL_FEATS_GLOB)
    cell_index_path = _glob_one(OUT_DIR, CELL_INDEX_GLOB)

    A_dist = load_npz(knn_path).tocsr()
    X_hell = load_npz(feats_path).tocsr()
    cell_index = pd.read_parquet(cell_index_path)

    n = A_dist.shape[0]
    assert X_hell.shape[0] == n and len(cell_index) == n

    # Similarity graph
    print("[GRAPH] Converting kNN distances to similarities…")
    A_sim = _distance_knn_to_similarity(A_dist, mode=SIMILARITY, beta=EXP_BETA)

    # Sweep resolutions
    print("[SWEEP] Resolution tuning…")
    runs = _sweep_resolutions(A_sim, RES_GRID, seed=SEED)

    # Choose best by (|k-TARGET_K|, -quality, -min_cluster_size)
    best_idx = int(np.argmin([_score_run(labs, inf, TARGET_K) for (labs, inf, _) in runs]))
    labels, info, k_chosen = runs[best_idx]
    print(f"[CHOOSE] res={info.get('resolution')} {info['algorithm']} -> k={k_chosen} quality={info.get('quality', info.get('modularity')):.5f}")

    # Merge tiny clusters first (optional)
    labels_m, n_small = _merge_small_clusters_to_neighbors(A_sim, labels, MIN_CLUSTER_SIZE)

    if n_small > 0:
        print(f"[MERGE] Temporarily marking {n_small} tiny clusters (<{MIN_CLUSTER_SIZE}) for merge.")

    # Greedy merge until TARGET_K (uses Hellinger centroids)
    uniq_k = np.unique(labels_m[labels_m >= 0]).size
    if uniq_k > TARGET_K:
        print(f"[MERGE] Greedy merging {uniq_k} -> {TARGET_K} clusters based on centroid similarity…")
        labels_final = _greedy_merge_to_K(X_hell, labels_m, TARGET_K)
    else:
        labels_final = labels_m.copy()
        # Compact labels
        uniq = np.unique(labels_final[labels_final >= 0])
        remap = {c:i for i,c in enumerate(uniq)}
        for c in uniq:
            labels_final[labels_final == c] = remap[c]

    # Save labels
    out_df = cell_index.copy()
    out_df["bioregion"] = labels_final.astype(np.int32)
    out_path = os.path.join(OUT_DIR, LABELS_OUT)
    out_df.to_parquet(out_path, index=False)
    print(f"[SAVE] Tuned labels -> {out_path}")

    # Report
    with open(os.path.join(OUT_DIR, REPORT_OUT), "w", encoding="utf-8") as f:
        f.write(f"TARGET_K: {TARGET_K}\n")
        f.write(f"MIN_CLUSTER_SIZE: {MIN_CLUSTER_SIZE}\n")
        f.write(f"Similarity: {SIMILARITY}\n")
        f.write("Resolution sweep (res -> k):\n")
        for (_, info_i, k_i) in runs:
            f.write(f"  {info_i.get('resolution')}: {k_i}\n")
        f.write(f"Chosen: res={info.get('resolution')} algo={info['algorithm']} k={k_chosen}\n")
        f.write(f"Final clusters: {np.unique(labels_final).size}\n")

    print("[DONE] Tuning complete.")


if __name__ == "__main__":
    main()
