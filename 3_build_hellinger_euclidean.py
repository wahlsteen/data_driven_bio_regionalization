import os
import math
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz, csr_matrix, coo_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import glob, os, shutil, re

# =========================
# CONFIG — EDIT AS NEEDED
# =========================
MATRIX_PATH = "/home/eric/GitHub/OUT/locality_species_presence.npz"
META_PATH   = "/home/eric/GitHub/OUT/locality_metadata.parquet"

# Spatial aggregation (meters); SWEREF 99 TM is metric.
GRID_SIZE_M = 10_000  # e.g. 10 km grid. Try 20_000 if you need fewer cells.

# Outputs
OUT_DIR = "/home/eric/GitHub/analysis_OUT/"
SAVE_KNN = True       # also build a sparse kNN graph on Hellinger features
K_NEIGHBORS = 15

# If you prefer the full square distance matrix (N×N) — set True.
# For large N this will be big; we write to a memory-mapped file.
SAVE_FULL_DISTANCE = True
# =========================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _grid_index(n_array, e_array, cell_meters):
    """
    Build a grid index per locality via floor-division in meters.
    Returns:
      cell_ids: array[str], shape (n,)
      keys: array of unique cell keys (tuple[int,int]) in input order
      group_index_map: dict[(gx,gy)] -> group row index
    """
    # Use int floor division; careful with NaNs:
    n_ok = np.isfinite(n_array)
    e_ok = np.isfinite(e_array)
    ok = n_ok & e_ok
    if not ok.all():
        # Drop localities without coords by throwing them into their own "NA" buckets
        # (or you can choose to filter them out)
        n_fill = np.nan_to_num(n_array, nan=-1.0)
        e_fill = np.nan_to_num(e_array, nan=-1.0)
    else:
        n_fill, e_fill = n_array, e_array

    gy = (n_fill // cell_meters).astype(np.int64)
    gx = (e_fill // cell_meters).astype(np.int64)
    # Use tuples as stable keys
    keys = list(zip(gx.tolist(), gy.tolist()))
    # Build a compact group index map preserving first-seen order
    group_index_map = {}
    next_idx = 0
    for k in keys:
        if k not in group_index_map:
            group_index_map[k] = next_idx
            next_idx += 1
    # Return pretty string cell ids as well
    cell_ids = np.array([f"gx{g}_gy{h}" for g, h in keys], dtype=object)
    return cell_ids, keys, group_index_map


def aggregate_rows_by_group(X_csr: csr_matrix, group_keys, group_index_map, weights=None):
    """
    Aggregate CSR rows by group using a 0/1 (or weighted) assignment matrix S and S@X.

    X_csr: (n_localities × n_species) CSR
    group_keys: list of per-row group tuple keys (len = n_localities)
    group_index_map: dict key->row index in the aggregated matrix
    weights: optional array of per-locality weights (shape n_localities,)
    """
    n_rows = X_csr.shape[0]
    n_groups = len(group_index_map)

    if weights is None:
        data = np.ones(n_rows, dtype=np.float32)
    else:
        data = weights.astype(np.float32)

    row_idx = np.fromiter((group_index_map[k] for k in group_keys), dtype=np.int64, count=n_rows)
    col_idx = np.arange(n_rows, dtype=np.int64)

    # S is (n_groups × n_rows), with 1 (or weight) where locality belongs to the group
    S = coo_matrix((data, (row_idx, col_idx)), shape=(n_groups, n_rows), dtype=np.float32).tocsr()

    # Aggregation: sum rows per group
    X_agg = (S @ X_csr).tocsr()
    return X_agg


def row_normalize_csr(X: csr_matrix, eps=1e-12):
    """
    Row-stochastic normalization: divide each row by its row sum.
    Keeps zeros as zeros (sparse), returns float32 CSR.
    """
    X = X.astype(np.float32, copy=False)
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    row_sums[row_sums < eps] = 1.0  # avoid divide-by-zero
    inv = 1.0 / row_sums

    # Scale data by corresponding row inverse
    # Efficient in-place scaling per row slice
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i+1]
        if start < end:
            X.data[start:end] *= inv[i]
    return X


def hellinger_csr(X_rel: csr_matrix):
    """
    Hellinger transform for relative abundances: sqrt of each nonzero entry.
    Preserves sparsity.
    """
    X = X_rel.copy()
    np.sqrt(X.data, out=X.data)
    return X


def save_memmap_distance(dist_mat: np.ndarray, out_path: str):
    """
    Save a dense (N×N) distance matrix as an on-disk memory-mapped .npy file.
    """
    # dist_mat is a dense ndarray; write with numpy.save for memmap reload later
    np.save(out_path, dist_mat)
    return out_path


def main():
    ensure_dir(OUT_DIR)

    print("[LOAD] Reading matrix and metadata…")
    X = load_npz(MATRIX_PATH)      # CSR
    meta = pd.read_parquet(META_PATH)

    assert X.shape[0] == len(meta), "Row mismatch between matrix and metadata."

    # Build grid groups
    print(f"[GRID] Aggregating localities into {GRID_SIZE_M/1000:.0f} km cells…")
    N = meta["SWEREF_N"].to_numpy(dtype="float64", na_value=np.nan)
    E = meta["SWEREF_E"].to_numpy(dtype="float64", na_value=np.nan)

    cell_ids, keys, group_idx_map = _grid_index(N, E, GRID_SIZE_M)
    n_groups = len(group_idx_map)
    print(f"[GRID] Localities: {len(meta)}, unique grid cells: {n_groups}")

    # Aggregate rows by grid cell (sum species counts)
    print("[AGG] Summing species per cell via sparse S@X…")
    X_cell = aggregate_rows_by_group(X, keys, group_idx_map, weights=None)
    # X_cell: (n_cells × n_species)

    # Normalize rows to relative abundances
    print("[NORM] Row-normalize to relative composition…")
    X_rel = row_normalize_csr(X_cell)

    # Hellinger transform (sqrt of relative abundances)
    print("[HELL] Applying Hellinger transform…")
    X_hell = hellinger_csr(X_rel).tocsr()

    # Save the features for reproducibility / later re-use
    hell_out = os.path.join(
        OUT_DIR,
        f"hellinger_features_c{X_hell.shape[0]}_f{X_hell.shape[1]}.npz",
    )
    save_npz(hell_out, X_hell)
    print(f"[SAVE] Hellinger features -> {hell_out}")

    # Find the sidecar written by consolidator (same base name as MATRIX_PATH)


    hell_f = X_hell.shape[1]
    sidecars = glob.glob(os.path.splitext(MATRIX_PATH)[0] + "_species_index_f*.parquet")
    if sidecars:
        src = sidecars[0]  # there should be one; pick first
        dst = os.path.join(
            OUT_DIR, f"species_index_c{X_hell.shape[0]}_f{hell_f}.parquet"
        )
        shutil.copy2(src, dst)
        print(f"[SAVE] Copied species index -> {dst}")
    else:
        print("[WARN] No species index sidecar found next to MATRIX_PATH.")

    # Build an index for cells and carry metadata for mapping/plots
    # Map group index -> representative cell centroid (median of N/E over localities in the cell)
    # We'll compute centroids by aggregating with the same S matrix, but over coordinates.
    print("[META] Building cell-level metadata…")
    # We already have group mapping; build a light COO to average coordinates per cell.
    n_rows = X.shape[0]
    row_idx = np.fromiter((group_idx_map[k] for k in keys), dtype=np.int64, count=n_rows)
    col_idx = np.arange(n_rows, dtype=np.int64)
    S = coo_matrix((np.ones(n_rows, dtype=np.float32), (row_idx, col_idx)), shape=(n_groups, n_rows)).tocsr()

    # Sum coords per cell and divide by counts
    counts = np.asarray(S.sum(axis=1)).ravel()
    N_sum = S @ csr_matrix(N.reshape(-1, 1))
    E_sum = S @ csr_matrix(E.reshape(-1, 1))
    N_cent = np.asarray(N_sum.todense()).ravel() / np.maximum(counts, 1)
    E_cent = np.asarray(E_sum.todense()).ravel() / np.maximum(counts, 1)

    cell_index = pd.DataFrame({
        "cell_id": [None]*n_groups,
        "gx": [None]*n_groups,
        "gy": [None]*n_groups,
        "SWEREF_N_centroid": N_cent,
        "SWEREF_E_centroid": E_cent,
        "n_localities": counts.astype(int),
    })

    # Reverse map index->key
    inv_map = [None]*n_groups
    for k, v in group_idx_map.items():
        inv_map[v] = k
    gx = np.array([k[0] for k in inv_map], dtype=np.int64)
    gy = np.array([k[1] for k in inv_map], dtype=np.int64)
    cell_index["gx"] = gx
    cell_index["gy"] = gy
    cell_index["cell_id"] = [f"gx{g}_gy{h}" for g, h in zip(gx, gy)]

    cell_index_path = os.path.join(OUT_DIR, f"cell_index_{n_groups}.parquet")
    cell_index.to_parquet(cell_index_path, index=False)
    print(f"[SAVE] Cell index -> {cell_index_path}")

    # ---- Optional: kNN graph (sparse)
    if SAVE_KNN:
        print(f"[KNN] Building {K_NEIGHBORS}-NN graph (Euclidean on Hellinger features)…")
        # Use brute-force on sparse CSR; output kneighbors_graph for a sparse adjacency
        nn = NearestNeighbors(
            n_neighbors=K_NEIGHBORS,
            algorithm="brute",            # safer for high-d sparse
            metric="euclidean",
            n_jobs=-1
        )
        nn.fit(X_hell)
        A = nn.kneighbors_graph(X_hell, mode="distance")  # CSR, distances to k neighbors
        knn_path = os.path.join(OUT_DIR, f"knn_k{K_NEIGHBORS}_c{n_groups}.npz")
        save_npz(knn_path, A.astype(np.float32))
        print(f"[SAVE] kNN graph -> {knn_path}")

    # ---- Optional: full dense distance matrix (N×N), memmapped
    if SAVE_FULL_DISTANCE:
        print("[DIST] Computing full Euclidean distance matrix (this may take a while)…")
        # sklearn.pairwise_distances supports CSR input and returns a dense ndarray
        # Keep float32 to reduce RAM.
        D = pairwise_distances(X_hell, X_hell, metric="euclidean", n_jobs=-1)
        D = D.astype(np.float32, copy=False)
        dist_path = os.path.join(OUT_DIR, f"euclidean_dist_c{n_groups}.npy")
        save_memmap_distance(D, dist_path)
        print(f"[SAVE] Dense distance matrix (memmap .npy) -> {dist_path}")

    print("[DONE] Hellinger + Euclidean pipeline complete.")


if __name__ == "__main__":
    main()
