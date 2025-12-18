import os, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import coo_matrix, save_npz

# ---------- CONFIG ----------
ROOT = "/home/eric/GitHub/OUT/"
PATTERN = os.path.join(ROOT, "*", "LocalitySpecies_ALL.txt")
OUT_MATRIX = "locality_species_presence.npz"
OUT_META   = "locality_metadata.parquet"
SPECIES_LOOKUP = "/home/eric/GitHub/metadata/name_lookup.csv"
EXCLUDE_LIST = "/home/eric/GitHub/metadata/filterlist.csv"
# ----------------------------

def load_exclude_ids(path: str) -> set[str]:
    """
    Load taxon IDs to exclude from the locality × species matrix.

    Expected file format (semicolon-delimited):
        taxonID;name
        247466;Bonellia viridis
        6003219;Maxmuelleria faex
        ...

    Returns a set of taxonID strings, trimmed, with NaNs removed.
    """
    if not path:
        print("[EXCLUDE] No EXCLUDE_LIST path configured; not excluding any species.")
        return set()

    if not os.path.exists(path):
        print(f"[EXCLUDE] File not found: {path}; not excluding any species.")
        return set()

    df_ex = pd.read_csv(
        path,
        sep=";",
        usecols=["taxonID"],
        dtype={"taxonID": "string"},
    )

    df_ex["taxonID"] = df_ex["taxonID"].str.strip()
    df_ex = df_ex.dropna(subset=["taxonID"])
    exclude_ids = set(df_ex["taxonID"].tolist())

    print(f"[EXCLUDE] Loaded {len(exclude_ids)} taxon IDs from {path}")
    return exclude_ids

# Aggregators
loc_to_species = defaultdict(set)         # locality -> set(species_id)
loc_to_latband_first = {}                 # locality -> first latband seen
loc_to_N_vals = defaultdict(list)         # locality -> list of SWEREF_N (for median)
loc_to_E_vals = defaultdict(list)         # locality -> list of SWEREF_E (for median)

exclude_ids = load_exclude_ids(EXCLUDE_LIST)
files = sorted(glob.glob(PATTERN))
print(f"[INIT] Found {len(files)} files")

for k, fp in enumerate(files, 1):
    # Read only the columns we need; be robust to missing latband
    df = pd.read_csv(
        fp,
        sep="\t",
        usecols=lambda c: c in {"LocalityID","Species_IDs","latband","SWEREF_N","SWEREF_E"},
        dtype={"LocalityID": "string", "Species_IDs": "string", "latband": "string"},
    )

    if "latband" not in df.columns:
        df["latband"] = "NA"

    # Clean numeric columns
    for c in ("SWEREF_N","SWEREF_E"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with empty Species_IDs quickly
    df = df.dropna(subset=["LocalityID", "Species_IDs"])
    df = df[df["Species_IDs"].str.strip().str.len() > 0]

    # Vectorized split -> explode; ensures 1 row per (LocalityID, Species_ID)
    df = df.assign(Species_IDs=df["Species_IDs"].str.split()).explode("Species_IDs")
    df = df.dropna(subset=["Species_IDs"])
    if exclude_ids:
        # Species_IDs is read as dtype "string" above, so this is string-to-string compare
        before = len(df)
        df = df[~df["Species_IDs"].isin(exclude_ids)]
        removed = before - len(df)
        if removed > 0:
            print(f"[EXCLUDE] Removed {removed} rows from {os.path.basename(fp)}")
    # De-duplicate duplicates within the same file
    df = df.drop_duplicates(subset=["LocalityID", "Species_IDs"])

    # Update species sets per locality
    for l, s in df[["LocalityID","Species_IDs"]].itertuples(index=False, name=None):
        loc_to_species[l].add(s)

    # Cache first latband per locality if not seen
    if "latband" in df.columns:
        for l, lb in df[["LocalityID","latband"]].drop_duplicates("LocalityID").itertuples(index=False, name=None):
            if l not in loc_to_latband_first:
                loc_to_latband_first[l] = "NA" if pd.isna(lb) else str(lb)

    # Collect coords for median later (avoid huge memory: lists of numbers only)
    for l, n, e in df[["LocalityID","SWEREF_N","SWEREF_E"]].dropna(subset=["SWEREF_N","SWEREF_E"]).itertuples(index=False, name=None):
        loc_to_N_vals[l].append(float(n))
        loc_to_E_vals[l].append(float(e))

    if k % 200 == 0 or k == len(files):
        print(f"[PASS] {k}/{len(files)} files processed")

# ---- Build vocabularies
locs = sorted(loc_to_species.keys())
spp = sorted({s for S in loc_to_species.values() for s in S})
loc_index = {l:i for i,l in enumerate(locs)}
sp_index  = {s:i for i,s in enumerate(spp)}

print(f"[VOCAB] localities={len(locs)} species={len(spp)}")

# ---- COO assembly (binary presence)
nnz_est = sum(len(S) for S in loc_to_species.values())
rows_i = np.empty(nnz_est, dtype=np.int64)
cols_j = np.empty(nnz_est, dtype=np.int64)

p = 0
for l, S in loc_to_species.items():
    i = loc_index[l]
    for s in S:
        rows_i[p] = i
        cols_j[p] = sp_index[s]
        p += 1

# Trim if our estimate overshot (shouldn't, but safe)
rows_i = rows_i[:p]
cols_j = cols_j[:p]
data = np.ones(p, dtype=np.uint8)

X = coo_matrix((data, (rows_i, cols_j)), shape=(len(locs), len(spp))).tocsr()
X.sum_duplicates()
X.data[:] = 1

# ---- Load taxonomy lookup and build species_index with names
# Expecting columns: taxonID;name;vernacular (semicolon-separated)
if not os.path.exists(SPECIES_LOOKUP):
    raise SystemExit(f"Missing SPECIES_LOOKUP: {SPECIES_LOOKUP}")

# Read with headers from file; pick only the two columns we need
tax = pd.read_csv(
    SPECIES_LOOKUP,
    sep=";",
    usecols=["taxonID", "vernacular"],
    dtype={"taxonID": "string", "vernacular": "string"},
)

# Clean & deduplicate by taxonID (keep the first occurrence)
tax["taxonID"] = tax["taxonID"].str.strip()
tax = tax.drop_duplicates(subset=["taxonID"], keep="first").copy()

# spp = list of species IDs (strings) in EXACT column order of the matrix
df_ids = pd.DataFrame({"species_id": pd.Series(spp, dtype="string")})
df_ids["species_id"] = df_ids["species_id"].str.strip()

# Left-merge then collapse any accidental expansions (paranoia guard),
# and reindex to the original spp order to guarantee 1:1 alignment.
species_index = (
    df_ids
    .merge(tax.rename(columns={"taxonID": "species_id", "vernacular": "species"}),
           on="species_id", how="left")
)

# If duplicates slipped through, collapse to one row per species_id
if species_index.shape[0] != len(spp):
    species_index = (species_index
                     .groupby("species_id", as_index=False, sort=False)
                     .agg({"species": "first"}))

# Reindex to the exact matrix column order
species_index = (species_index
                 .set_index("species_id")
                 .reindex(spp)            # guarantees length == len(spp) in right order
                 .reset_index())

# Fallback to ID when name is missing
species_index["species"] = species_index["species"].fillna(species_index["species_id"])

# Add column index for traceability (now lengths match by construction)
species_index.insert(0, "col_index", np.arange(len(spp), dtype=np.int32))

# ---- Save outputs into ROOT (deterministic locations)
os.makedirs(ROOT, exist_ok=True)
base = os.path.splitext(OUT_MATRIX)[0]  # "locality_species_presence"
sp_out = os.path.join(ROOT, f"{base}_species_index_f{len(spp)}.parquet")
species_index.to_parquet(sp_out, index=False)
print(f"[SAVE] Species index -> {sp_out}")

print(f"[MATRIX] shape={X.shape} nnz={X.nnz} indptr_size={X.indptr.size} (rows+1={X.shape[0]+1})")
# ---- Locality metadata
latband_series = pd.Series({l: loc_to_latband_first.get(l, "NA") for l in locs}, dtype="string")
N_med = [np.median(loc_to_N_vals[l]) if loc_to_N_vals[l] else np.nan for l in locs]
E_med = [np.median(loc_to_E_vals[l]) if loc_to_E_vals[l] else np.nan for l in locs]

loc_meta = pd.DataFrame(
    {
        "LocalityID": locs,
        "latband": latband_series.values.astype(object),
        "SWEREF_N": N_med,
        "SWEREF_E": E_med,
    }
)

out_matrix_path = os.path.join(ROOT, OUT_MATRIX)
out_meta_path   = os.path.join(ROOT, OUT_META)

save_npz(out_matrix_path, X)
loc_meta.to_parquet(out_meta_path, index=False)

print(f"[SAVE] Wrote matrix -> {out_matrix_path}")
print(f"[SAVE] Wrote metadata -> {out_meta_path}")

# ---- Sanity checks
assert X.indptr.size == X.shape[0] + 1
assert len(loc_meta) == X.shape[0]
assert os.path.exists(sp_out), "Species index write failed"
