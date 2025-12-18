from pathlib import Path
import re
import time
import numpy as np
from datetime import datetime
import unicodedata
from collections import Counter
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import csv
from typing import Optional
from scipy.spatial import cKDTree


# =============== CONFIG (PLANTS-ONLY VARIANT) =====================
# Root occurrence parquet parts
OCCURRENCE_DIR =  "/home/eric/GitHub/occurrence_ds/"          # Path to folder with parquet parts
REDLIST_CSV    = "/home/eric/GitHub/metadata/swedish_red-list.csv"   # National red list (semicolon CSV)

# Filter list (semicolon-separated; first column contains Dyntaxa "Taxon id")
PLANTLIST_CSV  = "/home/eric/GitHub/filterlist.csv"

# Output root for the run
OUTPUT_ROOT    = "/home/eric/GitHub/OUT/"

# Focal species selectors
FOCAL_TAXONID_NUMS = []                 # e.g., [227592, 6003469]
FOCAL_SCI_NAMES    = []                 # scientificName exact match(es)
FOCAL_CSV          = "/home/eric/GitHub/grandlist.csv"  # Path to focal taxon list

BUFFER_M       = 50      # buffer radius in meters
SAMPLE_SITES   = 400     # max number of focal occurrences (sites) to buffer; if fewer exist, use all
TARGET_CRS     = 3006    # SWEREF 99 TM
SWEDEN_BBOX    = (10.0, 55.0, 25.0, 69.5)  # lon_min, lat_min, lon_max, lat_max (coarse sanity filter)

LATBANDS_N = 10                 
LATBANDS_STRICT_EQUAL = True    
LATBANDS_ALLOW_REPLACE = False  
LATBANDS_BALANCE_EFFORT = False 
LATBANDS_WEIGHT_COL = None  

# Column names expected in the DwC-A parts (coming from your converter)
COL_LON        = "decimalLongitude"
COL_SPECIES_ID  = "taxonID_num"
COL_EVENTDATE  = "eventDate" 
COL_LAT        = "decimalLatitude"
COL_TAXONID    = "taxonID"            # LSID string
COL_SCI        = "scientificName"
COL_VERN       = "vernacularName"
COL_OCCID      = "occurrenceID"
COL_LOCID      = "localityID"         # optional
COL_LOCALITY   = "locality"           # optional
# ========================================================


def add_region_latbands(df: pd.DataFrame, n_bands: int = 10,
                        lat_col: str = "decimalLatitude") -> pd.DataFrame:
    d = df.copy()
    if lat_col in d.columns and d[lat_col].notna().any():
        lat = pd.to_numeric(d[lat_col], errors="coerce")
    elif "geometry" in d.columns and hasattr(d, "geometry") and not d.geometry.is_empty.all():
        g = gpd.GeoDataFrame(d, geometry="geometry", crs=getattr(d, "crs", None))
        if g.crs is None:
            g = g.set_crs(4326, allow_override=True)
        lat = g.to_crs(4326).geometry.y
    else:
        raise ValueError(f"Saknar kolumn {lat_col} och ingen geometri för latitudband.")

    lat = pd.to_numeric(lat, errors="coerce")
    mask = lat.notna()
    if not mask.any():
        raise ValueError("Hittade inga giltiga latitudvärden för att skapa latitudband.")

    if lat[mask].nunique(dropna=True) == 1:
        d.loc[mask, "latband"] = "LAT1"
        return d

    codes = pd.qcut(lat[mask], q=n_bands, labels=False, duplicates="drop")
    if hasattr(codes, "max") and pd.notna(codes.max()):
        k = int(codes.max()) + 1
    else:
        k = 1
    labels = [f"LAT{i+1}" for i in range(k)]
    d.loc[mask, "latband"] = pd.Categorical.from_codes(codes.astype(int), categories=labels, ordered=True)
    return d


def _weighted_sample(df: pd.DataFrame, n: int, seed: int,
                     prob_col: Optional[str], replace: bool) -> pd.DataFrame:
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0]
    if prob_col and prob_col in df.columns:
        probs = df[prob_col].clip(lower=0)
        if probs.sum() == 0 or probs.isna().all():
            return df.sample(n=min(n, len(df)), replace=replace, random_state=seed)
        probs = probs.fillna(0)
        probs = probs / probs.sum()
        k = n if replace else min(n, len(df))
        return df.sample(n=k, replace=replace, random_state=seed, weights=probs)
    else:
        k = n if replace else min(n, len(df))
        return df.sample(n=k, replace=replace, random_state=seed)


def stratified_sample_by_latband(df: pd.DataFrame,
                                 n_bands: int,
                                 n_per_band: int,
                                 seed: int = 42,
                                 effort_col: str = "effort_stratum",
                                 prob_col: str = "w_inv",
                                 balance_effort: bool = True,
                                 allow_replace: bool = False,
                                 strict_equal: bool = False) -> pd.DataFrame:
    if "latband" not in df.columns:
        raise ValueError("Saknar 'latband' – kör add_region_latbands först.")

    base = df[df["latband"].notna()].copy()
    if base.empty:
        return base

    if strict_equal and not allow_replace:
        counts = base["latband"].value_counts()
        nz = counts[counts > 0]
        if nz.empty:
            return base.iloc[0:0]
        min_size = nz.min()
        if min_size < n_per_band:
            n_per_band = int(min_size)
            if n_per_band <= 0:
                return base.iloc[0:0]

    out = []
    rng = np.random.RandomState(seed)

    for band, sub in base.groupby("latband", dropna=False, observed=False):
        if sub.empty:
            continue
        if balance_effort and effort_col in sub.columns:
            strata = list(sub[effort_col].fillna("NA").unique())
            if len(strata) == 0:
                strata = ["NA"]
            q, r = divmod(n_per_band, len(strata))
            take = {s: q for s in strata}
            if r > 0:
                add_ones = rng.choice(strata, size=r, replace=False)
                for s in add_ones:
                    take[s] += 1
            for s in strata:
                sub_s = sub[effort_col].fillna("NA").eq(s)
                sub_s = sub[sub_s]
                if sub_s.empty:
                    continue
                k = take[s]
                out.append(_weighted_sample(sub_s, n=k, seed=seed,
                                            prob_col=prob_col, replace=allow_replace))
        else:
            out.append(_weighted_sample(sub, n=n_per_band, seed=seed,
                                        prob_col=prob_col, replace=allow_replace))

    if len(out) == 0:
        return base.iloc[0:0]

    res = pd.concat(out, ignore_index=True)
    if allow_replace:
        res = (res.groupby("latband", group_keys=False, observed=False)
                 .apply(lambda g: g.head(n_per_band))
                 .reset_index(drop=True))
    return res


def read_focal_species_from_csv(csv_path: str):
    ids, names = [], []
    p = Path(csv_path)
    if not p.is_file():
        print(f"[WARN] No CSV found at {csv_path}")
        return ids, names

    with p.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        filtered_lines = (line for line in f if not line.strip().startswith("#"))
        reader = csv.DictReader(filtered_lines, dialect=dialect)

        def get_any(row, *keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return None

        for row in reader:
            tid = get_any(row, "taxonID", "TaxonID", "taxonId", "id")
            nm  = get_any(row, "name", "scientificName", "scientific_name")
            if tid:
                try:
                    ids.append(int(str(tid).strip()))
                except ValueError:
                    print(f"[WARN] Could not parse taxonID '{tid}'")
            if nm:
                names.append(str(nm).strip())

    ids = sorted(set(ids))
    names = sorted(set(names))
    print(f"[FOCAL] Loaded {len(ids)} taxonIDs and {len(names)} names from CSV.")
    print("  IDs head:", ids[:5])
    print("  Names head:", names[:5])
    return ids, names


def get_focal_species():
    ids, names = [], []
    ids.extend(FOCAL_TAXONID_NUMS)
    names.extend(FOCAL_SCI_NAMES)
    if FOCAL_CSV:
        csv_ids, csv_names = read_focal_species_from_csv(FOCAL_CSV)
        ids.extend(csv_ids)
        names.extend(csv_names)
    ids = sorted(set(ids))
    names = sorted(set(names))
    return ids, names


def sanitize_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name) if name is not None else "unknown"
    name = unicodedata.normalize("NFC", name.strip())
    name = name.replace(" ", "_")
    allowed = r"[^A-Za-z0-9_\-\.åäöÅÄÖ]"
    return re.sub(allowed, "_", name)


def extract_dyntaxa_num(lsid: pd.Series) -> pd.Series:
    pat = re.compile(r":Taxon:(\d+)$")
    out = lsid.astype("string").str.extract(pat, expand=False)
    return out


def load_redlist_ids(path: str) -> set:
    df = pd.read_csv(path, sep=";", dtype="string", low_memory=False)
    if "Taxon id" not in df.columns:
        raise SystemExit("Red-list CSV missing 'Taxon id' column.")
    ids = df["Taxon id"].dropna().astype("Int64").astype("string")
    return set(ids.tolist())


# NEW: load plant (Tracheophyta) taxon IDs (accepts first column if header differs)
def load_semicolon_firstcol_ids(path: str, expected_col: str = "Taxon id") -> set:
    df = pd.read_csv(path, sep=";", dtype="string", low_memory=False)
    if expected_col in df.columns:
        ser = df[expected_col]
    else:
        # fall back to first column regardless of name
        first = df.columns[0]
        ser = df[first]
    ids = ser.dropna().astype("Int64").astype("string")
    return set(ids.tolist())


def iter_parts(parquet_dir: str):
    p = Path(parquet_dir)
    parts = sorted(p.glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"No parquet parts found in {parquet_dir}")
    for fp in parts:
        yield fp, pd.read_parquet(fp)


def to_points_3006(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if COL_LON not in df.columns or COL_LAT not in df.columns:
        return gpd.GeoDataFrame(df.iloc[0:0], geometry=[], crs=TARGET_CRS)
    w = df.dropna(subset=[COL_LON, COL_LAT]).copy()
    if w.empty:
        return gpd.GeoDataFrame(w, geometry=[], crs=TARGET_CRS)

    w[COL_LON] = pd.to_numeric(w[COL_LON], errors="coerce")
    w[COL_LAT] = pd.to_numeric(w[COL_LAT], errors="coerce")
    w = w[w[COL_LON].between(-180, 180) & w[COL_LAT].between(-90, 90)]

    lon_min, lat_min, lon_max, lat_max = SWEDEN_BBOX
    w = w[w[COL_LON].between(lon_min, lon_max) & w[COL_LAT].between(lat_min, lat_max)]
    if w.empty:
        return gpd.GeoDataFrame(w, geometry=[], crs=TARGET_CRS)

    gdf = gpd.GeoDataFrame(
        w,
        geometry=[Point(xy) for xy in zip(w[COL_LON], w[COL_LAT])],
        crs=4326
    ).to_crs(TARGET_CRS)
    return gdf


def pick_locality_key(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype="string")

    if COL_LOCID in df.columns:
        out = df[COL_LOCID].astype("string")

    if COL_LOCALITY in df.columns:
        m = out.isna() | (out.str.len() == 0)
        out = out.where(~m, df[COL_LOCALITY].astype("string"))

    if (COL_LON in df.columns) and (COL_LAT in df.columns):
        lon = pd.to_numeric(df[COL_LON], errors="coerce")
        lat = pd.to_numeric(df[COL_LAT], errors="coerce")
        lon3 = lon.map(lambda x: f"{x:.3f}")
        lat3 = lat.map(lambda x: f"{x:.3f}")
        coord_key = "cell_" + lat3 + "_" + lon3
        m = out.isna() | (out.str.len() == 0)
        out = out.where(~m, coord_key.astype("string"))

    return out


def write_species_outputs(sci: str, r: dict, meta: pd.DataFrame, out_root: Path, focal_taxonid: dict):
    base_name = sanitize_filename(sci)
    out_dir = Path(out_root) / base_name
    if out_dir.exists():
        i = 1
        while True:
            candidate = Path(out_root) / f"{base_name}__{i:03d}"
            if not candidate.exists():
                out_dir = candidate
                break
            i += 1
    out_dir.mkdir(parents=True, exist_ok=False)

    freq = Counter(x for x in r["all_assoc_ids"] if x and x != "NA")
    keep_ids = {tid for tid, c in freq.items() if c >= 15}
    uniq_ids_filtered = sorted(keep_ids)
    red_after_filter = sorted({x for x in r["unique_red_associate_ids"] if x in keep_ids})

    focal_display = sci

    with open(out_dir / "uniqueFinds.txt", "w", encoding="utf-8") as f:
        f.write(f"{focal_display} {' '.join(uniq_ids_filtered)}\n")

    with open(out_dir / "totalTargetObs.txt", "w", encoding="utf-8") as f:
        f.write(f"{focal_display} {r['focal_obs_total']} {len(uniq_ids_filtered)} {len(red_after_filter)}\n")

    with open(out_dir / "redFinds.txt", "w", encoding="utf-8") as f:
        f.write(f"{focal_display} {' '.join(red_after_filter)}\n")

    filtered_all_ids = [x for x in r["all_assoc_ids"] if x and x in keep_ids]
    with open(out_dir / "allObservations.txt", "w", encoding="utf-8") as f:
        f.write(f"{focal_display} {' '.join(filtered_all_ids)}\n")

    with open(out_dir / "RedSpeciesLocal.txt", "w", encoding="utf-8") as f:
        for k in sorted(r["red_species_per_locality"].keys()):
            f.write(f"{k} {len(r['red_species_per_locality'][k])}\n")

    with open(out_dir / "LocalitySpecies_ALL.txt", "w", encoding="utf-8") as f:
        f.write("LocalityID\tSWEREF_N\tSWEREF_E\tlatband\tSpecies_IDs\n")
        meta_sorted = meta.sort_values("__FID__")
        for _, row in meta_sorted.iterrows():
            fid = int(row["__FID__"])
            loc_id = str(row["FocalLocalityKey"]).replace(" ", "_")
            N = str(int(round(float(row["FocalN"]))))
            E = str(int(round(float(row["FocalE"]))))
            lb = str(row.get("latband", "NA"))
            sp_ids = sorted(r["loc_species_all"].get(fid, set()))
            f.write(f"{loc_id}\t{N}\t{E}\t{lb}\t{' '.join(sp_ids)}\n")

    if r["locality_red_rows"]:
        df_lr = pd.concat(r["locality_red_rows"], ignore_index=True)
        df_lr["LocalityID"] = df_lr["LocalityKey"].astype("string")
        mask_bad = df_lr["LocalityID"].isna() | (df_lr["LocalityID"] == "") | (df_lr["LocalityID"] == "NA")
        if mask_bad.any():
            pts = gpd.GeoSeries(
                gpd.points_from_xy(
                    pd.to_numeric(df_lr.loc[mask_bad, "E"], errors="coerce"),
                    pd.to_numeric(df_lr.loc[mask_bad, "N"], errors="coerce"),
                    crs=3006
                )
            ).to_crs(4326)
            lat = pts.y.map(lambda v: f"{float(v):.3f}")
            lon = pts.x.map(lambda v: f"{float(v):.3f}")
            df_lr.loc[mask_bad, "LocalityID"] = ("cell_" + lat + "_" + lon).astype("string")
        df_lr = df_lr.dropna(subset=["LocalityID"])
        agg = (
            df_lr.groupby("LocalityID", as_index=False)
                .agg(
                    SWEREF_N=("N", "median"),
                    SWEREF_E=("E", "median"),
                    Species_IDs=("taxonID_num", lambda s: " ".join(sorted(set(s.astype("string").dropna()))))
                )
        )
        agg["SWEREF_N"] = agg["SWEREF_N"].round(0).astype("Int64").astype("string")
        agg["SWEREF_E"] = agg["SWEREF_E"].round(0).astype("Int64").astype("string")
        agg[["LocalityID", "SWEREF_N", "SWEREF_E", "Species_IDs"]].to_csv(out_dir / "LocalitySpecies_RED.txt", sep="\t", index=False)
    else:
        (out_dir / "LocalitySpecies_RED.txt").write_text("LocalityID\tSWEREF_N\tSWEREF_E\tSpecies_IDs\n", encoding="utf-8")

    print(f"[OK] Wrote outputs for {focal_display} → {out_dir}")



def main(seed=None):
    import numpy as np
    import random
    start = time.time()

    if seed is None:
        seed = int(time.time() * 1e6) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    print(f"[INIT] Using random seed = {seed}")

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
 
    # Red-list IDs (strings)
    red_ids = load_redlist_ids(REDLIST_CSV)
    # PLANT FILTER (Tracheophyta)
    plant_ids = load_semicolon_firstcol_ids(PLANTLIST_CSV)  # strings
    print(f"[PLANTS] Loaded {len(plant_ids):,} Tracheophyta IDs from {PLANTLIST_CSV}")

    # --- fast proj ---
    from pyproj import Transformer
    _transformer_4326_to_3006 = Transformer.from_crs(4326, 3006, always_xy=True)

    def lonlat_to_EN_np(lon_np: np.ndarray, lat_np: np.ndarray):
        E, N = _transformer_4326_to_3006.transform(lon_np, lat_np)
        return np.asarray(E, dtype="float64"), np.asarray(N, dtype="float64")

    # --- only the columns we use ---
    PL_COLS = [
        "decimalLongitude","decimalLatitude",
        "taxonID","scientificName","vernacularName",
        "localityID","locality","eventDate"
    ]

    # ---------- PASS 1: focal sampling ----------
    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds
    from collections import defaultdict

    def _scan_all_parts(path: str):
        try:
            lf = pl.scan_parquet(str(Path(path) / "**" / "*.parquet"), recursive=True)
        except TypeError:
            files = [str(p) for p in Path(path).rglob("*.parquet")]
            lf = pl.concat([pl.scan_parquet(p) for p in files])
        return lf

    def _extract_taxon_num_expr():
        return (
            pl.col("taxonID").cast(pl.Utf8, strict=False)
            .str.extract(r"(\d+)$", 1)
            .alias("taxonID_num")
        )

    focal_ids, focal_names = get_focal_species()
    focal_id_set = set(map(str, focal_ids))
    focal_name_set = set(focal_names)

    scan0 = _scan_all_parts(OCCURRENCE_DIR)
    _schema_names = scan0.collect_schema().names()

    want_cols = [
        "decimalLongitude", "decimalLatitude",
        "taxonID", "scientificName", "vernacularName", "eventDate",
        "localityID", "locality"
    ]
    cols_present = [c for c in want_cols if c in _schema_names]

    scan_all = scan0.select([*(pl.col(c) for c in cols_present), _extract_taxon_num_expr()])

    base_cols_focal = [
        "scientificName", "vernacularName", "taxonID_num",
        "decimalLongitude", "decimalLatitude", "eventDate",
    ]
    opt_cols_focal = [c for c in ("localityID", "locality") if c in _schema_names]

    # IMPORTANT: FOCAL species are restricted to plants
    focal_lf = (
        scan_all
        .filter(
            (
                pl.col("taxonID_num").cast(pl.Utf8).is_in(list(focal_id_set))
                | pl.col("scientificName").cast(pl.Utf8).is_in(list(focal_name_set))
            )
            & pl.col("taxonID_num").cast(pl.Utf8).is_in(list(plant_ids))
        )
        .select(base_cols_focal + opt_cols_focal)
    )

    focal_df = focal_lf.collect().to_pandas()
    print(f"[PASS 1] focal rows: {len(focal_df)} across {focal_df['scientificName'].nunique()} species (plants only)")

    if focal_df.empty:
        raise SystemExit("No focal plant occurrences found. Check FOCAL_* and PLANTLIST_CSV.")

    focal_totals = defaultdict(int)
    focal_meta   = {}
    focal_taxonid = {}

    species_samples = {}
    species_index   = {}

    for sci, sub in focal_df.groupby("scientificName", dropna=False):
        if sub.empty:
            continue
        sub = sub.copy()

        focal_totals[sci] += len(sub)
        if sci not in focal_meta:
            vn = sub["vernacularName"].dropna().astype("string").head(1).tolist()
            focal_meta[sci] = vn[0] if vn else None
        if sci not in focal_taxonid:
            dyn = sub["taxonID_num"].dropna().astype("string").head(1).tolist()
            focal_taxonid[sci] = dyn[0] if dyn else None

        # lon/lat sanity & Sweden bbox
        # --- FOCAL (PASS 1) lon/lat sanity & bbox filter on the focal subset ---
        lon = pd.to_numeric(sub["decimalLongitude"], errors="coerce")
        lat = pd.to_numeric(sub["decimalLatitude"], errors="coerce")
        mask_bbox = lon.between(10.0, 25.0) & lat.between(55.0, 69.5)
        sub = sub.loc[mask_bbox].copy()
        if sub.empty:
            continue

        # recompute lon/lat AFTER slicing, then project
        lon = pd.to_numeric(sub["decimalLongitude"], errors="coerce")
        lat = pd.to_numeric(sub["decimalLatitude"], errors="coerce")
        E, N = lonlat_to_EN_np(
            lon.to_numpy(dtype="float64", copy=False),
            lat.to_numpy(dtype="float64", copy=False),
        )
        sub["E"] = E
        sub["N"] = N

        # locality key and latbands on the FOCAL subset
        sub["FocalLocalityKey"] = pick_locality_key(sub)
        sub = add_region_latbands(sub, n_bands=LATBANDS_N, lat_col="decimalLatitude")
        if len(sub) > SAMPLE_SITES:
            n_per_band = max(1, SAMPLE_SITES // LATBANDS_N)
        else:
            n_per_band = max(1, len(sub) // LATBANDS_N) or 1

        sub = stratified_sample_by_latband(
            df=sub,
            n_bands=LATBANDS_N,
            n_per_band=n_per_band,
            seed=42,
            effort_col="effort_stratum",
            prob_col=None,
            balance_effort=False,
            allow_replace=LATBANDS_ALLOW_REPLACE,
            strict_equal=LATBANDS_STRICT_EQUAL
        )

        MAX_SITES = int(SAMPLE_SITES) if (isinstance(SAMPLE_SITES, int) and SAMPLE_SITES > 0) else 200
        if len(sub) > MAX_SITES:
            sub = sub.sample(n=MAX_SITES, random_state=42)

        if ("_SPECCOUNTER" not in globals()):
            _SPECCOUNTER = 0  # type: ignore
        _SPECCOUNTER += 1
        if (_SPECCOUNTER % 50) == 0:
            print(f"[PASS 1] prepped { _SPECCOUNTER } species, last={sci}, sites={len(sub)}", flush=True)

        sub = sub.reset_index(drop=True)
        sub["__FID__"] = sub.index.astype("int64")
        xy = sub[["E","N"]].to_numpy(dtype="float64", copy=False)
        tree = cKDTree(xy)
        bbox = (float(np.nanmin(sub["E"])), float(np.nanmin(sub["N"])),
                float(np.nanmax(sub["E"])), float(np.nanmax(sub["N"])))

        species_samples[sci] = sub
        species_index[sci] = dict(
            xy=xy,
            fids=sub["__FID__"].to_numpy(dtype="int64", copy=False),
            bbox=bbox,
            tree=tree
        )

    results = {}
    for sci, sub in species_samples.items():
        results[sci] = dict(
            focal_obs_total = int(focal_totals.get(sci, 0)),
            unique_associate_ids = set(),
            unique_red_associate_ids = set(),
            all_assoc_ids = [],
            loc_species_all = {int(fid): set() for fid in sub["__FID__"]},
            loc_species_red = {int(fid): set() for fid in sub["__FID__"]},
            red_species_per_locality = {},
            locality_red_rows = [],
        )

    species_list = list(species_index.items())
    radius = float(BUFFER_M)

    # --- coarse 5 km grid index for species bboxes ---
    GRID = 5000.0
    def _cell_range(a_min, a_max, step):
        i0 = int(np.floor(a_min / step))
        i1 = int(np.floor(a_max / step))
        return range(min(i0, i1), max(i0, i1) + 1)

    grid2species = {}
    for sci, idx in species_index.items():
        minE, minN, maxE, maxN = idx["bbox"]
        ix_rng = _cell_range(minE, maxE, GRID)
        iy_rng = _cell_range(minN, maxN, GRID)
        for ix in ix_rng:
            for iy in iy_rng:
                grid2species.setdefault((ix, iy), set()).add(sci)

    # ---------- PASS 2: iterate parts ONCE ----------
    dataset = ds.dataset(OCCURRENCE_DIR, format="parquet")

    schema_names = set(dataset.schema.names)
    cols_for_scan = [c for c in PL_COLS if c in schema_names]
    cols_for_scan = sorted(set(cols_for_scan + ["taxonID"]))

    scanner = ds.Scanner.from_dataset(dataset, columns=cols_for_scan)
    t0 = time.time()
    batch_i = 0
    rows_seen = 0
    print("[PASS 2] Scanning associates with KD-tree joins…", flush=True)

    for batch in scanner.to_batches():
        batch_i += 1
        rows_seen += len(batch)
        if (batch_i % 50) == 0:
            dt = time.time() - t0
            print(f"[PASS 2] batches={batch_i} rows≈{rows_seen:,} elapsed={dt:,.1f}s", flush=True)

        tbl = pa.Table.from_batches([batch])
        if len(tbl.schema.names) != len(set(tbl.schema.names)):
            raise RuntimeError(f"Duplicate columns from Arrow scan: {tbl.schema.names}")

        df = pl.from_arrow(tbl).to_pandas()
        if df.empty:
            continue

        # numeric Dyntaxa ID as string
        df["taxonID_num"] = extract_dyntaxa_num(df.get("taxonID", pd.Series(index=df.index)))

        # Filter OUT non-plant rows early
        df = df[df["taxonID_num"].astype("string").isin(plant_ids)]
        if df.empty:
            continue

        # lon/lat sanity & Sweden bbox
        lon = pd.to_numeric(df["decimalLongitude"], errors="coerce")
        lat = pd.to_numeric(df["decimalLatitude"], errors="coerce")
        mask = lon.between(10.0, 25.0) & lat.between(55.0, 69.5)
        if not mask.any():
            continue
        df = df.loc[mask].copy()

        # drop focal rows so we only keep associates here
        # (focal species are also plants by construction in PASS 1)
        # We need the set of focal taxon IDs for exclusion
        # Build once (outside loop would be nicer, but cheap here)
        # NOTE: we infer from species_samples taxon IDs stored in "focal_taxonid"
        # but that mapping is per species; faster is to build a set from focal_df earlier
        # For clarity, we recompute here from focal_df
        # However, to avoid a closure on focal_df, we built focal_id_set at the top; use that instead.
        mask_focal = pd.Series(False, index=df.index)
        if focal_id_set:
            mask_focal |= df["taxonID_num"].astype("string").isin(focal_id_set)
        if focal_name_set and "scientificName" in df.columns:
            mask_focal |= df["scientificName"].astype("string").isin(focal_name_set)
        df = df.loc[~mask_focal]
        if df.empty:
            continue

        # project
        # IMPORTANT: recompute from current df after all filtering
        lon = pd.to_numeric(df["decimalLongitude"], errors="coerce")
        lat = pd.to_numeric(df["decimalLatitude"], errors="coerce")
        E, N = lonlat_to_EN_np(
            lon.to_numpy(dtype="float64", copy=False),
            lat.to_numpy(dtype="float64", copy=False),
        )
        df["E"] = E
        df["N"] = N

        pminE, pminN = float(np.nanmin(E)), float(np.nanmin(N))
        pmaxE, pmaxN = float(np.nanmax(E)), float(np.nanmax(N))

        df["LocalityKey"] = pick_locality_key(df)
        df["is_red"] = df["taxonID_num"].astype("string").isin(red_ids)

        ix_min = int(np.floor((pminE - radius) / GRID))
        ix_max = int(np.floor((pmaxE + radius) / GRID))
        iy_min = int(np.floor((pminN - radius) / GRID))
        iy_max = int(np.floor((pmaxN + radius) / GRID))

        candidates = set()
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                s = grid2species.get((ix, iy))
                if s:
                    candidates.update(s)
        if not candidates:
            continue

        assoc_xy = df[["E", "N"]].to_numpy(dtype=np.float64, copy=False)
        assoc_tree = cKDTree(assoc_xy)

        assoc_taxon = df["taxonID_num"].astype("string").to_numpy()
        assoc_is_red = df["is_red"].to_numpy()
        assoc_loc = df["LocalityKey"].astype("string").to_numpy()
        assoc_E = df["E"].to_numpy(dtype=np.float64, copy=False)
        assoc_N = df["N"].to_numpy(dtype=np.float64, copy=False)

        for sci in candidates:
            idx = species_index[sci]
            minE, minN, maxE, maxN = idx["bbox"]
            if (pmaxE < (minE - radius)) or (pmaxN < (minN - radius)) or \
               (pminE > (maxE + radius)) or (pminN > (maxN + radius)):
                continue

            xy = idx["xy"]
            fids = idx["fids"]
            sel = (xy[:, 0] >= (pminE - radius)) & (xy[:, 0] <= (pmaxE + radius)) & \
                  (xy[:, 1] >= (pminN - radius)) & (xy[:, 1] <= (pmaxN + radius))
            if not np.any(sel):
                continue
            xy_q = xy[sel]
            fids_q = fids[sel]

            hits = assoc_tree.query_ball_point(xy_q, r=radius)
            if not any(hits):
                continue

            sizes = np.fromiter((len(h) for h in hits), count=len(hits), dtype=np.int64)
            if sizes.sum() == 0:
                continue
            rep_f = np.repeat(fids_q, sizes)
            hit_idx = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])

            hit_taxon = assoc_taxon[hit_idx]
            hit_isred = assoc_is_red[hit_idx]
            hit_loc   = assoc_loc[hit_idx]
            hit_E     = assoc_E[hit_idx]
            hit_N     = assoc_N[hit_idx]

            r = results[sci]

            if len(hit_taxon):
                r["all_assoc_ids"].extend([t for t in hit_taxon if t and t != "NA"])
                if len(rep_f):
                    order = np.argsort(rep_f, kind="mergesort")
                    rep_f_sorted = rep_f[order]
                    taxa_sorted = hit_taxon[order]
                    uniq_fids, start_idx = np.unique(rep_f_sorted, return_index=True)
                    start_idx = np.append(start_idx, len(rep_f_sorted))
                    for i in range(len(uniq_fids)):
                        fid = int(uniq_fids[i])
                        seg = taxa_sorted[start_idx[i]:start_idx[i+1]]
                        r["loc_species_all"][fid].update([t for t in seg if t and t != "NA"])

            if np.any(hit_isred):
                red_mask = hit_isred
                red_taxa = hit_taxon[red_mask]
                red_loc  = hit_loc[red_mask]
                red_E    = hit_E[red_mask]
                red_N    = hit_N[red_mask]

                r["unique_red_associate_ids"].update([t for t in red_taxa if t and t != "NA"])

                if len(rep_f):
                    rep_f_red = rep_f[red_mask]
                    order = np.argsort(rep_f_red, kind="mergesort")
                    rep_f_red = rep_f_red[order]
                    red_taxa_sorted = red_taxa[order]
                    uniq_fids, start_idx = np.unique(rep_f_red, return_index=True)
                    start_idx = np.append(start_idx, len(rep_f_red))
                    for i in range(len(uniq_fids)):
                        fid = int(uniq_fids[i])
                        seg = red_taxa_sorted[start_idx[i]:start_idx[i+1]]
                        r["loc_species_red"][fid].update([t for t in seg if t and t != "NA"])

                for lk, t in zip(red_loc, red_taxa):
                    if lk and lk != "NA" and t and t != "NA":
                        bucket = r["red_species_per_locality"].get(lk, set())
                        bucket.add(t)
                        r["red_species_per_locality"][lk] = bucket

                r["locality_red_rows"].append(pd.DataFrame({
                    "LocalityKey": red_loc,
                    "taxonID_num": red_taxa,
                    "N": red_N,
                    "E": red_E,
                }))

        print(f"[PASS 2] Done batches={batch_i} rows≈{rows_seen:,} in {time.time()-t0:,.1f}s", flush=True)

    out_root = Path(OUTPUT_ROOT); out_root.mkdir(parents=True, exist_ok=True)
    for sci in list(results.keys()):
        meta = species_samples[sci][["__FID__","FocalLocalityKey","N","E","latband"]].copy()
        meta.rename(columns={"N":"FocalN","E":"FocalE"}, inplace=True)
        write_species_outputs(sci, results[sci], meta, out_root, {})
        del results[sci], species_samples[sci], species_index[sci]

    dt = time.time() - start
    print(f"[DONE] Finished in {dt:.1f}s")


if __name__ == "__main__":
    main()
