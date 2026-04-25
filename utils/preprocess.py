"""
preprocess.py — Load and transform the MHT-CET dataset from wide to long format.
"""

import pandas as pd
import re

# ── Column metadata ──────────────────────────────────────────────────────────

META_COLS = [
    "Sr. No", "Institute Code", "Institute Name",
    "DISTRICT", "Status", "Fees", "Hostel",
    "Avg Placement", "Highest Placement", "Minority", "Branch",
]

# Friendly display names for messy category column headers
CATEGORY_RENAME = {
    "GST( S & H)":   "GST",
    "GVJ(S & H)":    "GVJ",
    "GNT1(S & H)":   "GNT1",
    "GNT2( S& H)":   "GNT2",
    "GNT3( S& H)":   "GNT3",
    "GSEBC (S & H)": "GSEBC",
    "LOPEN( S & H)": "LOPEN",
    "LOBC(S & H)":   "LOBC",
    "LSC( S&  H)":   "LSC",
    "LST(S & H)":    "LST",
    "LVJ(S & H)":    "LVJ",
    "LNT1(S & H)":   "LNT1",
    "LNT2(S & H)":   "LNT2",
    "LNT3(S & H)":   "LNT3",
    "LSEBC(S & H)":  "LSEBC",
}

# Rank column suffix patterns (to exclude from cutoff melt)
RANK_PATTERN = re.compile(r" R$")


def _get_cutoff_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that are cutoff columns (not metadata, not rank)."""
    return [
        c for c in df.columns
        if c not in META_COLS and not RANK_PATTERN.search(c)
    ]


def load_and_preprocess(path: str = "data/dataset.csv") -> pd.DataFrame:
    """
    1. Load CSV
    2. Clean text / nulls
    3. Melt to long format  →  (Institute, Branch, District, Category, Cutoff)
    4. Return tidy DataFrame
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    raw = pd.read_csv(path, low_memory=False)

    # ── 2. Basic cleaning ────────────────────────────────────────────────────
    # Strip whitespace from string columns
    for col in raw.select_dtypes("object").columns:
        raw[col] = raw[col].astype(str).str.strip()

    # Normalise key text columns to UPPER CASE
    for col in ["Institute Name", "DISTRICT", "Branch", "Status"]:
        if col in raw.columns:
            raw[col] = raw[col].str.upper()

    # Replace "NAN" strings (from .astype(str)) back to real NaN
    raw.replace("NAN", pd.NA, inplace=True)

    # Keep only rows that have at minimum an institute name and branch
    raw.dropna(subset=["Institute Name", "Branch"], inplace=True)

    # ── 3. Rename messy category headers to clean labels ─────────────────────
    raw.rename(columns=CATEGORY_RENAME, inplace=True)

    # ── 4. Melt to long format ───────────────────────────────────────────────
    cutoff_cols = _get_cutoff_cols(raw)          # pure cutoff columns

    keep_meta = [
        "Institute Code", "Institute Name", "DISTRICT",
        "Status", "Fees", "Hostel",
        "Avg Placement", "Highest Placement", "Minority", "Branch",
    ]

    long = raw[keep_meta + cutoff_cols].melt(
        id_vars=keep_meta,
        value_vars=cutoff_cols,
        var_name="Category",
        value_name="Cutoff",
    )

    # Drop rows where cutoff is missing
    long.dropna(subset=["Cutoff"], inplace=True)

    # Cutoff must be numeric
    long["Cutoff"] = pd.to_numeric(long["Cutoff"], errors="coerce")
    long.dropna(subset=["Cutoff"], inplace=True)

    # Trim remaining strings, reset index
    long["Category"] = long["Category"].str.strip()
    long.reset_index(drop=True, inplace=True)

    return long


def get_filter_options(df: pd.DataFrame) -> dict:
    """Return sorted unique values for each dropdown filter."""
    return {
        "branches":   sorted(df["Branch"].dropna().unique().tolist()),
        "districts":  sorted(df["DISTRICT"].dropna().unique().tolist()),
        "categories": sorted(df["Category"].dropna().unique().tolist()),
    }
