"""
predictor.py — MHT-CET Admission System (v2)

New recommendation logic:
  - DREAM  : cutoff > user_percentile  (aspirational, sorted desc by cutoff)
  - TARGET : cutoff within ±2 of user_percentile (sorted by proximity)
  - SAFE   : cutoff < user_percentile  (sorted desc by cutoff → closest first)

Final list: ~5-7 Dream · ~8-10 Target · remaining Safe  (default 30 total)
Probability is computed and shown but is NOT the primary sort key.
"""

import numpy as np
import pandas as pd

# ── Branch fallback chain ─────────────────────────────────────────────────────
BRANCH_FALLBACK = {
    "COMPUTER SCIENCE AND ENGINEERING": [
        "INFORMATION TECHNOLOGY",
        "ELECTRONICS AND TELECOMMUNICATION ENGG",
    ],
    "INFORMATION TECHNOLOGY": [
        "COMPUTER SCIENCE AND ENGINEERING",
        "ELECTRONICS AND TELECOMMUNICATION ENGG",
    ],
    "ELECTRONICS AND TELECOMMUNICATION ENGG": [
        "ELECTRICAL ENGINEERING",
        "INSTRUMENTATION ENGINEERING",
    ],
    "MECHANICAL ENGINEERING": [
        "CIVIL ENGINEERING",
        "PRODUCTION ENGINEERING",
    ],
}

# Target band: cutoffs within this many percentile points of user score
TARGET_BAND = 2.0


# ── Model scoring ─────────────────────────────────────────────────────────────
def _run_model(subset: pd.DataFrame, percentile: float,
               clf, enc_branch, enc_district, enc_category) -> pd.DataFrame:
    """Attach RF probability to every row. Does NOT sort."""
    if subset.empty:
        return subset

    def safe_encode(enc, series):
        known = set(enc.classes_)
        mapped = series.map(lambda v: v if v in known else enc.classes_[0])
        return enc.transform(mapped)

    X = np.column_stack([
        subset["Cutoff"].values,
        safe_encode(enc_branch,   subset["Branch"].astype(str)),
        safe_encode(enc_district, subset["DISTRICT"].astype(str)),
        safe_encode(enc_category, subset["Category"].astype(str)),
    ])

    proba = clf.predict_proba(X)[:, 1]

    # Gap-adjusted probability for display (clamped 0–1)
    gap   = percentile - subset["Cutoff"].values
    score = np.clip(proba + np.clip(gap / 100, -0.3, 0.3), 0.0, 1.0)

    out = subset.copy()
    out["RF_Prob"]    = proba
    out["Admit_Prob"] = score
    return out


# ── CAP-style bucket builder ──────────────────────────────────────────────────
def _build_buckets(scored: pd.DataFrame, percentile: float,
                   n_dream: int, n_target: int, n_safe: int) -> pd.DataFrame:
    """
    Split scored rows into Dream / Target / Safe buckets and
    pick the requested count from each, then return combined.
    """
    cutoffs = scored["Cutoff"].values

    dream_mask  = cutoffs > percentile
    safe_mask   = cutoffs < (percentile - TARGET_BAND)
    target_mask = (~dream_mask) & (~safe_mask)   # within band below + the band itself

    dream_df  = scored[dream_mask].copy()
    target_df = scored[target_mask].copy()
    safe_df   = scored[safe_mask].copy()

    # ── Sort each bucket ──────────────────────────────────────────────────────
    # DREAM  : highest cutoff first (hardest → easiest within dream)
    dream_df.sort_values("Cutoff", ascending=False, inplace=True)

    # TARGET : closest to user percentile first (abs gap ascending)
    target_df["_gap"] = (target_df["Cutoff"] - percentile).abs()
    target_df.sort_values(["_gap", "Admit_Prob"], ascending=[True, False], inplace=True)
    target_df.drop(columns="_gap", inplace=True)

    # SAFE   : highest cutoff first (closest realistic colleges first)
    safe_df.sort_values("Cutoff", ascending=False, inplace=True)

    # ── Pick counts, fill shortfalls from other buckets ──────────────────────
    dream_pick  = dream_df.head(n_dream)
    target_pick = target_df.head(n_target)
    safe_pick   = safe_df.head(n_safe)

    shortage = (n_dream  - len(dream_pick)  +
                n_target - len(target_pick) +
                n_safe   - len(safe_pick))

    # Fill shortage from whichever bucket has surplus, priority: safe → target → dream
    if shortage > 0:
        extra_safe   = safe_df.iloc[len(safe_pick):len(safe_pick) + shortage]
        safe_pick    = pd.concat([safe_pick, extra_safe])
        shortage     = max(0, shortage - len(extra_safe))

    if shortage > 0:
        extra_target = target_df.iloc[len(target_pick):len(target_pick) + shortage]
        target_pick  = pd.concat([target_pick, extra_target])
        shortage     = max(0, shortage - len(extra_target))

    if shortage > 0:
        extra_dream  = dream_df.iloc[len(dream_pick):len(dream_pick) + shortage]
        dream_pick   = pd.concat([dream_pick, extra_dream])

    # ── Tag types ─────────────────────────────────────────────────────────────
    dream_pick  = dream_pick.copy();  dream_pick["Type"]  = "🔥 Dream"
    target_pick = target_pick.copy(); target_pick["Type"] = "🎯 Target"
    safe_pick   = safe_pick.copy();   safe_pick["Type"]   = "✅ Safe"

    # ── Final order: Dream on top, then Target, then Safe ────────────────────
    combined = pd.concat([dream_pick, target_pick, safe_pick], ignore_index=True)
    return combined


# ── Public API ────────────────────────────────────────────────────────────────
def predict_colleges(
    df: pd.DataFrame,
    clf,
    enc_branch,
    enc_district,
    enc_category,
    percentile: float,
    category: str,
    branch: str,
    district: str | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Return CAP-style recommendations (Dream / Target / Safe).

    top_n is total results. Bucket split:
      Dream  ≈ 20 %  (min 5)
      Target ≈ 27 %  (min 8)
      Safe   = remainder
    """
    category = category.upper().strip()
    branch   = branch.upper().strip()
    district = district.upper().strip() if district else None

    # Bucket sizes
    n_dream  = max(5,  round(top_n * 0.20))
    n_target = max(8,  round(top_n * 0.27))
    n_safe   = max(1,  top_n - n_dream - n_target)

    # ── Filter: category → branch (+ fallback) → district (+ relax) ──────────
    filtered = df[df["Category"] == category].copy()
    if filtered.empty:
        return pd.DataFrame()

    branches_to_try = [branch] + BRANCH_FALLBACK.get(branch, [])
    final_subset = pd.DataFrame()

    for br in branches_to_try:
        br_df = filtered[filtered["Branch"] == br]
        if district:
            dist_df = br_df[br_df["DISTRICT"] == district]
            dist_df = dist_df if not dist_df.empty else br_df  # relax location
        else:
            dist_df = br_df
        if not dist_df.empty:
            final_subset = pd.concat([final_subset, dist_df])

    # Deduplicate (branch fallback may have added rows)
    final_subset = final_subset.drop_duplicates()

    # If still too few rows, relax: use all branches containing keyword
    MIN_ROWS = max(top_n, 30)
    if len(final_subset) < MIN_ROWS:
        keyword = branch.split()[0]  # e.g. "ELECTRONICS" from "ELECTRONICS ENGINEERING"
        broad = filtered[filtered["Branch"].str.contains(keyword, na=False)]
        if district:
            broad_d = broad[broad["DISTRICT"] == district]
            broad = broad_d if not broad_d.empty else broad
        final_subset = pd.concat([final_subset, broad]).drop_duplicates()

    if final_subset.empty:
        return pd.DataFrame()

    # ── Score with model ──────────────────────────────────────────────────────
    scored = _run_model(final_subset, percentile, clf,
                        enc_branch, enc_district, enc_category)

    # ── Build CAP buckets ─────────────────────────────────────────────────────
    result = _build_buckets(scored, percentile, n_dream, n_target, n_safe)

    # ── Final display columns ─────────────────────────────────────────────────
    result["Admit_Prob_%"] = (result["Admit_Prob"] * 100).round(1)

    display_cols = [
        "Institute Name", "Branch", "DISTRICT", "Category",
        "Cutoff", "Admit_Prob_%", "Type",
        "Fees", "Avg Placement", "Highest Placement",
    ]
    display_cols = [c for c in display_cols if c in result.columns]
    return result[display_cols].reset_index(drop=True)
