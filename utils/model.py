"""
model.py — Train a RandomForestClassifier on the long-format MHT-CET data.

Target  : Admit = 1  if cutoff >= mean cutoff for that category (competitive seat)
Features: Cutoff, Branch_enc, District_enc, Category_enc
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

MODEL_PATH = "utils/model.pkl"


def _build_encoders(df: pd.DataFrame):
    """Fit label encoders for categorical features."""
    enc_branch    = LabelEncoder().fit(df["Branch"].astype(str))
    enc_district  = LabelEncoder().fit(df["DISTRICT"].astype(str))
    enc_category  = LabelEncoder().fit(df["Category"].astype(str))
    return enc_branch, enc_district, enc_category


def _build_features(df: pd.DataFrame, enc_branch, enc_district, enc_category):
    """Return feature matrix X and target y."""
    branch_enc   = enc_branch.transform(df["Branch"].astype(str))
    district_enc = enc_district.transform(df["DISTRICT"].astype(str))
    category_enc = enc_category.transform(df["Category"].astype(str))

    X = np.column_stack([
        df["Cutoff"].values,
        branch_enc,
        district_enc,
        category_enc,
    ])

    # Target: 1 if cutoff is >= mean cutoff across the whole dataset
    # (high cutoff = competitive = popular / good seat → "admitted" proxy)
    mean_cutoff = df["Cutoff"].mean()
    y = (df["Cutoff"] >= mean_cutoff).astype(int).values

    return X, y


def train_model(df: pd.DataFrame):
    """
    Train and persist a RandomForestClassifier.
    Returns (model, enc_branch, enc_district, enc_category).
    """
    enc_branch, enc_district, enc_category = _build_encoders(df)
    X, y = _build_features(df, enc_branch, enc_district, enc_category)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Persist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, enc_branch, enc_district, enc_category), f)

    return clf, enc_branch, enc_district, enc_category


def load_model():
    """Load persisted model and encoders (raises if not found)."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_model(df: pd.DataFrame):
    """Return (model, encoders) — loads from disk if available, else trains."""
    try:
        return load_model()
    except FileNotFoundError:
        return train_model(df)
