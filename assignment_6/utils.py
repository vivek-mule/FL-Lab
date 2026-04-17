"""
Vertical Federated Learning – Utilities
=========================================
Helper functions for:
  • Loading the Heart Disease CSV and splitting features **vertically**
    among 3 parties (hospitals)
  • Tensor serialisation / deserialisation for JSON transport

Dataset columns (heart_disease.csv – 1 025 rows, 14 columns)
-------------------------------------------------------------
age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target

Vertical feature split
----------------------
  Party 0 (Hospital A): age, sex, cp, trestbps, chol      → 5 features
  Party 1 (Hospital B): fbs, restecg, thalach, exang       → 4 features
  Party 2 (Hospital C): oldpeak, slope, ca, thal           → 4 features
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Explicit vertical feature assignment for 3 parties ────────────────────────
PARTY_FEATURES = {
    "party_0": ["age", "sex", "cp", "trestbps", "chol"],        # demographics & blood
    "party_1": ["fbs", "restecg", "thalach", "exang"],            # cardiac function tests
    "party_2": ["oldpeak", "slope", "ca", "thal"],                # stress test & imaging
}


# ── Data loading & vertical partitioning ──────────────────────────────────────

def load_and_split_data(num_parties=3, test_size=0.2, seed=42):
    """
    Load ``heart_disease.csv`` and split features vertically among 3 parties.

    The CSV is expected in the **same directory** as this file.

    Returns
    -------
    party_train   : dict  – {party_id: X_train tensor}
    party_test    : dict  – {party_id: X_test  tensor}
    y_train       : Tensor
    y_test        : Tensor
    feature_dims  : dict  – {party_id: number of features}
    feature_names : dict  – {party_id: list of column names}
    """
    csv_path = os.path.join(os.path.dirname(__file__), "heart_disease.csv")
    df = pd.read_csv(csv_path)

    # Drop duplicate rows – the 1025-row Kaggle version contains ~70%
    # duplicates which cause data leakage between train and test splits.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[Data] {len(df)} unique samples after deduplication.")

    y = df["target"].values

    # Train / test split (stratified) – split indices first
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )

    party_train   = {}
    party_test    = {}
    feature_dims  = {}
    feature_names = {}

    for pid, cols in PARTY_FEATURES.items():
        X_cols = df[cols].values.astype(np.float64)

        # Standardise each party's features independently
        scaler = StandardScaler()
        X_cols = scaler.fit_transform(X_cols)

        party_train[pid]   = torch.tensor(X_cols[idx_train], dtype=torch.float32)
        party_test[pid]    = torch.tensor(X_cols[idx_test],  dtype=torch.float32)
        feature_dims[pid]  = len(cols)
        feature_names[pid] = cols

    y_train = torch.tensor(y[idx_train], dtype=torch.long)
    y_test  = torch.tensor(y[idx_test],  dtype=torch.long)

    return party_train, party_test, y_train, y_test, feature_dims, feature_names


# ── Serialisation helpers ─────────────────────────────────────────────────────

def serialize_tensor(tensor):
    """Convert a PyTorch tensor to a JSON-friendly nested Python list."""
    return tensor.detach().cpu().numpy().tolist()


def deserialize_tensor(data):
    """Convert a nested Python list back to a PyTorch float tensor."""
    return torch.tensor(data, dtype=torch.float32)
