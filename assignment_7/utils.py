import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Weight serialisation helpers ──────────────────────────────────────────────

def get_model_weights(model):
    """Return model state-dict as plain Python lists (JSON-serialisable)."""
    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.cpu().numpy().tolist()
    return weights

def set_model_weights(model, weights):
    """Load a weights dict (lists) back into a model."""
    state_dict = {}
    for k, v in weights.items():
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)

def federated_average(weight_list):
    """Compute element-wise average of a list of weight dicts."""
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = np.mean(
            [np.array(w[key]) for w in weight_list], axis=0
        ).tolist()
    return avg_weights

# ── Dataset helpers ───────────────────────────────────────────────────────────

SMOKING_CATEGORIES = ["No Info", "never", "former", "current", "not current", "ever"]

def preprocess_diabetes_data(df: pd.DataFrame):
    """
    Encode categorical columns and scale numeric columns.

    Returns
    -------
    X : torch.Tensor  shape (N, 10)
    y : torch.Tensor  shape (N,)   long
    """
    df = df.copy()

    # --- categorical encoding ---
    df["gender"] = LabelEncoder().fit_transform(df["gender"].astype(str))

    # Ordinal encode smoking_history with a fixed mapping so all clients
    # produce the same encoding even when their local data is a subset.
    smoking_map = {cat: idx for idx, cat in enumerate(SMOKING_CATEGORIES)}
    df["smoking_history"] = (
        df["smoking_history"]
        .astype(str)
        .map(smoking_map)
        .fillna(0)
        .astype(int)
    )

    feature_cols = [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["diabetes"].values.astype(np.int64)

    # --- scale continuous columns (indices 1, 5, 6, 7) ---
    scaler = StandardScaler()
    cont_idx = [1, 5, 6, 7]          # age, bmi, HbA1c_level, blood_glucose_level
    X[:, cont_idx] = scaler.fit_transform(X[:, cont_idx])

    return torch.tensor(X), torch.tensor(y)
