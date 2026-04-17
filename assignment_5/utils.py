"""
Federated Learning – Utilities
================================
Helper functions for:
  • serialising / deserialising model weights (JSON-friendly)
  • weighted federated averaging  (FedAvg with sample-count weights)
"""

import torch
import numpy as np


# ── Weight serialisation helpers ──────────────────────────────────────────────

def get_model_weights(model):
    """Return model state-dict as plain Python lists (JSON-serialisable)."""
    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.cpu().numpy().tolist()
    return weights


def set_model_weights(model, weights):
    """Load a weights dict (plain lists) back into a PyTorch model."""
    state_dict = {}
    for k, v in weights.items():
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)


# ── Aggregation ───────────────────────────────────────────────────────────────

def weighted_federated_average(client_updates):
    """
    Compute a **weighted** average of client model weights.

    Each entry in *client_updates* is a dict with:
        "weights"     – model state-dict (lists)
        "num_samples" – number of local training samples

    The global weight for every parameter is:

        w_global = Σ_k (n_k / N) * w_k

    where n_k is the sample count of client k and N = Σ n_k.

    Returns
    -------
    avg_weights : dict   – aggregated weight dict (plain lists)
    """
    total_samples = sum(u["num_samples"] for u in client_updates)

    # Initialise accumulator with zeros
    keys = list(client_updates[0]["weights"].keys())
    avg_weights = {key: np.zeros_like(np.array(client_updates[0]["weights"][key]))
                   for key in keys}

    for update in client_updates:
        weight_factor = update["num_samples"] / total_samples
        for key in keys:
            avg_weights[key] += weight_factor * np.array(update["weights"][key])

    # Convert numpy arrays back to plain lists for JSON serialisation
    for key in keys:
        avg_weights[key] = avg_weights[key].tolist()

    return avg_weights
