"""
Federated Learning – Client (Healthcare Institution)
=====================================================
Each client represents a hospital that holds a private shard of the
diabetes dataset.  The client:
  1. Loads its local data partition.
  2. Optionally pulls the latest global model from the server.
  3. Trains locally for several epochs.
  4. Sends the updated weights (+ metrics) to the server – NO raw data is
     ever transmitted, preserving patient privacy.

Usage
-----
  python client.py --client_id hospital_A --partition 0 --num_partitions 3
  python client.py --client_id hospital_B --partition 1 --num_partitions 3
  python client.py --client_id hospital_C --partition 2 --num_partitions 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from model import DiabetesNN
from utils import get_model_weights, set_model_weights, preprocess_diabetes_data

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_URL    = "http://127.0.0.1:5000"
DATASET_PATH  = os.path.join(os.path.dirname(__file__),
                              "diabetes_prediction_dataset.csv")
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.001
BATCH_SIZE    = 64


# ── Data loading ──────────────────────────────────────────────────────────────

def load_partition(partition_id: int, num_partitions: int):
    """
    Split the CSV into `num_partitions` equal shards and return the shard
    identified by `partition_id`.  This simulates separate hospital databases.
    """
    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

    shard_size = len(df) // num_partitions
    start = partition_id * shard_size
    end   = start + shard_size if partition_id < num_partitions - 1 else len(df)

    shard = df.iloc[start:end]
    print(f"[Client] Loaded partition {partition_id}/{num_partitions}  "
          f"({len(shard)} records, "
          f"{shard['diabetes'].sum()} diabetic / "
          f"{(~shard['diabetes'].astype(bool)).sum()} non-diabetic)")
    return shard


# ── Server communication ──────────────────────────────────────────────────────

def pull_global_model(model: DiabetesNN):
    """Fetch latest global weights from the server and load them locally."""
    try:
        resp = requests.get(f"{SERVER_URL}/global_model", timeout=5)
        if resp.status_code == 200:
            set_model_weights(model, resp.json())
            print("[Client] Pulled global model from server.")
        else:
            print("[Client] Could not pull global model – starting fresh.")
    except requests.exceptions.ConnectionError:
        print("[Client] Server unreachable – training from scratch.")


def send_update(model: DiabetesNN, client_id: str,
                accuracy: float, loss: float):
    """Send local weights + training metrics to the server."""
    payload = {
        "client_id": client_id,
        "weights"  : get_model_weights(model),
        "accuracy" : round(accuracy, 4),
        "loss"     : round(loss, 4)
    }
    try:
        resp = requests.post(f"{SERVER_URL}/upload", json=payload, timeout=10)
        print(f"[Client] Update sent – server response: {resp.json()}")
    except requests.exceptions.ConnectionError as exc:
        print(f"[Client] Failed to reach server: {exc}")
        sys.exit(1)


# ── Local training ─────────────────────────────────────────────────────────────

def train_local_model(model: DiabetesNN, X: torch.Tensor, y: torch.Tensor):
    """
    Train the model on local data for LOCAL_EPOCHS epochs.

    Returns
    -------
    model     : updated DiabetesNN
    final_loss: float
    accuracy  : float  (on local training set)
    """
    dataset   = torch.utils.data.TensorDataset(X, y)
    loader    = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{LOCAL_EPOCHS}  loss={avg_loss:.4f}")

    # ── Evaluate on local data ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds  = torch.argmax(logits, dim=1).numpy()

    acc      = accuracy_score(y.numpy(), preds)
    fin_loss = nn.CrossEntropyLoss()(model(X), y).item()

    print(f"[Client] Local training done – "
          f"accuracy={acc:.4f}  loss={fin_loss:.4f}")
    return model, fin_loss, acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning client for diabetes prediction.")
    parser.add_argument("--client_id",       default="hospital_A",
                        help="Unique identifier for this institution.")
    parser.add_argument("--partition",       type=int, default=0,
                        help="Which data shard this client holds (0-indexed).")
    parser.add_argument("--num_partitions",  type=int, default=3,
                        help="Total number of data shards / clients.")
    parser.add_argument("--rounds",          type=int, default=3,
                        help="Number of federated learning rounds to run.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Federated Learning Client  |  {args.client_id}")
    print(f"  Healthcare Institution – Diabetes Prediction")
    print(f"{'='*60}\n")

    # Load local data partition (simulates isolated hospital database)
    df   = load_partition(args.partition, args.num_partitions)
    X, y = preprocess_diabetes_data(df)

    model = DiabetesNN(input_dim=X.shape[1])

    for fl_round in range(1, args.rounds + 1):
        print(f"\n--- Federated Round {fl_round}/{args.rounds} ---")

        # Pull current global model before training
        pull_global_model(model)

        # Train locally (raw data NEVER leaves this client)
        model, loss, accuracy = train_local_model(model, X, y)

        # Send only the model weights (not the data) to the server
        send_update(model, args.client_id, accuracy, loss)

    print(f"\n[Client] {args.client_id} finished all {args.rounds} rounds.")


if __name__ == "__main__":
    main()
