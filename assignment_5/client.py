"""
Federated Learning – Client
=============================
Each client holds a **private** local dataset of varying size.
The client:
  1. Optionally pulls the latest global model from the server.
  2. Trains locally for several epochs on its own data.
  3. Sends the updated weights, its **sample count**, and training metrics
     to the server — raw data is never transmitted.

The sample count allows the server to perform **weighted** federated
averaging, giving more influence to clients with larger datasets.

Usage
-----
  python client.py --client_id client_1 --num_samples 300 --rounds 3
  python client.py --client_id client_2 --num_samples 150 --rounds 3
  python client.py --client_id client_3 --num_samples 550 --rounds 3
"""

import argparse
import sys

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from model import SimpleNN
from utils import get_model_weights, set_model_weights

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_URL    = "http://127.0.0.1:5000"
LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.01
BATCH_SIZE    = 32
INPUT_DIM     = 10


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_local_data(num_samples: int, seed: int = 42):
    """
    Generate a synthetic binary-classification dataset.

    Different clients can use different sample counts to demonstrate
    that weighted FedAvg gives proportional influence.
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(num_samples, INPUT_DIM).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)   # simple rule
    return torch.tensor(X), torch.tensor(y)


# ── Server communication ──────────────────────────────────────────────────────

def pull_global_model(model: SimpleNN):
    """Fetch the latest global weights from the server and load them."""
    try:
        resp = requests.get(f"{SERVER_URL}/global_model", timeout=5)
        if resp.status_code == 200:
            set_model_weights(model, resp.json())
            print("[Client] Pulled global model from server.")
        else:
            print("[Client] Could not pull global model – using local weights.")
    except requests.exceptions.ConnectionError:
        print("[Client] Server unreachable – training from scratch.")


def send_update(model: SimpleNN, client_id: str,
                num_samples: int, accuracy: float, loss: float):
    """Send local weights + sample count + metrics to the server."""
    payload = {
        "client_id"  : client_id,
        "weights"    : get_model_weights(model),
        "num_samples": num_samples,
        "accuracy"   : round(accuracy, 4),
        "loss"       : round(loss, 4)
    }
    try:
        resp = requests.post(f"{SERVER_URL}/upload", json=payload, timeout=10)
        print(f"[Client] Update sent – server response: {resp.json()}")
    except requests.exceptions.ConnectionError as exc:
        print(f"[Client] Failed to reach server: {exc}")
        sys.exit(1)


# ── Local training ─────────────────────────────────────────────────────────────

def train_local_model(model: SimpleNN, X: torch.Tensor, y: torch.Tensor):
    """
    Train the model on local data for LOCAL_EPOCHS epochs.

    Returns
    -------
    model      : updated SimpleNN
    final_loss : float
    accuracy   : float
    """
    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset,
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
        description="Federated Learning client – Weighted FedAvg")
    parser.add_argument("--client_id",    default="client_1",
                        help="Unique identifier for this client.")
    parser.add_argument("--num_samples",  type=int, default=200,
                        help="Number of local training samples.")
    parser.add_argument("--rounds",       type=int, default=3,
                        help="Number of federated learning rounds.")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for data generation.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Federated Learning Client  |  {args.client_id}")
    print(f"  Weighted Federated Averaging – {args.num_samples} samples")
    print(f"{'='*60}\n")

    # Generate private local data (different sizes per client)
    X, y = generate_local_data(args.num_samples, seed=args.seed)
    print(f"[Client] Generated {args.num_samples} local samples  "
          f"(class 0: {(y == 0).sum().item()}, class 1: {(y == 1).sum().item()})")

    model = SimpleNN(input_dim=INPUT_DIM)

    for fl_round in range(1, args.rounds + 1):
        print(f"\n--- Federated Round {fl_round}/{args.rounds} ---")

        # 1. Pull current global model from the server
        pull_global_model(model)

        # 2. Train locally – raw data never leaves this client
        model, loss, accuracy = train_local_model(model, X, y)

        # 3. Send weights + sample count to server for weighted aggregation
        send_update(model, args.client_id, args.num_samples, accuracy, loss)

    print(f"\n[Client] {args.client_id} finished all {args.rounds} rounds.")


if __name__ == "__main__":
    main()
