"""
Vertical Federated Learning – Runner
======================================
Orchestrates the full VFL training pipeline with **3 clients (parties)**:

  1. Starts the Flask server in a background thread.
  2. Loads ``heart_disease.csv`` and splits 13 features **vertically**
     among 3 parties:
       • Party 0 (Hospital A) → age, sex, cp, trestbps, chol
       • Party 1 (Hospital B) → fbs, restecg, thalach, exang
       • Party 2 (Hospital C) → oldpeak, slope, ca, thal
  3. Creates one VFLClient per party (each with its own BottomModel).
  4. Runs multiple training epochs.  In every mini-batch:
       a. Each party computes an embedding and sends it to the server.
       b. The server concatenates embeddings, runs the TopModel,
          computes the loss, back-propagates, and stores per-party gradients.
       c. Each party fetches its gradient and updates its BottomModel.
  5. Evaluates the joint model on held-out test data.

Usage
-----
    cd assignment_6
    python run_vfl.py
"""

import time
import threading
import warnings

import numpy as np
import requests

from utils import load_and_split_data, serialize_tensor
from client import VFLClient

# Suppress Flask / urllib connection warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SERVER_URL    = "http://127.0.0.1:5000"
NUM_PARTIES   = 3
EMBEDDING_DIM = 32
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def start_server():
    """Import and run the Flask server in a daemon thread."""
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)          # silence per-request logs

    from server import app
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def wait_for_server(url=SERVER_URL, timeout=15):
    """Block until the server responds to /health."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{url}/health", timeout=1)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(0.3)
    raise RuntimeError("Server did not start in time.")


def init_server_model(total_emb_dim):
    """Tell the server to create its TopModel."""
    resp = requests.post(f"{SERVER_URL}/init", json={
        "total_embedding_dim": total_emb_dim,
        "num_parties"        : NUM_PARTIES,
        "lr"                 : LEARNING_RATE,
    }, timeout=5)
    return resp.json()


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(clients, party_train, y_train):
    """Run one full epoch of VFL training with mini-batches."""
    n_samples = len(y_train)
    indices   = np.random.permutation(n_samples)

    total_loss = 0.0
    total_correct = 0
    total_count   = 0

    for start in range(0, n_samples, BATCH_SIZE):
        batch_idx = indices[start : start + BATCH_SIZE]

        # 1) Each party computes & sends its embedding
        for pid, client in clients.items():
            X_batch = party_train[pid][batch_idx]
            emb     = client.compute_embedding(X_batch)
            client.send_embedding(emb)

        # 2) Send labels for this batch
        labels = y_train[batch_idx].tolist()
        requests.post(f"{SERVER_URL}/set_batch_labels",
                      json={"labels": labels}, timeout=5)

        # 3) Trigger forward + backward on the server
        resp = requests.get(f"{SERVER_URL}/forward_backward", timeout=10).json()
        batch_size_actual = len(batch_idx)
        total_loss    += resp["loss"]     * batch_size_actual
        total_correct += resp["accuracy"] * batch_size_actual
        total_count   += batch_size_actual

        # 4) Each party fetches its gradient and updates its BottomModel
        for pid, client in clients.items():
            grad = client.fetch_gradient()
            client.apply_gradient(grad)

    return total_loss / total_count, total_correct / total_count


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clients, party_test, y_test):
    """Evaluate the joint VFL model on the test set (no weight updates)."""
    sorted_pids = sorted(clients.keys())

    # Each party computes its embedding (no gradient)
    for pid in sorted_pids:
        emb = clients[pid].get_embedding_no_grad(party_test[pid])
        payload = {
            "client_id" : pid,
            "embedding" : serialize_tensor(emb),
        }
        requests.post(f"{SERVER_URL}/upload_embedding",
                      json=payload, timeout=10)

    # Send test labels
    requests.post(f"{SERVER_URL}/set_batch_labels",
                  json={"labels": y_test.tolist()}, timeout=5)

    # Forward only (no backward / no optimizer step)
    resp = requests.get(f"{SERVER_URL}/predict", timeout=10).json()
    return resp["loss"], resp["accuracy"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Vertical Federated Learning – Joint Feature-Based Prediction")
    print("=" * 62)

    # 1. Start the Flask server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    wait_for_server()
    print("[Runner] Server is up.\n")

    # 2. Load data and split features vertically among 3 parties
    party_train, party_test, y_train, y_test, feature_dims, feature_names = \
        load_and_split_data(num_parties=NUM_PARTIES)

    print(f"Dataset : Heart Disease (heart_disease.csv)  –  "
          f"{len(y_train)} train / {len(y_test)} test samples")
    print(f"Parties : {NUM_PARTIES}")
    for pid, dim in feature_dims.items():
        print(f"  {pid} → {dim} features  {feature_names[pid]}")
    print()

    # 3. Initialise the server's TopModel
    total_emb_dim = NUM_PARTIES * EMBEDDING_DIM
    init_server_model(total_emb_dim)

    # 4. Create one VFLClient per party
    clients = {}
    for pid, dim in feature_dims.items():
        clients[pid] = VFLClient(
            client_id     = pid,
            input_dim     = dim,
            embedding_dim = EMBEDDING_DIM,
            lr            = LEARNING_RATE,
        )
    print(f"[Runner] Created {len(clients)} VFL clients.\n")

    # 5. Training loop
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>10}")
    print("-" * 32)
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(clients, party_train, y_train)
        print(f"{epoch:5d}  {loss:10.4f}  {acc:10.4f}")

    # 6. Evaluation on held-out test data
    print("\n[Runner] Evaluating on test set …")
    test_loss, test_acc = evaluate(clients, party_test, y_test)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
