"""
Vertical Federated Learning – Server
======================================
Flask-based coordinator that hosts the **TopModel**.

In vertical FL the server never sees raw features.  Instead it receives
fixed-size *embeddings* from each party, concatenates them, and runs
the TopModel to produce predictions.  Gradients w.r.t. each party's
embedding are sent back so that parties can update their local
BottomModels.

Endpoints
---------
GET  /health             – liveness check
POST /init               – create the TopModel with given dimensions
POST /set_batch_labels   – receive ground-truth labels for the current batch
POST /upload_embedding   – receive one party's embedding
GET  /forward_backward   – run TopModel forward + backward, store gradients
GET  /predict            – forward-only pass (evaluation, no weight update)
GET  /get_gradient/<id>  – return the gradient for a specific party
GET  /metrics            – return the full training-loss / accuracy history
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn

from model import TopModel
from utils import serialize_tensor, deserialize_tensor

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
top_model     = None
top_optimizer = None
criterion     = nn.CrossEntropyLoss()

embeddings    = {}       # client_id → embedding tensor (requires_grad)
gradients     = {}       # client_id → serialised gradient list
batch_labels  = None
num_parties   = 3
metrics_log   = []


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Initialisation ────────────────────────────────────────────────────────────

@app.route("/init", methods=["POST"])
def init_server():
    """Create the TopModel with the total embedding dimension."""
    global top_model, top_optimizer, num_parties, metrics_log

    config        = request.get_json()
    total_emb_dim = config["total_embedding_dim"]
    num_parties   = config.get("num_parties", 2)
    lr            = config.get("lr", 0.01)

    top_model     = TopModel(input_dim=total_emb_dim, num_classes=2)
    top_optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)
    metrics_log   = []

    print(f"[Server] TopModel initialised  "
          f"(input_dim={total_emb_dim}, parties={num_parties})")
    return jsonify({"status": "initialised"})


# ── Receive labels ────────────────────────────────────────────────────────────

@app.route("/set_batch_labels", methods=["POST"])
def set_batch_labels():
    """Store the ground-truth labels for the current mini-batch."""
    global batch_labels
    data         = request.get_json()
    batch_labels = torch.tensor(data["labels"], dtype=torch.long)
    return jsonify({"status": "labels_set", "count": len(batch_labels)})


# ── Receive embeddings ────────────────────────────────────────────────────────

@app.route("/upload_embedding", methods=["POST"])
def upload_embedding():
    """Receive one party's embedding for the current batch."""
    global embeddings
    data      = request.get_json()
    client_id = data["client_id"]

    emb = deserialize_tensor(data["embedding"])
    emb.requires_grad_(True)
    embeddings[client_id] = emb

    all_ready = len(embeddings) == num_parties
    print(f"[Server] Embedding from {client_id}  "
          f"({len(embeddings)}/{num_parties})")
    return jsonify({"status": "received", "all_ready": all_ready})


# ── Forward + backward (training) ────────────────────────────────────────────

@app.route("/forward_backward", methods=["GET"])
def forward_backward():
    """
    Concatenate all embeddings → TopModel forward → loss → backward.
    Stores per-party gradients so clients can fetch them.
    """
    global embeddings, gradients, top_model, top_optimizer, batch_labels

    if len(embeddings) < num_parties:
        return jsonify({"error": "Not all embeddings received yet."}), 400

    sorted_keys = sorted(embeddings.keys())
    concat      = torch.cat([embeddings[k] for k in sorted_keys], dim=1)

    # Forward
    top_optimizer.zero_grad()
    output = top_model(concat)
    loss   = criterion(output, batch_labels)

    # Backward
    loss.backward()
    top_optimizer.step()

    # Metrics
    preds = torch.argmax(output, dim=1)
    acc   = (preds == batch_labels).float().mean().item()

    # Store gradients for each party
    for key in sorted_keys:
        gradients[key] = serialize_tensor(embeddings[key].grad)

    result = {"loss": round(loss.item(), 4), "accuracy": round(acc, 4)}
    metrics_log.append(result)
    embeddings = {}                 # reset for next batch

    print(f"[Server] Train step – loss={result['loss']}, acc={result['accuracy']}")
    return jsonify({"status": "done", **result})


# ── Forward only (evaluation) ────────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict():
    """Forward-only pass – no gradient update.  Used for evaluation."""
    global embeddings

    if len(embeddings) < num_parties:
        return jsonify({"error": "Not all embeddings received yet."}), 400

    sorted_keys = sorted(embeddings.keys())
    concat      = torch.cat([embeddings[k] for k in sorted_keys], dim=1)

    top_model.eval()
    with torch.no_grad():
        output = top_model(concat)
        loss   = criterion(output, batch_labels)
        preds  = torch.argmax(output, dim=1)
        acc    = (preds == batch_labels).float().mean().item()
    top_model.train()

    embeddings = {}                 # reset
    return jsonify({
        "loss"       : round(loss.item(), 4),
        "accuracy"   : round(acc, 4),
        "predictions": preds.tolist()
    })


# ── Gradient retrieval ────────────────────────────────────────────────────────

@app.route("/get_gradient/<client_id>", methods=["GET"])
def get_gradient(client_id):
    """Return and pop the stored gradient for *client_id*."""
    if client_id not in gradients:
        return jsonify({"error": f"No gradient for {client_id}"}), 404
    grad = gradients.pop(client_id)
    return jsonify({"gradient": grad})


# ── Metrics history ──────────────────────────────────────────────────────────

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return accumulated training metrics."""
    return jsonify({"metrics": metrics_log})


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Vertical Federated Learning – Server")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
