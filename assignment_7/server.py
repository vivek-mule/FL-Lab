"""
Federated Learning – Server
============================
Aggregates model updates from multiple hospital clients and maintains
the global DiabetesNN model.

Endpoints
---------
POST /upload      – receive a weight update from a client
GET  /aggregate   – federated-average all received updates → new global model
GET  /global_model – return the current global model weights to clients
GET  /metrics     – return latest aggregated accuracy reported by clients
"""

from flask import Flask, request, jsonify
from model import DiabetesNN
from utils import federated_average, set_model_weights, get_model_weights

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
global_model   = DiabetesNN(input_dim=8)   # 8 features after one-hot encoding
client_updates = []                        # list of weight dicts
client_metrics = []                        # list of {accuracy, loss, client_id}
round_number   = 0

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def receive_update():
    """Receive local model weights + optional metrics from a client."""
    global client_updates, client_metrics

    payload   = request.get_json()
    weights   = payload.get("weights", {})
    client_id = payload.get("client_id", "unknown")
    accuracy  = payload.get("accuracy", None)
    loss      = payload.get("loss", None)

    client_updates.append(weights)
    if accuracy is not None:
        client_metrics.append({
            "client_id": client_id,
            "accuracy" : accuracy,
            "loss"     : loss
        })

    print(f"[Server] Received update from {client_id}. "
          f"Total updates this round: {len(client_updates)}")
    return jsonify({"status": "received", "total_updates": len(client_updates)})


@app.route("/aggregate", methods=["GET"])
def aggregate():
    """Federated-average all received updates and refresh the global model."""
    global client_updates, global_model, round_number

    if not client_updates:
        return jsonify({"error": "No client updates received yet."}), 400

    avg_weights = federated_average(client_updates)
    set_model_weights(global_model, avg_weights)

    round_number  += 1
    client_updates = []     # reset for the next round

    print(f"[Server] Global model updated – round {round_number}.")
    return jsonify({
        "status"      : "aggregated",
        "round"       : round_number,
        "global_weights": get_model_weights(global_model)
    })


@app.route("/global_model", methods=["GET"])
def global_model_weights():
    """Return the current global model weights so clients can pull them."""
    return jsonify(get_model_weights(global_model))


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return per-client metrics from the last round."""
    if not client_metrics:
        return jsonify({"message": "No metrics available yet."})

    avg_acc  = sum(m["accuracy"] for m in client_metrics) / len(client_metrics)
    avg_loss = sum(m["loss"]     for m in client_metrics) / len(client_metrics)

    return jsonify({
        "round"            : round_number,
        "clients"          : client_metrics,
        "avg_accuracy"     : round(avg_acc,  4),
        "avg_loss"         : round(avg_loss, 4)
    })


if __name__ == "__main__":
    print("[Server] Federated Learning Server starting on port 5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False)
