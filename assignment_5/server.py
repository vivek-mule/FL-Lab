"""
Federated Learning – Server
============================
Aggregates model updates from multiple clients using **weighted** federated
averaging and distributes the resulting global model.

Endpoints
---------
POST /upload        – receive weights + sample count from a client
GET  /aggregate     – weighted-average all updates → new global model
GET  /global_model  – return the current global model weights to clients
GET  /metrics       – return latest per-client training metrics
"""

from flask import Flask, request, jsonify
from model import SimpleNN
from utils import weighted_federated_average, set_model_weights, get_model_weights

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
global_model   = SimpleNN(input_dim=10)
client_updates = []        # list of {"weights": …, "num_samples": …}
client_metrics = []        # list of {"client_id": …, "accuracy": …, "loss": …}
round_number   = 0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def receive_update():
    """
    Receive local model weights + metadata from a client.

    Expected JSON payload::

        {
            "client_id"  : "client_1",
            "weights"    : { ... },
            "num_samples": 300,
            "accuracy"   : 0.87,
            "loss"       : 0.35
        }
    """
    global client_updates, client_metrics

    payload     = request.get_json()
    weights     = payload.get("weights", {})
    num_samples = payload.get("num_samples", 0)
    client_id   = payload.get("client_id", "unknown")
    accuracy    = payload.get("accuracy", None)
    loss        = payload.get("loss", None)

    client_updates.append({
        "weights"    : weights,
        "num_samples": num_samples
    })

    if accuracy is not None:
        client_metrics.append({
            "client_id"  : client_id,
            "accuracy"   : accuracy,
            "loss"       : loss,
            "num_samples": num_samples
        })

    print(f"[Server] Received update from {client_id} "
          f"(n={num_samples}).  Total updates this round: {len(client_updates)}")

    return jsonify({"status": "received", "total_updates": len(client_updates)})


@app.route("/aggregate", methods=["GET"])
def aggregate():
    """
    Perform **weighted** federated averaging over all received client
    updates and refresh the global model.  Weights are proportional to
    each client's number of training samples.
    """
    global client_updates, global_model, round_number

    if not client_updates:
        return jsonify({"error": "No client updates received yet."}), 400

    # ── Weighted FedAvg ───────────────────────────────────────────────────
    sample_counts = [u["num_samples"] for u in client_updates]
    total_samples = sum(sample_counts)
    print(f"[Server] Aggregating {len(client_updates)} client updates  "
          f"(total samples = {total_samples})")
    for i, u in enumerate(client_updates):
        pct = 100.0 * u["num_samples"] / total_samples
        print(f"  Client {i+1}: {u['num_samples']} samples  "
              f"→ weight {pct:.1f}%")

    avg_weights = weighted_federated_average(client_updates)
    set_model_weights(global_model, avg_weights)

    round_number  += 1
    client_updates = []        # reset for next round

    print(f"[Server] Global model updated – round {round_number}.")

    return jsonify({
        "status"       : "aggregated",
        "round"        : round_number,
        "total_samples": total_samples,
        "global_weights": get_model_weights(global_model)
    })


@app.route("/global_model", methods=["GET"])
def global_model_weights():
    """Distribute the current global model weights to any requesting client."""
    return jsonify(get_model_weights(global_model))


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return per-client training metrics from the last round."""
    if not client_metrics:
        return jsonify({"message": "No metrics available yet."})

    avg_acc  = sum(m["accuracy"] for m in client_metrics) / len(client_metrics)
    avg_loss = sum(m["loss"]     for m in client_metrics) / len(client_metrics)

    return jsonify({
        "round"       : round_number,
        "clients"     : client_metrics,
        "avg_accuracy": round(avg_acc, 4),
        "avg_loss"    : round(avg_loss, 4)
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[Server] Federated Learning Server (Weighted FedAvg) "
          "starting on port 5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False)
