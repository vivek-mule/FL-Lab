from flask import Flask, request, jsonify
from model import SimpleNN
from utils import federated_average, set_model_weights, get_model_weights

app = Flask(__name__)

global_model = SimpleNN()
client_updates = []

@app.route("/upload", methods=["POST"])
def receive_update():
    global client_updates
    
    weights = request.get_json()
    client_updates.append(weights)
    
    print(f"Received update from client. Total updates: {len(client_updates)}")
    
    return jsonify({"status": "received"})

@app.route("/aggregate", methods=["GET"])
def aggregate():
    global client_updates, global_model
    
    if len(client_updates) == 0:
        return jsonify({"error": "No updates received"})
    
    avg_weights = federated_average(client_updates)
    set_model_weights(global_model, avg_weights)
    
    client_updates = []  # reset after aggregation
    
    print("Global model updated!")
    
    return jsonify(get_model_weights(global_model))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
