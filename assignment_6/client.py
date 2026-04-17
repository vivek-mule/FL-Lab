"""
Vertical Federated Learning – Client (Party)
===============================================
Each party holds a **BottomModel** and a vertical slice of features.

Per training step the party:
  1. Computes an embedding from its local features (forward through BottomModel).
  2. Sends the embedding to the server – **raw data never leaves the party**.
  3. Receives the gradient of the loss w.r.t. its embedding from the server.
  4. Back-propagates through the BottomModel and updates its local weights.
"""

import torch
import torch.optim as optim
import requests

from model import BottomModel
from utils import serialize_tensor, deserialize_tensor

SERVER_URL = "http://127.0.0.1:5000"


class VFLClient:
    """Vertical Federated Learning client representing one data party."""

    def __init__(self, client_id, input_dim, embedding_dim=32, lr=0.01):
        self.client_id = client_id
        self.model     = BottomModel(input_dim=input_dim, output_dim=embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self._last_embedding = None      # kept alive for backward pass

    # ── Forward ───────────────────────────────────────────────────────────

    def compute_embedding(self, X):
        """
        Run BottomModel on local features.
        The computation graph is retained so that ``apply_gradient``
        can back-propagate through it later.
        """
        self.optimizer.zero_grad()
        self._last_embedding = self.model(X)
        return self._last_embedding

    def send_embedding(self, embedding):
        """Upload the embedding to the VFL server."""
        payload = {
            "client_id" : self.client_id,
            "embedding" : serialize_tensor(embedding),
        }
        resp = requests.post(f"{SERVER_URL}/upload_embedding",
                             json=payload, timeout=10)
        return resp.json()

    # ── Backward ──────────────────────────────────────────────────────────

    def fetch_gradient(self):
        """Pull the gradient of the loss w.r.t. this party's embedding."""
        resp = requests.get(
            f"{SERVER_URL}/get_gradient/{self.client_id}", timeout=10)
        return deserialize_tensor(resp.json()["gradient"])

    def apply_gradient(self, grad):
        """Back-propagate the received gradient through BottomModel."""
        self._last_embedding.backward(gradient=grad)
        self.optimizer.step()

    # ── Inference (no gradient) ───────────────────────────────────────────

    def get_embedding_no_grad(self, X):
        """Compute embedding without building a computation graph."""
        self.model.eval()
        with torch.no_grad():
            emb = self.model(X)
        self.model.train()
        return emb
