"""
Assignment 1
Name: Vivek Mule
Roll: 381072
PRN: 22420145


Problem Statement: Design and implement a basic simulation of a Federated Learning system in Python where 
multiple clients train local models on their own data without sharing raw datasets. A central 
server aggregates the locally trained model parameters using averaging to form a global model, 
demonstrating the core federated learning workflow.
"""


"""
Basic Federated Learning simulation.

Multiple clients train a simple logistic regression model locally on their own
data. A central server aggregates the client weights with FedAvg (arithmetic
mean) to produce a new global model. No raw data ever leaves the clients.
"""

from __future__ import annotations

import dataclasses
from typing import List, Sequence, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-z))


@dataclasses.dataclass
class Client:
	"""Client holds private data and performs local training."""

	client_id: int
	x: np.ndarray  # shape (n_samples, n_features)
	y: np.ndarray  # shape (n_samples,)
	learning_rate: float = 0.1
	local_epochs: int = 3

	def train(self, global_weights: np.ndarray) -> np.ndarray:
		"""Run local SGD starting from the provided global weights."""

		weights = global_weights.copy()
		for _ in range(self.local_epochs):
			preds = sigmoid(self.x @ weights)
			grad = (self.x.T @ (preds - self.y)) / len(self.y)
			weights -= self.learning_rate * grad
		return weights


@dataclasses.dataclass
class Server:
	"""Central server orchestrates aggregation and evaluation."""

	num_features: int
	global_weights: np.ndarray = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		self.global_weights = np.zeros(self.num_features)

	def aggregate(self, client_weights: Sequence[np.ndarray]) -> None:
		self.global_weights = np.mean(client_weights, axis=0)

	def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
		"""Return accuracy of the current global model."""

		probs = sigmoid(x @ self.global_weights)
		preds = (probs >= 0.5).astype(int)
		return float((preds == y).mean())


def make_synthetic_clients(
	num_clients: int,
	samples_per_client: int,
	num_features: int,
	rng: np.random.Generator,
) -> List[Client]:
	"""Create linearly separable toy data per client."""

	true_weights = rng.normal(size=num_features)
	clients: List[Client] = []
	for cid in range(num_clients):
		x = rng.normal(size=(samples_per_client, num_features))
		logits = x @ true_weights
		y = (sigmoid(logits) > rng.random(size=logits.shape)).astype(int)
		clients.append(Client(client_id=cid, x=x, y=y))
	return clients


def federated_training(
	server: Server,
	clients: Sequence[Client],
	num_rounds: int,
) -> Tuple[np.ndarray, List[float]]:
	"""Run FedAvg rounds and track global accuracy after each round."""

	history: List[float] = []
	all_x = np.vstack([c.x for c in clients])
	all_y = np.hstack([c.y for c in clients])

	for _ in range(num_rounds):
		client_weights = [client.train(server.global_weights) for client in clients]
		server.aggregate(client_weights)
		history.append(server.evaluate(all_x, all_y))
	return server.global_weights, history


def main() -> None:
	rng = np.random.default_rng(seed=7)

	num_clients = 5
	samples_per_client = 80
	num_features = 4
	num_rounds = 15

	clients = make_synthetic_clients(
		num_clients=num_clients,
		samples_per_client=samples_per_client,
		num_features=num_features,
		rng=rng,
	)

	server = Server(num_features=num_features)
	weights, accuracy_history = federated_training(
		server=server, clients=clients, num_rounds=num_rounds
	)

	print("Final global weights:\n", weights)
	for idx, acc in enumerate(accuracy_history, start=1):
		print(f"Round {idx:02d}: accuracy={acc:.3f}")


if __name__ == "__main__":
	main()
