"""
Federated Learning – Model
===========================
Simple feedforward neural network for binary classification.
Used by both server (global model) and clients (local models).
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    A 3-layer fully-connected network.
    Input : 10 features
    Output: 2 classes (binary classification)
    """
    def __init__(self, input_dim=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
