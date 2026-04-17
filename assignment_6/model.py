"""
Vertical Federated Learning – Models
======================================
Two-part split neural network for vertical federated learning:

  • BottomModel – local feature extractor held by each party
  • TopModel   – aggregation model held by the server/coordinator

In VFL every party sees a **different subset of features** for the
same set of samples.  Each party's BottomModel converts its local
features into a fixed-size embedding.  The server's TopModel takes
the concatenated embeddings from all parties and produces the final
classification output.
"""

import torch
import torch.nn as nn


class BottomModel(nn.Module):
    """
    Local feature extractor held by one party.

    Parameters
    ----------
    input_dim  : int – number of features this party owns
    output_dim : int – embedding size sent to the server
    """
    def __init__(self, input_dim, output_dim=32):
        super(BottomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class TopModel(nn.Module):
    """
    Aggregation / classification model held by the server.

    Parameters
    ----------
    input_dim   : int – total embedding size (sum of all parties' output_dim)
    num_classes : int – number of output classes
    """
    def __init__(self, input_dim, num_classes=2):
        super(TopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
