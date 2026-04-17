import torch
import torch.nn as nn
import torch.optim as optim
import requests
import numpy as np
from model import SimpleNN
from utils import get_model_weights, set_model_weights

SERVER_URL = "http://127.0.0.1:5000"

# Dummy local dataset
def get_local_data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_local_model():
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X, y = get_local_data()

    for epoch in range(3):  # local epochs
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

def send_update(model):
    weights = get_model_weights(model)
    response = requests.post(f"{SERVER_URL}/upload", json=weights)
    print("Update sent:", response.json())

if __name__ == "__main__":
    local_model = train_local_model()
    send_update(local_model)
