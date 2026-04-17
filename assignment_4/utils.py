import torch
import numpy as np

def get_model_weights(model):
    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.cpu().numpy().tolist()  # convert ndarray → list
    return weights

def set_model_weights(model, weights):
    state_dict = {}
    for k, v in weights.items():
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)

def federated_average(weight_list):
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = np.mean([np.array(w[key]) for w in weight_list], axis=0).tolist()
    return avg_weights
