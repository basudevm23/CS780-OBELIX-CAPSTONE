##agent.py for train_reinforce.py
import torch
import torch.nn as nn
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        probs = torch.softmax(self.policy_head(x), dim=-1)
        return probs

_model = None

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _model
    if _model is None:
        _model = PolicyNetwork()
        path = os.path.join(os.path.dirname(__file__), "reinforce_weights.pth")
        if os.path.exists(path):
            _model.load_state_dict(torch.load(path, map_location="cpu"))
        _model.eval()
    
    with torch.no_grad():
        t_obs = torch.FloatTensor(obs).unsqueeze(0)
        probs = _model(t_obs)
        action_idx = probs.argmax().item()
    
    return ACTIONS[action_idx]