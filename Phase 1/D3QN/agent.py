import torch
import torch.nn as nn
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, n_actions)

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

_model = None

def _load():
    global _model
    if _model is not None: return
    _model = DuelingDQN()
    path = os.path.join(os.path.dirname(__file__), "weights.pth")
    # map_location='cpu' is vital for Codabench compatibility
    if os.path.exists(path):
        _model.load_state_dict(torch.load(path, map_location="cpu"))
    _model.eval()

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    with torch.no_grad():
        t_obs = torch.FloatTensor(obs).unsqueeze(0)
        action_idx = _model(t_obs).argmax().item()
    return ACTIONS[action_idx]