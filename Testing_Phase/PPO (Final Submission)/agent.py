##agent.py for train_ppo.py
import torch
import torch.nn as nn
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class ActorCriticRNN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, n_actions=5):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, h):
        out, h_new = self.gru(x, h)
        probs = torch.softmax(self.actor(out), dim=-1)
        return probs, h_new

def reduce_state(obs):
    obs = obs.astype(int)
    reduced = np.zeros(9, dtype=np.float32)
    for i in range(7):
        reduced[i] = obs[i*2] | obs[i*2 + 1]
    reduced[7] = obs[16]
    reduced[8] = obs[17]
    return reduced

_model = None
_hidden_state = None

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _model, _hidden_state
    
    if _model is None:
        _model = ActorCriticRNN()
        path = os.path.join(os.path.dirname(__file__), "ppo_rnn_weights.pth")
        if os.path.exists(path):
            _model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        _model.eval()
        _hidden_state = torch.zeros(1, 1, 64) 
    
    reduced_obs = reduce_state(obs)
    
    with torch.no_grad():
        t_obs = torch.FloatTensor(reduced_obs).unsqueeze(0).unsqueeze(0)
        probs, _hidden_state = _model(t_obs, _hidden_state)
       
        action_idx = probs.argmax().item()
        
    return ACTIONS[action_idx]