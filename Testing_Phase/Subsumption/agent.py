#agent.py for subsumption.py
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

_finder, _pusher, _unwedger = None, None, None
_h_f, _h_p, _h_u = None, None, None
_unwedge_timer = 0
_push_timer = 0

def _load_model(filename):
    model = ActorCriticRNN()
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    else:
        print(f"Warning: Could not find {filename}")
    model.eval()
    return model

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _finder, _pusher, _unwedger
    global _h_f, _h_p, _h_u
    global _unwedge_timer, _push_timer
    
    if _finder is None:
        _finder = _load_model("finder_rnn.pth")
        _pusher = _load_model("pusher_rnn.pth")
        _unwedger = _load_model("unwedger_rnn.pth")
        
        _h_f = torch.zeros(1, 1, 64)
        _h_p = torch.zeros(1, 1, 64)
        _h_u = torch.zeros(1, 1, 64)
        _unwedge_timer, _push_timer = 0, 0
    
    s = reduce_state(obs)
    is_stuck = (s[8] == 1)
    is_bump = (s[7] == 1)
    
    if is_stuck: _unwedge_timer = 5
    if is_bump: _push_timer = 5
    
    with torch.no_grad():
        t_obs = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)

        if _unwedge_timer > 0:
            probs, _h_u = _unwedger(t_obs, _h_u)
            _unwedge_timer -= 1
        elif _push_timer > 0:
            probs, _h_p = _pusher(t_obs, _h_p)
            _push_timer -= 1
        else:
            probs, _h_f = _finder(t_obs, _h_f)
        action_idx = probs.argmax().item()
        
    return ACTIONS[action_idx]