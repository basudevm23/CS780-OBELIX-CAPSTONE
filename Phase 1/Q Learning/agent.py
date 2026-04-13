import numpy as np
import os
from typing import List

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

_q_table = None

def _load_weights():
    """
    Loads the dictionary-based Q-table from the .npy file.
    This runs only once during the first call to policy().
    """
    global _q_table
    if _q_table is not None:
        return

    here = os.path.dirname(__file__)
    path = os.path.join(here, "q_lambda_table.npy")
    
    if os.path.exists(path):
        _q_table = np.load(path, allow_pickle=True).item()
    else:
        _q_table = {}

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Mapping sensor observations to discrete actions
    """
    _load_weights()

    state = tuple(obs.astype(int))

    if _q_table and state in _q_table:
        action_idx = np.argmax(_q_table[state])
        return ACTIONS[action_idx] 
    probs = np.array([0.05, 0.1, 0.7, 0.1, 0.05])
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]