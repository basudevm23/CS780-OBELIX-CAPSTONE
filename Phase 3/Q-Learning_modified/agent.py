##agent.py for train_qlearning_phasenew.py
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_q_table = None

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _q_table
    if _q_table is None:
        path = os.path.join(os.path.dirname(__file__), "phase3_weights.npy")
        if os.path.exists(path):
            _q_table = np.load(path, allow_pickle=True).item()
        else:
            _q_table = {}
    
    s = tuple(obs.astype(int))

    if s in _q_table:
        return ACTIONS[np.argmax(_q_table[s])]
    
 
    return "FW"
