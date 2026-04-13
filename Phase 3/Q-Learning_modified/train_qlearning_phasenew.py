##train_qlearning_phasenew.py 
##Score: -5589
import argparse, random, os
import numpy as np
from tqdm import trange

HP = {
    "EPISODES": 600,
    "MAX_STEPS": 5000,
    "ALPHA": 0.1,         
    "GAMMA": 0.9,        
    "LAMBDA": 0.8,      
    "EPS_START": 1.0,
    "EPS_MIN": 0.05,
    "EPS_DECAY": 0.999,
    "WALL_OBSTACLES": True,
    "DIFFICULTY": 3,  
    "HAMMING_THRESHOLD": 2 
}

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class Phase3Agent:
    def __init__(self):
        self.q_table = {} 
        self.e_trace = {}
        self.weights = np.ones(18)
        self.weights[0:16:2] = 2  
        self.weights[16] = 5  
        self.weights[17] = 5   

    def get_state(self, obs):
        return tuple(obs.astype(int))

    def weighted_hamming(self, s1, s2):
        
        s1, s2 = np.array(s1), np.array(s2)
        diff = (s1 != s2).astype(int)
        return np.sum(diff * self.weights)

    def get_q(self, s):
        if s in self.q_table:
            return self.q_table[s]
        similar_values = [self.q_table[st] for st in self.q_table 
                          if self.weighted_hamming(s, st) <= HP["HAMMING_THRESHOLD"]]
        
        if similar_values:
            self.q_table[s] = np.mean(similar_values, axis=0)
        else:
            self.q_table[s] = np.zeros(len(ACTIONS))
        return self.q_table[s]

    def update(self, s, a, r, s2, a2, done):
        q_s = self.get_q(s)
        q_s2 = self.get_q(s2)

        td_target = r
        if not done:
            td_target += HP["GAMMA"] * q_s2[a2]
        td_error = td_target - q_s[a]

        if s not in self.e_trace:
            self.e_trace[s] = np.zeros(len(ACTIONS))
        self.e_trace[s][a] += 1

  
        states_to_prune = []
        for sk in list(self.e_trace.keys()):

            self.get_q(sk) 
            self.q_table[sk] += HP["ALPHA"] * td_error * self.e_trace[sk]

            self.e_trace[sk] *= HP["GAMMA"] * HP["LAMBDA"]

            if np.max(self.e_trace[sk]) < 0.005:
                states_to_prune.append(sk)
        
        for sk in states_to_prune:
            del self.e_trace[sk]

    def choose_action(self, s, epsilon):

        if random.random() < epsilon:
            return random.randint(0, 4)
        return np.argmax(self.get_q(s))

def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent = Phase3Agent()
    epsilon = HP["EPS_START"]

    for ep in trange(HP["EPISODES"], desc="Phase 3 Training"):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=HP["MAX_STEPS"], 
                     wall_obstacles=HP["WALL_OBSTACLES"], difficulty=HP["DIFFICULTY"], seed=ep)
        s = agent.get_state(env.reset(seed=ep))
        a = agent.choose_action(s, epsilon)
        agent.e_trace = {}
        total_reward = 0

        for _ in range(HP["MAX_STEPS"]):
            s2_raw, r, done = env.step(ACTIONS[a], render=False)
            s2 = agent.get_state(s2_raw)
            a2 = agent.choose_action(s2, epsilon)
            
            agent.update(s, a, r, s2, a2, done)
            
            total_reward += r
            s, a = s2, a2
            if done: break
        
        epsilon = max(HP["EPS_MIN"], epsilon * HP["EPS_DECAY"])
        if (ep + 1) % 10 == 0:
            print(f" Ep {ep+1:4} | Reward: {total_reward:8.1f} | States: {len(agent.q_table)}")

    np.save("phase3_weights.npy", agent.q_table)

if __name__ == "__main__":
    main()