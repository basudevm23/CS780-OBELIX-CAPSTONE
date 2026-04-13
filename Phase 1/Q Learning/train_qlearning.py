##train_qlearning.py
##Score: -1990
import argparse
import random
import numpy as np
import pickle
from tqdm import trange
import os
import matplotlib.pyplot as plt 

HP = {
    "EPISODES": 2000,
    "MAX_STEPS": 1500,
    "ALPHA": 0.1,         
    "GAMMA": 0.99,        
    "LAMBDA": 0.9,        
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.996,    
    "REPLACE_TRACE": True,
    "WALL_OBSTACLES": True,
    "TRACE_THRESHOLD": 0.005
}

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class QLambdaAgent:
    def __init__(self):
        self.q_table = {} 
        self.e_trace = {}  
        
    def get_state(self, obs):
        return tuple(obs.astype(int))

    def get_q(self, s):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(ACTIONS))
        return self.q_table[s]

    def update(self, s, a, r, s_prime, a_prime, done):
        q_s_prime = self.get_q(s_prime)
        is_greedy = (q_s_prime[a_prime] == np.max(q_s_prime))
        
        td_target = r
        if not done:
            td_target += HP["GAMMA"] * np.max(q_s_prime)
        td_error = td_target - self.get_q(s)[a]
        
        if HP["REPLACE_TRACE"]:
            self.e_trace[s] = np.zeros(len(ACTIONS))
            
        if s not in self.e_trace:
            self.e_trace[s] = np.zeros(len(ACTIONS))
        self.e_trace[s][a] += 1

        states_to_prune = []
        for state_key, e_vals in self.e_trace.items():
            self.get_q(state_key) 
            self.q_table[state_key] += HP["ALPHA"] * td_error * e_vals
        
            if is_greedy:
                self.e_trace[state_key] *= HP["GAMMA"] * HP["LAMBDA"]

                if np.max(self.e_trace[state_key]) < HP["TRACE_THRESHOLD"]:
                    states_to_prune.append(state_key)
            else:
                states_to_prune.append(state_key)
        for sk in states_to_prune:
            del self.e_trace[sk]

    def choose_action(self, s, epsilon):
        if random.random() < epsilon:
            return random.randint(0, len(ACTIONS) - 1)
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
    agent = QLambdaAgent()
    epsilon = HP["EPS_START"]
    
    episode_rewards = []

    for ep in trange(HP["EPISODES"]):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=HP["MAX_STEPS"], 
                     wall_obstacles=HP["WALL_OBSTACLES"], difficulty=0, seed=ep)
        s_raw = env.reset(seed=ep)
        s = agent.get_state(s_raw)
        a = agent.choose_action(s, epsilon)
        agent.e_trace = {} 
        total_reward = 0
        step_count = 0

        for _ in range(HP["MAX_STEPS"]):
            s_prime_raw, r, done = env.step(ACTIONS[a], render=False)
            s_prime = agent.get_state(s_prime_raw)
            a_prime = agent.choose_action(s_prime, epsilon)
            
            agent.update(s, a, r, s_prime, a_prime, done)
            
            total_reward += r
            step_count += 1
            s, a = s_prime, a_prime
            if done: break
        
        epsilon = max(HP["EPS_MIN"], epsilon * HP["EPS_DECAY"])
        episode_rewards.append(total_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:4} | Reward: {total_reward:8.1f} | Eps: {epsilon:.4f} | States: {len(agent.q_table)}")


    with open("q_lambda_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    np.save("q_lambda_rewards.npy", np.array(episode_rewards))
    print("\nTraining Finished. Weights saved to q_lambda_table.pkl and rewards to q_lambda_rewards.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color="lightcoral", alpha=0.6)

    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(episode_rewards)), moving_avg, color="darkred", label=f"{window}-Ep Moving Average")
    
    plt.title("Q(λ) Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.savefig("q_lambda_learning_curve.png", dpi=300, bbox_inches="tight")
    print("Saved plot to q_lambda_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()