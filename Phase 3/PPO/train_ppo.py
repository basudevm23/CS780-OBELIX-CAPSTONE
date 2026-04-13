##train_ppo.py 
##Score: -1515
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
import matplotlib.pyplot as plt 

HP = {
    "EPISODES": 500,         
    "MAX_STEPS": 2000,       
    "LR": 1e-3,
    "GAMMA": 0.99,
    "EPS_CLIP": 0.2,
    "K_EPOCHS": 4,           
    "ENTROPY_COEF": 0.05,
    "STUCK_LIMIT": 50,       
    "MAX_TURNS": 8          
}

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def reduce_state(obs):
    obs = obs.astype(int)
    reduced = np.zeros(9, dtype=np.float32)
    for i in range(7):
        reduced[i] = obs[i*2] | obs[i*2 + 1]
    reduced[7] = obs[16]
    reduced[8] = obs[17] 
    return reduced

class ActorCriticRNN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, n_actions=5):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, h):
        out, h_new = self.gru(x, h)
        probs = torch.softmax(self.actor(out), dim=-1)
        value = self.critic(out)
        return probs, value, h_new

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

    torch.set_num_threads(4) 
    
    OBELIX = import_obelix(args.obelix_py)
    model = ActorCriticRNN()
    optimizer = optim.Adam(model.parameters(), lr=HP["LR"])


    episode_rewards = []

    for ep in trange(HP["EPISODES"], desc="PPO Training"):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=HP["MAX_STEPS"], 
                     wall_obstacles=True, difficulty=3, seed=ep)
        
        obs_raw = env.reset(seed=ep)
        s = reduce_state(obs_raw)
        h = torch.zeros(1, 1, 64) 
        
        states, actions, rewards, log_probs = [], [], [], []
        total_reward = 0
   
        stuck_count = 0
        consecutive_turns = 0 

        for step in range(HP["MAX_STEPS"]):
            with torch.no_grad():
                s_t = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
                probs, value, h_new = model(s_t, h)
                m = Categorical(probs.squeeze())
                a = m.sample()
                log_prob = m.log_prob(a)
                
            s2_raw, r, done = env.step(ACTIONS[a.item()], render=False) 
            s2 = reduce_state(s2_raw)
            
  
            if a.item() == 2: 
              
                consecutive_turns = 0
            else:
            
                consecutive_turns += 1
                
            if consecutive_turns >= HP["MAX_TURNS"]:
                r -= 800 
                consecutive_turns = 0 
                
       
            if s2[8] == 1: 
                stuck_count += 1
            else:
                stuck_count = 0
                
            if stuck_count >= HP["STUCK_LIMIT"]:
                r -= 1000 
                done = True 
                
            states.append(s)
            actions.append(a.item())
            rewards.append(r)
            log_probs.append(log_prob.item())
            
            total_reward += r
            s = s2
            h = h_new
            if done: break
            
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + HP["GAMMA"] * R
            returns.insert(0, R)
            
        returns_t = torch.FloatTensor(returns)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)
            
        states_t = torch.FloatTensor(np.array(states)).unsqueeze(1) 
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(log_probs)

        h_0 = torch.zeros(1, len(states_t), 64) 
        
        for _ in range(HP["K_EPOCHS"]):
            probs, values, _ = model(states_t, h_0)
            values = values.squeeze()
            
            m = Categorical(probs.squeeze())
            new_log_probs = m.log_prob(actions_t)
            entropy = m.entropy()
            
            ratios = torch.exp(new_log_probs - old_log_probs_t)
            advantages = returns_t - values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - HP["EPS_CLIP"], 1 + HP["EPS_CLIP"]) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns_t)
            
            loss = actor_loss + 0.5 * critic_loss - HP["ENTROPY_COEF"] * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        episode_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(f" Ep {ep+1:4} | Steps Survived: {step:4} | Reward: {total_reward:8.1f}")

    torch.save(model.state_dict(), "ppo_rnn_weights.pth")
    np.save("ppo_rewards.npy", np.array(episode_rewards))
    print("Training Complete. Saved to ppo_rnn_weights.pth and ppo_rewards.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color="lightblue", alpha=0.6)

    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
  
        plt.plot(np.arange(window-1, len(episode_rewards)), moving_avg, color="blue", label=f"{window}-Ep Moving Average")
    
    plt.title("PPO Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.savefig("ppo_learning_curve.png", dpi=300, bbox_inches="tight")
    print("Saved plot to ppo_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()