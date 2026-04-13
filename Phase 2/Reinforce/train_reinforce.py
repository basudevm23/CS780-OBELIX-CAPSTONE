# import argparse, random, os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from tqdm import trange

# ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# class PolicyNetwork(nn.Module):
#     def __init__(self, in_dim=18, n_actions=5):
#         super().__init__()
#         self.common = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         # Policy head (Actor)
#         self.policy_head = nn.Linear(128, n_actions)
#         # Value head (Baseline/Critic)
#         self.value_head = nn.Linear(128, 1)

#     def forward(self, x):
#         x = self.common(x)
#         probs = torch.softmax(self.policy_head(x), dim=-1)
#         value = self.value_head(x)
#         return probs, value

# def import_obelix(obelix_py):
#     import importlib.util
#     spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     return mod.OBELIX

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--obelix_py", type=str, required=True)
#     parser.add_argument("--episodes", type=int, default=2000)
#     args = parser.parse_args()

#     OBELIX = import_obelix(args.obelix_py)
#     model = PolicyNetwork()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
#     GAMMA = 0.99
#     ENTROPY_BETA = 0.01 # Encourages exploration

#     for ep in trange(args.episodes):
#         env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1000000, wall_obstacles=True, difficulty=1, seed=ep)
#         s = env.reset(seed=ep)
        
#         saved_log_probs = []
#         saved_values = []
#         rewards = []
#         total_reward = 0

#         # 1. Generate an entire episode (Monte Carlo)
#         for _ in range(1000):
#             s_t = torch.FloatTensor(s).unsqueeze(0)
#             probs, value = model(s_t)
            
#             m = Categorical(probs)
#             action = m.sample()
            
#             s2, r, done = env.step(ACTIONS[action.item()], render=True)
            
#             saved_log_probs.append(m.log_prob(action))
#             saved_values.append(value)
#             rewards.append(r)
            
#             s = s2
#             total_reward += r
#             if done: break

#         # 2. Calculate Returns (G_t)
#         returns = []
#         R = 0
#         for r in reversed(rewards):
#             R = r + GAMMA * R
#             returns.insert(0, R)
        
#         returns = torch.FloatTensor(returns)
#         # Standardize returns for stability
#         returns = (returns - returns.mean()) / (returns.std() + 1e-9)

#         # 3. Policy and Value Updates
#         policy_loss = []
#         value_loss = []
        
#         for log_prob, val, R in zip(saved_log_probs, saved_values, returns):
#             advantage = R - val.item() # Advantage = Actual Return - Baseline
            
#             # Policy loss (Actor)
#             policy_loss.append(-log_prob * advantage)
            
#             # Value loss (Critic) - MSE between predicted value and actual return
#             value_loss.append(nn.functional.smooth_l1_loss(val, torch.tensor([[R]])))

#         optimizer.zero_grad()
#         # Total loss = Actor Loss + Critic Loss - Entropy Bonus
#         loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
#         loss.backward()
#         optimizer.step()

#         if (ep + 1) % 10 == 0:
#             print(f"Episode {ep+1:4} | Reward: {total_reward:8.1f}")

#     torch.save(model.state_dict(), "reinforce_weights.pth")

# if __name__ == "__main__":
#     main()
import argparse, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
import matplotlib.pyplot as plt # NEW: Added for plotting

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
        # Policy head (Actor)
        self.policy_head = nn.Linear(128, n_actions)
        # Value head (Baseline/Critic)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        probs = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return probs, value

def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    GAMMA = 0.99
    ENTROPY_BETA = 0.01 # Encourages exploration

    # NEW: List to track rewards across all episodes
    episode_rewards = []

    for ep in trange(args.episodes, desc="REINFORCE Training"):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1000000, wall_obstacles=True, difficulty=1, seed=ep)
        s = env.reset(seed=ep)
        
        saved_log_probs = []
        saved_values = []
        rewards = []
        total_reward = 0

        # 1. Generate an entire episode (Monte Carlo)
        for _ in range(1000):
            s_t = torch.FloatTensor(s).unsqueeze(0)
            probs, value = model(s_t)
            
            m = Categorical(probs)
            action = m.sample()
            
            # Switched render to False for faster training
            s2, r, done = env.step(ACTIONS[action.item()], render=False) 
            
            saved_log_probs.append(m.log_prob(action))
            saved_values.append(value)
            rewards.append(r)
            
            s = s2
            total_reward += r
            if done: break

        # 2. Calculate Returns (G_t)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        # Standardize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 3. Policy and Value Updates
        policy_loss = []
        value_loss = []
        
        for log_prob, val, R in zip(saved_log_probs, saved_values, returns):
            advantage = R - val.item() # Advantage = Actual Return - Baseline
            
            # Policy loss (Actor)
            policy_loss.append(-log_prob * advantage)
            
            # Value loss (Critic) - MSE between predicted value and actual return
            value_loss.append(nn.functional.smooth_l1_loss(val, torch.tensor([[R]])))

        optimizer.zero_grad()
        # Total loss = Actor Loss + Critic Loss - Entropy Bonus
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()

        # NEW: Log the total reward for plotting
        episode_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(f" Episode {ep+1:4} | Reward: {total_reward:8.1f}")

    # --- SAVE AND PLOT ---
    torch.save(model.state_dict(), "reinforce_weights.pth")
    np.save("reinforce_rewards.npy", np.array(episode_rewards))
    print("Training Complete. Saved to reinforce_weights.pth and reinforce_rewards.npy")

    # Generate the learning curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color="lightgreen", alpha=0.6)
    
    # Calculate a moving average (window of 20 episodes)
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(episode_rewards)), moving_avg, color="darkgreen", label=f"{window}-Ep Moving Average")
    
    plt.title("REINFORCE (Actor-Critic) Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.5) 
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    
    # Save and show the plot
    plt.savefig("reinforce_learning_curve.png", dpi=300, bbox_inches="tight")
    print("Saved plot to reinforce_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()