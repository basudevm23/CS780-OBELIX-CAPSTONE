##train_d3qn
import argparse, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

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
    parser.add_argument("--out", type=str, default="weights.pth")
    args = parser.parse_args()

    torch.set_num_threads(4)
    
    OBELIX = import_obelix(args.obelix_py)
    q_net = DuelingDQN()
    target_net = DuelingDQN()
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = deque(maxlen=100000)
    
    epsilon = 1.0
    steps = 0
    UPDATE_FREQ = 4 
    pbar = tqdm(total=args.episodes, desc="Training Progress")

    for ep in range(args.episodes):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1500, wall_obstacles=True, difficulty=0, seed=ep)
        s = env.reset(seed=ep)
        total_reward = 0

        for _ in range(1500):
            if random.random() < epsilon:
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax().item()

            s2, r, done = env.step(ACTIONS[a], render=False)
            
            replay.append((s, a, r, s2, done))
            
            if len(replay) > 2000 and steps % UPDATE_FREQ == 0:
                batch = random.sample(replay, 128)
                sb, ab, rb, s2b, db = zip(*batch)
                
                sb_t = torch.FloatTensor(np.array(sb))
                ab_t = torch.LongTensor(ab)
                rb_t = torch.FloatTensor(rb)
                s2b_t = torch.FloatTensor(np.array(s2b))
                db_t = torch.FloatTensor(db)

                with torch.no_grad():
                    best_a = q_net(s2b_t).argmax(1).unsqueeze(1)
                    max_next_q = target_net(s2b_t).gather(1, best_a).squeeze()
                    target = rb_t + (0.99 * max_next_q * (1 - db_t))

                curr_q = q_net(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze()
                loss = nn.MSELoss()(curr_q, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if steps % 2000 == 0:
                    target_net.load_state_dict(q_net.state_dict())

            s = s2
            total_reward += r
            steps += 1
            if done: break
            
        epsilon = max(0.01, epsilon * 0.997)
        pbar.update(1)    
        if (ep + 1) % 10 == 0:
            tqdm.write(f"Episode {ep+1:4} | Reward: {total_reward:8.1f} | Eps: {epsilon:.4f} | Buffer: {len(replay)}")
    pbar.close()
    torch.save(q_net.state_dict(), args.out)
    print(f"Saved weights to {args.out}")

if __name__ == "__main__":
    main()