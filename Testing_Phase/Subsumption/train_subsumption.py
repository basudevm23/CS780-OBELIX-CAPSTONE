##Testing Phase
##Score: -4286
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange


HP = {
    "EPISODES": 3001,       
    "MAX_STEPS": 2000,       
    "LR": 1e-3,
    "GAMMA": 0.99,
    "EPS_CLIP": 0.2,
    "K_EPOCHS": 4,           
    "ENTROPY_COEF": 0.15,
    "PERSISTENCE": 5,       
    "STUCK_LIMIT": 50    
}

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ROTATIONS = {0: 45, 1: 22.5, 2: 0, 3: -22.5, 4: -45}

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

def update_ppo(model, optimizer, states, actions, rewards, log_probs):
    if len(states) == 0: return 
    
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

    # Prevent thread lockup on Windows
    torch.set_num_threads(4) 
    OBELIX = import_obelix(args.obelix_py)
    
    # Init 3 Separate Brains
    finder = ActorCriticRNN()
    pusher = ActorCriticRNN()
    unwedger = ActorCriticRNN()
    
    opt_f = optim.Adam(finder.parameters(), lr=HP["LR"])
    opt_p = optim.Adam(pusher.parameters(), lr=HP["LR"])
    opt_u = optim.Adam(unwedger.parameters(), lr=HP["LR"])

    # Tracker for the learning curve
    learning_curve_rewards = []

    for ep in trange(HP["EPISODES"], desc="Subsumption Training"):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=HP["MAX_STEPS"], 
                     wall_obstacles=True, difficulty=3, seed=ep)
        
        s = reduce_state(env.reset(seed=ep))
        
        h_f, h_p, h_u = [torch.zeros(1, 1, 64) for _ in range(3)]
        
        buf_f = {"s":[], "a":[], "r":[], "lp":[]}
        buf_p = {"s":[], "a":[], "r":[], "lp":[]}
        buf_u = {"s":[], "a":[], "r":[], "lp":[]}
        
        total_global_reward = 0
        unwedge_timer, push_timer, stuck_count, angle_sum = 0, 0, 0, 0

        for step in range(HP["MAX_STEPS"]):
            is_stuck = (s[8] == 1)
            is_bump = (s[7] == 1)
            is_near_box = np.any(s[2:5] == 1) 
            
            # THE SWITCHER (Priority Network)
            if is_stuck: unwedge_timer = HP["PERSISTENCE"]
            if is_bump: push_timer = HP["PERSISTENCE"]
            
            active_module = "FINDER"
            if unwedge_timer > 0:
                active_module = "UNWEDGER"
            elif push_timer > 0:
                active_module = "PUSHER"

            # ACTION SELECTION
            with torch.no_grad():
                s_t = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
                if active_module == "UNWEDGER":
                    probs, val, h_u_new = unwedger(s_t, h_u)
                elif active_module == "PUSHER":
                    probs, val, h_p_new = pusher(s_t, h_p)
                else:
                    probs, val, h_f_new = finder(s_t, h_f)
                    
                m = Categorical(probs.squeeze())
                a = m.sample()
                log_prob = m.log_prob(a)
                
            s2_raw, global_r, done = env.step(ACTIONS[a.item()], render=False)
            s2 = reduce_state(s2_raw)
            total_global_reward += global_r
            
            is_stuck_next = (s2[8] == 1)
            is_bump_next = (s2[7] == 1)
            is_near_next = np.any(s2[2:5] == 1)


            shaping = 0.0
            if a.item() == 2: 
                angle_sum = 0
            else:
                angle_sum += ROTATIONS[a.item()]
           
                if active_module != "UNWEDGER":
                    shaping -= 0.5 
     
                if abs(angle_sum) >= 360:
                    shaping -= 100.0 
                    angle_sum = 0

            if is_stuck_next:
                stuck_count += 1
            else:
                stuck_count = 0
                
            if stuck_count >= HP["STUCK_LIMIT"]:
                shaping -= 500 
                done = True

            if active_module == "UNWEDGER":
                custom_r = 1.0 if (not is_stuck_next and a.item() == 2) else -3.0
                custom_r += shaping
                buf_u["s"].append(s); buf_u["a"].append(a.item()); buf_u["r"].append(custom_r); buf_u["lp"].append(log_prob.item())
                h_u = h_u_new
                unwedge_timer -= 1
                
            elif active_module == "PUSHER":
                custom_r = 1.0 if (is_bump_next and a.item() == 2) else -3.0
                custom_r += shaping
                buf_p["s"].append(s); buf_p["a"].append(a.item()); buf_p["r"].append(custom_r); buf_p["lp"].append(log_prob.item())
                h_p = h_p_new
                push_timer -= 1
                
            else:
                custom_r = 3.0 if is_near_next else -1.0
                custom_r += shaping
                buf_f["s"].append(s); buf_f["a"].append(a.item()); buf_f["r"].append(custom_r); buf_f["lp"].append(log_prob.item())
                h_f = h_f_new
                
            s = s2
            if done: break
            
        update_ppo(finder, opt_f, buf_f["s"], buf_f["a"], buf_f["r"], buf_f["lp"])
        update_ppo(pusher, opt_p, buf_p["s"], buf_p["a"], buf_p["r"], buf_p["lp"])
        update_ppo(unwedger, opt_u, buf_u["s"], buf_u["a"], buf_u["r"], buf_u["lp"])
        
        learning_curve_rewards.append(total_global_reward)

        if (ep + 1) % 10 == 0:
            print(f" Ep {ep+1:4} | Env Score: {total_global_reward:8.1f} | Active Timers -> U:{unwedge_timer} P:{push_timer}")

    torch.save(finder.state_dict(), "finder_rnn.pth")
    torch.save(pusher.state_dict(), "pusher_rnn.pth")
    torch.save(unwedger.state_dict(), "unwedger_rnn.pth")
    np.save("subsumption_rewards.npy", np.array(learning_curve_rewards))
    print("Training Complete. Saved weights and subsumption_rewards.npy")

if __name__ == "__main__":
    main()