import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import tensor
import torch
import numpy as np
import torch.optim.optimizer
import copy
import Helper

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
env = gym.make("CartPole-v1")
run_n = 0

class Actor(nn.Module):
    def __init__(self, lr=0.0001, input_size=4, output_size=2):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.softmax(self.layer4(x), dim=-1)
    
class Critic(nn.Module):
    def __init__(self, lr=0.0001, input_size=4):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
def run_ep(policy, env, deterministic=False):
    trace = []
    episode_over = False
    observation, _ = env.reset()
    tot_r = 0
    while not episode_over:
        old_obs = observation
        net_out = policy(tensor(observation, dtype=torch.float32).to(device))
        if deterministic:
            try:
                action = net_out.argmax().cpu().item()
            except ValueError:
                print(actor_net.layer1.weight.grad)
                print(observation)
                print(net_out)
                quit()
        else:
            try:
                action = torch.distributions.categorical.Categorical(net_out).sample().cpu().item()
            except ValueError:
                print(actor_net.layer1.weight.grad)
                print(observation)
                print(net_out)
                quit()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        tot_r += reward
        trace.append((old_obs, action, reward, episode_over))
    return trace, tot_r

    
def train(actor, critic, n_runs, episodes_per_descent=1, estimation_depth=5, gamma=0.99, report_freq=5000, max_steps=1000000):
    global run_n
    test_returns = []
    env = gym.make("CartPole-v1", render_mode=None)
    loss_fn = nn.MSELoss()
    tot_s = 0

    for n in range(n_runs):
        if n % episodes_per_descent == 0:
            if n != 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                actor.optimizer.step()
                critic.optimizer.step()
            actor.optimizer.zero_grad()
            critic.optimizer.zero_grad()
            
        with torch.no_grad():
            trace, _ = run_ep(actor, env)

        state_tensor = tensor(np.array([step[0] for step in trace]), device=device, requires_grad=True)
        actions_tensor = tensor([step[1] for step in trace], device=device)
        q_values = torch.zeros(len(trace), device=device)
        v_values = critic(state_tensor)

        for s in range(len(trace)):
            tot_s += 1
            if tot_s % report_freq == 0:
                with torch.no_grad():
                    _, trace_r = run_ep(actor, env)
                print(f'Step {tot_s}: test reward is {trace_r}')
                test_returns.append(trace_r)
            if tot_s >= max_steps:
                return test_returns
            
            r = 0
            for k in range(min(len(trace)-s, estimation_depth)):
                r += trace[s+k][2] * (gamma ** k)
            if len(trace)-s > estimation_depth:
                r += v_values[s+estimation_depth].cpu().item() * (gamma ** estimation_depth)
            q_values[s] = r

        advantage = q_values.unsqueeze(1) - v_values
        #advantage = (advantage - advantage.mean()) / advantage.std()
        
        likelihoods = actor(state_tensor).gather(1, actions_tensor.unsqueeze(1))
        log_likelihoods = torch.log(likelihoods)
        actor_loss = -(log_likelihoods * advantage.detach()).sum()

        critic_loss = torch.pow(advantage, 2).sum()

        actor_loss.backward()
        critic_loss.backward()
        run_n += 1

    return test_returns

if __name__ == "__main__":

    a2c_res = np.zeros((5, 200))
    for i in range(5):
        actor_net = Actor(lr=0.0007)
        actor_net.to(device)
        critic_net = Critic(lr=0.0007)
        critic_net.to(device)
        env = gym.make("CartPole-v1", render_mode=None)
        a2c_res[i] = train(actor_net, critic_net, 50000)
        with open('a2c_res.npy', 'wb') as f:
            np.save(f, a2c_res)

    plotter = Helper.LearningCurvePlot("A2C Learning Curve Graph", "Environment Step", "Episode Return")
    plotter.add_hline(500, "Optimal Performance")

    avg_return = np.mean(a2c_res, axis=0)
    std_dev = np.std(a2c_res, axis=0)

    smoothed_curve = Helper.smooth(avg_return, window=21)
    smoothed_std = Helper.smooth(std_dev, window=21)

    plotter.add_curve(np.linspace(5000, 1000000, 200), smoothed_curve, "A2C", "c")
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), smoothed_curve-std_dev, np.where(smoothed_curve+std_dev > 500, np.zeros(200)+500, smoothed_curve+std_dev), color="c", alpha=0.4)
    plotter.save("a2c_plot.png")
    
    

