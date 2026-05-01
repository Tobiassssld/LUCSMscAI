import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import tensor
import torch
import random
import numpy as np
import torch.optim.optimizer
import Helper
import matplotlib.pyplot as plt
import copy
import sys

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
env = gym.make("CartPole-v1", render_mode="human")
batch_size = 128
max_buffer_size = 10000
run_n = 0


class DQN(nn.Module):
    def __init__(self, nb_h=2, lr=0.0001):
        super(DQN, self).__init__()
        self.nb_h = nb_h
        self.layers = nn.ModuleList([nn.Linear(4, 128)])
        self.layers.extend([nn.Linear(128, 128) for n in range(2, nb_h+2)])
        self.layers.append(nn.Linear(128, 2))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.xp_buffer = []
        self.xp_count = 0
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def train(network, n_runs, gamma, epsilon, steps_per_update=1, replay_buffer=False, target_net=False, target_update_freq=200, eps_decay=0.999, min_eps=0.05, return_rec=5000):
    global run_n
    returns = np.zeros(1000000 // return_rec)
    env = gym.make("CartPole-v1", render_mode=None)
    s = 0
    tot_r_eps = [0 for x in range(50)]
    target_network = copy.deepcopy(network)
    network.optimizer.zero_grad()
    loss_fn = nn.MSELoss()
    for n in range(n_runs):
        if s >= 1000000:
            print("step count over 1M: returning")
            return returns
        episode_over = False
        ep = []
        observation, _ = env.reset()
        tot_r = 0
        while not episode_over:
            s += 1
            old_obs = observation
            if random.random() > epsilon:
                net_out = network(tensor(observation, dtype=torch.float32, requires_grad=(not target_net)).to(device))
                action = net_out.argmax().cpu().item()
                out = net_out.gather(-1, tensor([action], device=device))
            else:
                action = env.action_space.sample()
                out = network(tensor(observation, dtype=torch.float32, requires_grad=(not target_net)).to(device)).gather(-1, tensor([action], device=device))
            observation, reward, terminated, truncated, info = env.step(action)
            tot_r += reward
            with torch.no_grad():
                if target_net:
                    net_out = target_network(tensor(observation, dtype=torch.float32).to(device))
                else:
                    net_out = network(tensor(observation, dtype=torch.float32).to(device))
            
            episode_over = terminated or truncated
            target = reward + (gamma * net_out.max().cpu().item() * int(not episode_over))
            
            if not replay_buffer:
                loss = loss_fn(out, tensor([target], dtype=torch.float32).to(device))
                loss.backward()
            
            if replay_buffer:
                    if network.xp_count < max_buffer_size:
                        network.xp_buffer.append((old_obs, action, reward, observation, episode_over))
                        network.xp_count += 1
                    else:
                        network.xp_buffer[random.randint(0,max_buffer_size-1)] = (old_obs, action, reward, observation, episode_over)
            if s % steps_per_update == 0:
                if replay_buffer and len(network.xp_buffer) >= batch_size:
                    states = torch.zeros((batch_size, 4), device=device)
                    actions = torch.zeros(batch_size, device=device, dtype=torch.int64)
                    rewards = torch.zeros(batch_size)
                    next_state = torch.zeros((batch_size, 4), device=device)
                    done = torch.zeros(batch_size, device=device, dtype=torch.bool)
                    for x, xp in enumerate(random.sample(network.xp_buffer, batch_size)):
                        states[x] = tensor(xp[0])
                        actions[x] = xp[1]
                        rewards[x] = xp[2]
                        next_state[x] = tensor(xp[3])
                        done[x] = xp[4]
                    states.requires_grad = True
                    out = network(states).gather(1, actions.unsqueeze(1))
                    with torch.no_grad():
                        if target_net:
                            net_out = target_network(next_state).max(1).values
                        else:
                            net_out = network(next_state).max(1).values
                        for i in range(batch_size):
                            if done[i] == True:
                                net_out[i] = rewards[i]
                            else:
                                net_out[i] = rewards[i] + (gamma*net_out[i])
                    loss = loss_fn(out, net_out.unsqueeze(1))
                    loss.backward()
                network.optimizer.step()
                network.optimizer.zero_grad()

            if s % target_update_freq == 0:
                target_network = copy.deepcopy(network)
            if s % return_rec == 0:
                returns[(s // return_rec)-1] = test(network, 1)
                print(f'run {run_n}, episode {n}, step {s}: {returns[(s // return_rec)-1]}')
        tot_r_eps[n%50] = tot_r
        if epsilon > min_eps:
            epsilon *= eps_decay

def test(network, n_runs, debug=False):
    env = gym.make("CartPole-v1", render_mode=None)
    tot_r = 0
    for n in range(n_runs):
        if n == n_runs-1 and debug:
            env = gym.make("CartPole-v1", render_mode="human")
        observation, _ = env.reset()
        episode_over = False
        while not episode_over:
            with torch.no_grad():
                if n == n_runs-1 and debug:
                    env.render()
                net_out = network(tensor(observation, dtype=torch.float32).to(device))
                action = net_out.argmax().cpu().item()
                observation, reward, terminated, truncated, info = env.step(action)
                tot_r += reward
                episode_over = terminated or truncated
    return tot_r/n_runs

def naive_DQN():
    global run_n
    returns = np.zeros((5, 200))
    for i in range(5):
        run_n += 1
        net = DQN()
        net.to(device)
        returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('naive.npy', 'wb') as f:
            np.save(f, returns)

    env.close()
    avg_return = np.average(returns, axis=0)
    std_dev = np.std(returns, axis=0)
    plotter = Helper.LearningCurvePlot("Naive DQN Learning Curve", "Environment Step", "Episode Return")
    plotter.add_hline(500, "Optimal Performance")
    curve = Helper.smooth(avg_return, window=21)
    plotter.add_curve(np.linspace(5000, 1000000, 200), curve, "DQN", "orange")
    std_smooth = Helper.smooth(std_dev, window=21)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), curve-std_smooth, np.where(curve+std_smooth > 500, np.zeros(200)+500, curve+std_smooth), color="orange", alpha=0.4)
    plotter.save("naive_dqn.png")

def ablation_study():
    global run_n
    low_nbh_returns = np.zeros((5, 200))
    med_nbh_returns = np.zeros((5, 200))
    high_nbh_returns = np.zeros((5, 200))

    low_lr_returns = np.zeros((5, 200))
    med_lr_returns = np.zeros((5, 200))
    high_lr_returns = np.zeros((5, 200))

    low_eps_returns = np.zeros((5, 200))
    med_eps_returns = np.zeros((5, 200))
    high_eps_returns = np.zeros((5, 200))

    low_s2u_returns = np.zeros((5, 200))
    med_s2u_returns = np.zeros((5, 200))
    high_s2u_returns = np.zeros((5, 200))

    for i in range(5):
        run_n += 1
        net = DQN(nb_h=1)
        net.to(device)
        low_nbh_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('low_nbh.npy', 'wb') as f:
            np.save(f, low_nbh_returns)
        
        net = DQN(nb_h=3)
        net.to(device)
        med_nbh_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('med_nbh.npy', 'wb') as f:
            np.save(f, med_nbh_returns)

        net = DQN(nb_h=5)
        net.to(device)
        high_nbh_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('high_nbh.npy', 'wb') as f:
            np.save(f, high_nbh_returns)

        net = DQN(lr=0.000005)
        net.to(device)
        low_lr_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('low_lr.npy', 'wb') as f:
            np.save(f, low_lr_returns)
        
        net = DQN(lr=0.00005)
        net.to(device)
        med_lr_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('med_lr.npy', 'wb') as f:
            np.save(f, med_lr_returns)

        net = DQN(lr=0.0005)
        net.to(device)
        high_lr_returns[i] = train(net, 50000, 0.99, 0.9, replay_buffer=False, target_net=False)
        with open('high_lr.npy', 'wb') as f:
            np.save(f, high_lr_returns)

    env.close()

    low_nbh_avg_return = np.average(low_nbh_returns, axis=0)
    med_nbh_avg_return = np.average(med_nbh_returns, axis=0)
    high_nbh_avg_return = np.average(high_nbh_returns, axis=0)

    low_lr_avg_return = np.average(low_lr_returns, axis=0)
    med_lr_avg_return = np.average(med_lr_returns, axis=0)
    high_lr_avg_return = np.average(high_lr_returns, axis=0)

    low_eps_avg_return = np.average(low_eps_returns, axis=0)
    med_eps_avg_return = np.average(med_eps_returns, axis=0)
    high_eps_avg_return = np.average(high_eps_returns, axis=0)

    low_s2u_avg_return = np.average(low_s2u_returns, axis=0)
    med_s2u_avg_return = np.average(med_s2u_returns, axis=0)
    high_s2u_avg_return = np.average(high_s2u_returns, axis=0)

    low_nbh_std = np.std(low_nbh_returns, axis=0)
    med_nbh_std = np.std(med_nbh_returns, axis=0)
    high_nbh_std = np.std(high_nbh_returns, axis=0)

    low_lr_std = np.std(low_lr_returns, axis=0)
    med_lr_std = np.std(med_lr_returns, axis=0)
    high_lr_std = np.std(high_lr_returns, axis=0)

    low_eps_std = np.std(low_eps_returns, axis=0)
    med_eps_std = np.std(med_eps_returns, axis=0)
    high_eps_std = np.std(high_eps_returns, axis=0)

    low_s2u_std = np.std(low_s2u_returns, axis=0)
    med_s2u_std = np.std(med_s2u_returns, axis=0)
    high_s2u_std = np.std(high_s2u_returns, axis=0)

    nbh_plotter = Helper.LearningCurvePlot("Number of Hidden Layers Learning Curves", "Environment Step", "Episode Return")
    nbh_plotter.add_hline(500, "Optimal Performance")

    lr_plotter = Helper.LearningCurvePlot("Learning Rate Learning Curves", "Environment Step", "Episode Return")
    lr_plotter.add_hline(500, "Optimal Performance")

    eps_plotter = Helper.LearningCurvePlot("Epsilon Decay Learning Curves", "Environment Step", "Episode Return")
    eps_plotter.add_hline(500, "Optimal Performance")

    s2u_plotter = Helper.LearningCurvePlot("Steps per Update Learning Curves", "Environment Step", "Episode Return")
    s2u_plotter.add_hline(500, "Optimal Performance")

    low_nbh_curve = Helper.smooth(low_nbh_avg_return, window=21)
    med_nbh_curve = Helper.smooth(med_nbh_avg_return, window=21)
    high_nbh_curve = Helper.smooth(high_nbh_avg_return, window=21)

    low_lr_curve = Helper.smooth(low_lr_avg_return, window=21)
    med_lr_curve = Helper.smooth(med_lr_avg_return, window=21)
    high_lr_curve = Helper.smooth(high_lr_avg_return, window=21)

    low_eps_curve = Helper.smooth(low_eps_avg_return, window=21)
    med_eps_curve = Helper.smooth(med_eps_avg_return, window=21)
    high_eps_curve = Helper.smooth(high_eps_avg_return, window=21)

    low_s2u_curve = Helper.smooth(low_s2u_avg_return, window=21)
    med_s2u_curve = Helper.smooth(med_s2u_avg_return, window=21)
    high_s2u_curve = Helper.smooth(high_s2u_avg_return, window=21)

    low_nbh_std_curve = Helper.smooth(low_nbh_std, window=21)
    med_nbh_std_curve = Helper.smooth(med_nbh_std, window=21)
    high_nbh_std_curve = Helper.smooth(high_nbh_std, window=21)

    low_lr_std_curve = Helper.smooth(low_lr_std, window=21)
    med_lr_std_curve = Helper.smooth(med_lr_std, window=21)
    high_lr_std_curve = Helper.smooth(high_lr_std, window=21)

    low_eps_std_curve = Helper.smooth(low_eps_std, window=21)
    med_eps_std_curve = Helper.smooth(med_eps_std, window=21)
    high_eps_std_curve = Helper.smooth(high_eps_std, window=21)

    low_s2u_std_curve = Helper.smooth(low_s2u_std, window=21)
    med_s2u_std_curve = Helper.smooth(med_s2u_std, window=21)
    high_s2u_std_curve = Helper.smooth(high_s2u_std, window=21)

    nbh_plotter.add_curve(np.linspace(5000, 1000000, 200), low_nbh_curve, "Low (1 Layer)", "r")
    nbh_plotter.add_curve(np.linspace(5000, 1000000, 200), med_nbh_curve, "Medium (3 Layers)", "g")
    nbh_plotter.add_curve(np.linspace(5000, 1000000, 200), high_nbh_curve, "High (5 Layers)", "b")

    lr_plotter.add_curve(np.linspace(5000, 1000000, 200), low_lr_curve, "Low (0.000005)", "r")
    lr_plotter.add_curve(np.linspace(5000, 1000000, 200), med_lr_curve, "Medium (0.00005)", "g")
    lr_plotter.add_curve(np.linspace(5000, 1000000, 200), high_lr_curve, "High (0.0005)", "b")

    eps_plotter.add_curve(np.linspace(5000, 1000000, 200), low_eps_curve, "Low (0.99999)", "r")
    eps_plotter.add_curve(np.linspace(5000, 1000000, 200), med_eps_curve, "Medium (0.99)", "g")
    eps_plotter.add_curve(np.linspace(5000, 1000000, 200), high_eps_curve, "High (0.9)", "b")

    s2u_plotter.add_curve(np.linspace(5000, 1000000, 200), low_s2u_curve, "Low (3 steps per update)", "r")
    s2u_plotter.add_curve(np.linspace(5000, 1000000, 200), med_s2u_curve, "Medium (5 steps per update)", "g")
    s2u_plotter.add_curve(np.linspace(5000, 1000000, 200), high_s2u_curve, "High (10 steps per update)", "b")

    nbh_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), low_nbh_curve-low_nbh_std_curve, np.where(low_nbh_curve+low_nbh_std_curve > 500, np.zeros(200)+500, low_nbh_curve+low_nbh_std_curve), color="r", alpha=0.4)
    nbh_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), med_nbh_curve-med_nbh_std_curve, np.where(med_nbh_curve+med_nbh_std_curve > 500, np.zeros(200)+500, med_nbh_curve+med_nbh_std_curve), color="g", alpha=0.4)
    nbh_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), high_nbh_curve-high_nbh_std_curve, np.where(high_nbh_curve+high_nbh_std_curve > 500, np.zeros(200)+500, high_nbh_curve+high_nbh_std_curve), color="b", alpha=0.4)

    lr_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), low_lr_curve-low_lr_std_curve, np.where(low_lr_curve+low_lr_std_curve > 500, np.zeros(200)+500, low_lr_curve+low_lr_std_curve), color="r", alpha=0.4)
    lr_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), med_lr_curve-med_lr_std_curve, np.where(med_lr_curve+med_lr_std_curve > 500, np.zeros(200)+500, med_lr_curve+med_lr_std_curve), color="g", alpha=0.4)
    lr_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), high_lr_curve-high_lr_std_curve, np.where(high_lr_curve+high_lr_std_curve > 500, np.zeros(200)+500, high_lr_curve+high_lr_std_curve), color="b", alpha=0.4)

    eps_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), low_eps_curve-low_eps_std_curve, np.where(low_eps_curve+low_eps_std_curve > 500, np.zeros(200)+500, low_eps_curve+low_eps_std_curve), color="r", alpha=0.4)
    eps_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), med_eps_curve-med_eps_std_curve, np.where(med_eps_curve+med_eps_std_curve > 500, np.zeros(200)+500, med_eps_curve+med_eps_std_curve), color="g", alpha=0.4)
    eps_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), high_eps_curve-high_eps_std_curve, np.where(high_eps_curve+high_eps_std_curve > 500, np.zeros(200)+500, high_eps_curve+high_eps_std_curve), color="b", alpha=0.4)

    s2u_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), low_s2u_curve-low_s2u_std_curve, np.where(low_s2u_curve+low_s2u_std_curve > 500, np.zeros(200)+500, low_s2u_curve+low_s2u_std_curve), color="r", alpha=0.4)
    s2u_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), med_s2u_curve-med_s2u_std_curve, np.where(med_s2u_curve+med_s2u_std_curve > 500, np.zeros(200)+500, med_s2u_curve+med_s2u_std_curve), color="g", alpha=0.4)
    s2u_plotter.ax.fill_between(np.linspace(5000, 1000000, 200), high_s2u_curve-high_s2u_std_curve, np.where(high_s2u_curve+high_s2u_std_curve > 500, np.zeros(200)+500, high_s2u_curve+high_s2u_std_curve), color="b", alpha=0.4)

    nbh_plotter.save("nbh.png")
    lr_plotter.save("lr.png")
    eps_plotter.save("eps.png")
    s2u_plotter.save("s2u.png")
    
def DQN_improvements():
    global run_n
    naive_returns = np.zeros((5, 200))
    er_returns = np.zeros((5, 200))
    tn_returns = np.zeros((5, 200))
    ertn_returns = np.zeros((5, 200))
    for i in range(5):
        run_n += 1 
        naive_net = DQN()
        er_net = DQN()
        tn_net = DQN()
        ertn_net = DQN()

        naive_net.to(device)
        er_net.to(device)
        tn_net.to(device)
        ertn_net.to(device)

        naive_returns[i] = train(naive_net, 50000, 0.99, 0.9, steps_per_update=10)
        with open('naive_improv.npy', 'wb') as f:
            np.save(f, naive_returns)
        er_returns[i] = train(er_net, 50000, 0.99, 0.9, replay_buffer=True, steps_per_update=10)
        with open('er.npy', 'wb') as f:
            np.save(f, er_returns)
        tn_returns[i] = train(tn_net, 50000, 0.99, 0.9, target_net=True, steps_per_update=10)
        with open('tn.npy', 'wb') as f:
            np.save(f, tn_returns)
        ertn_returns[i] = train(ertn_net, 50000, 0.99, 0.9, replay_buffer=True, target_net=True, steps_per_update=10)
        with open('ertn.npy', 'wb') as f:
            np.save(f, ertn_returns)

    env.close()
    naive_avg_return = np.average(naive_returns, axis=0)
    naive_std_dev = np.std(naive_returns, axis=0)
    er_avg_return = np.average(er_returns, axis=0)
    er_std_dev = np.std(er_returns, axis=0)
    tn_avg_return = np.average(tn_returns, axis=0)
    tn_std_dev = np.std(tn_returns, axis=0)
    ertn_avg_return = np.average(ertn_returns, axis=0)
    ertn_std_dev = np.std(ertn_returns, axis=0)


    
    plotter = Helper.LearningCurvePlot("DQN Versions Learning Curves", "Environment Step", "Episode Return")
    plotter.add_hline(500, "Optimal Performance")

    naive_curve = Helper.smooth(naive_avg_return, window=21)
    er_curve = Helper.smooth(er_avg_return, window=21)
    tn_curve = Helper.smooth(tn_avg_return, window=21)
    ertn_curve = Helper.smooth(ertn_avg_return, window=21)

    plotter.add_curve(np.linspace(5000, 1000000, 200), naive_curve, "Naive DQN", "orange")
    plotter.add_curve(np.linspace(5000, 1000000, 200), er_curve, "ER DQN", "b")
    plotter.add_curve(np.linspace(5000, 1000000, 200), tn_curve, "TN DQN", "g")
    plotter.add_curve(np.linspace(5000, 1000000, 200), ertn_curve, "ER+TN DQN", "m")

    naive_std_smooth = Helper.smooth(naive_std_dev, window=21)
    er_std_smooth = Helper.smooth(er_std_dev, window=21)
    tn_std_smooth = Helper.smooth(tn_std_dev, window=21)
    ertn_std_smooth = Helper.smooth(ertn_std_dev, window=21)

    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), naive_curve-naive_std_smooth, np.where(naive_curve+naive_std_smooth > 500, np.zeros(200)+500, naive_curve+naive_std_smooth), color="orange", alpha=0.4)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), er_curve-er_std_smooth, np.where(er_curve+er_std_smooth > 500, np.zeros(200)+500, er_curve+er_std_smooth), color="b", alpha=0.4)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), tn_curve-tn_std_smooth, np.where(tn_curve+tn_std_smooth > 500, np.zeros(200)+500, tn_curve+tn_std_smooth), color="g", alpha=0.4)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), ertn_curve-ertn_std_smooth, np.where(ertn_curve+ertn_std_smooth > 500, np.zeros(200)+500, ertn_curve+ertn_std_smooth), color="m", alpha=0.4)

    plotter.save("dqn_versions.png")

def main():
    if sys.argv[-1] == "naive":
        naive_DQN()
    elif sys.argv[-1] == "ablation":
        ablation_study
    elif sys.argv[-1] == "improvements":
        DQN_improvements()

if __name__ == "__main__":
    main()