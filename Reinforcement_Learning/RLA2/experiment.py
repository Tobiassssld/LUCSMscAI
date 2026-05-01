import Helper
import a2c
import basic_ac_agent
import dqn_agent
import reinforce
import numpy as np
import torch
import gymnasium as gym

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def run_experiment():
    a2c_res = np.zeros((5, 200))
    dqn_res = np.zeros((5, 200))
    for i in range(5):
        actor_net = a2c.Actor(lr=0.0007)
        actor_net.to(device)
        critic_net = a2c.Critic(lr=0.0007)
        critic_net.to(device)
        env = gym.make("CartPole-v1", render_mode=None)
        a2c_res[i] = a2c.train(actor_net, critic_net, 50000)
        with open('a2c_res.npy', 'wb') as f:
            np.save(f, a2c_res)

        dqn = dqn_agent.DQN()
        dqn.to(device)
        dqn_res[i] = dqn_agent.train(dqn, 50000, 0.99, 0.9, replay_buffer=True, target_net=True, steps_per_update=10)
        with open('low_nbh.npy', 'wb') as f:
            np.save(f, dqn_res)

    ac_rewards, ac_eval_steps, ac_train_rewards, ac_train_steps = basic_ac_agent.s4213211_run_ac_experiment(
        actor_lr=0.00005,
        critic_lr=0.0001,
        gamma=0.99,
        hidden_size=128,
        max_env_steps=1000000,
        repetitions=5,
        num_envs=4,
        render=False,
        eval_interval=5000,
        eval_episodes=5
    )
    ac_rewards = np.array(ac_rewards)
    
    with open('ac_results.npy', 'wb') as f:
        np.save(f, ac_rewards)

    reinforce_rewards, _ = reinforce.s4213211_train_reinforce(max_env_steps=1000000, repetitions=5, num_envs=4)
    reinforce_rewards = np.array(reinforce_rewards)

    with open('reinforce_res.npy', 'wb') as f:
        np.save(f, reinforce_rewards)

    return dqn_res, reinforce_rewards, ac_rewards, a2c_res

def smooth_curves(results):
    avg_return = np.mean(results, axis=0)
    std_dev = np.std(results, axis=0)

    smoothed_curve = Helper.smooth(avg_return, window=21)
    smoothed_std = Helper.smooth(std_dev, window=21)

    return smoothed_curve, smoothed_std


if __name__ == "__main__":
    dqn_res, reinforce_res, ac_res, a2c_res = run_experiment()

    dqn_mean, dqn_std = smooth_curves(dqn_res)
    reinforce_mean, reinforce_std = smooth_curves(reinforce_res)
    ac_mean, ac_std = smooth_curves(ac_res)
    a2c_mean, a2c_std = smooth_curves(a2c_res)

    plotter = Helper.LearningCurvePlot("Unified Learning Curve Graph", "Environment Step", "Episode Return")
    plotter.add_hline(500, "Optimal Performance")

    plotter.add_curve(np.linspace(5000, 1000000, 200), dqn_mean, "DQN", "orange")
    plotter.add_curve(np.linspace(5000, 1000000, 200), reinforce_mean, "REINFORCE", "m")
    plotter.add_curve(np.linspace(5000, 1000000, 200), ac_mean, "AC", "c")
    plotter.add_curve(np.linspace(5000, 1000000, 200), a2c_mean, "A2C", "lime")

    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), dqn_mean-dqn_std, np.where(dqn_mean+dqn_std > 500, np.zeros(200)+500, dqn_mean+dqn_std), color="orange", alpha=0.3)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), reinforce_mean-reinforce_std, np.where(reinforce_mean+reinforce_std > 500, np.zeros(200)+500, reinforce_mean+reinforce_std), color="r", alpha=0.3)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), ac_mean-ac_std, np.where(a2c_mean+ac_std > 500, np.zeros(200)+500, a2c_mean+ac_std), color="c", alpha=0.3)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), a2c_mean-a2c_std, np.where(a2c_mean+a2c_std > 500, np.zeros(200)+500, a2c_mean+a2c_std), color="lime", alpha=0.3)

    plotter.save("unified_plot.png")


    