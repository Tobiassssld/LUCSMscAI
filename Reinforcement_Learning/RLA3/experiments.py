import Helper
import a2c
import basic_ac_agent
import dqn_agent
import reinforce
import numpy as np
import torch
import gymnasium as gym
import argparse
import PPO

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def run_experiment():
    '''
    Runs the actual experiments for the carpole environment
    '''
    a2c_res = np.zeros((5, 200))
    dqn_res = np.zeros((5, 200))
    ppo_res = np.zeros((5, 200))
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
        with open('dqn.npy', 'wb') as f:
            np.save(f, dqn_res)

        agent = PPO.Agent(4).to(device)
        ppo_res[i] = PPO.train(agent)
        with open('ppo.npy', 'wb') as f:
            np.save(f, ppo_res)

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

    return dqn_res, reinforce_rewards, ac_rewards, a2c_res, ppo_res

def smooth_curves(results):
    '''
    retrieve the mean and standard deviaiton and smooth the curves from the results
    '''
    avg_return = np.mean(results, axis=0)
    std_dev = np.std(results, axis=0)

    smoothed_curve = Helper.smooth(avg_return, window=21)
    smoothed_std = Helper.smooth(std_dev, window=21)

    return smoothed_curve, smoothed_std


def cartpole_experiment():
    '''
    Run experiment on Cartpole and create plots
    '''
    dqn_res, reinforce_res, ac_res, a2c_res, ppo_res = run_experiment()

    dqn_mean, dqn_std = smooth_curves(dqn_res)
    reinforce_mean, reinforce_std = smooth_curves(reinforce_res)
    ac_mean, ac_std = smooth_curves(ac_res)
    a2c_mean, a2c_std = smooth_curves(a2c_res)
    ppo_mean, ppo_std = smooth_curves(ppo_res)

    plotter = Helper.LearningCurvePlot("Unified Learning Curve Graph", "Environment Step", "Episode Return")
    plotter.add_hline(500, "Optimal Performance")

    plotter.add_curve(np.linspace(5000, 1000000, 200), dqn_mean, "DQN", "orange")
    plotter.add_curve(np.linspace(5000, 1000000, 200), reinforce_mean, "REINFORCE", "m")
    plotter.add_curve(np.linspace(5000, 1000000, 200), ac_mean, "AC", "c")
    plotter.add_curve(np.linspace(5000, 1000000, 200), a2c_mean, "A2C", "lime")
    plotter.add_curve(np.linspace(5000, 1000000, 200), ppo_mean, "PPO", "y")

    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), dqn_mean-dqn_std, np.where(dqn_mean+dqn_std > 500, np.zeros(200)+500, dqn_mean+dqn_std), color="orange", alpha=0.2)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), reinforce_mean-reinforce_std, np.where(reinforce_mean+reinforce_std > 500, np.zeros(200)+500, reinforce_mean+reinforce_std), color="m", alpha=0.2)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), ac_mean-ac_std, np.where(a2c_mean+ac_std > 500, np.zeros(200)+500, a2c_mean+ac_std), color="c", alpha=0.2)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), a2c_mean-a2c_std, np.where(a2c_mean+a2c_std > 500, np.zeros(200)+500, a2c_mean+a2c_std), color="lime", alpha=0.2)
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), ppo_mean-ppo_std, np.where(ppo_mean+ppo_std > 500, np.zeros(200)+500, ppo_mean+ppo_std), color="y", alpha=0.2)

    plotter.save("unified_plot.png")

    plotter = Helper.LearningCurvePlot("PPO Individual Run Returns", "Environment Step", "Episode Return")
    plotter.add_curve(np.linspace(5000, 200000, 40), ppo_res[0,:40], "Run 1", "orange")
    plotter.add_curve(np.linspace(5000, 200000, 40), ppo_res[1,:40], "Run 2", "m")
    plotter.add_curve(np.linspace(5000, 200000, 40), ppo_res[2,:40], "Run 3", "c")
    plotter.add_curve(np.linspace(5000, 200000, 40), ppo_res[3,:40], "Run 4", "lime")
    plotter.add_curve(np.linspace(5000, 200000, 40), ppo_res[4,:40], "Run 5", "y")
    plotter.save("individual_runs_plot.png")


def other_envs_experiment(env_id):
    '''
    Run experiment on Acrobot and Lunar Lander and create plots
    '''
    acrobot_res = np.zeros((5, 200))
    lunar_lander_res = np.zeros((5, 200))
    for i in range(5):
        agent = PPO.Agent(6, output_size=3).to(device)
        acrobot_res[i] = PPO.train(agent, env_id="Acrobot-v1")
        with open('acrobot_res.npy', 'wb') as f:
            np.save(f, acrobot_res)

    
        agent = PPO.Agent(8, output_size=4).to(device)
        lunar_lander_res[i] = PPO.train(agent, env_id="LunarLander-v3")
        with open('lunar_lander_res.npy', 'wb') as f:
            np.save(f, lunar_lander_res)

    plotter = Helper.LearningCurvePlot("Acrobot Learning Curve Graph", "Environment Step", "Episode Return")
    plotter.add_hline(0, "Optimal Performance")

    acrobot_mean, acrobot_std = smooth_curves(acrobot_res)
    plotter.add_curve(np.linspace(5000, 1000000, 200), acrobot_mean, "PPO", "y")
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), acrobot_mean-acrobot_std, acrobot_mean+acrobot_std, color="y", alpha=0.2)
    plotter.save("acrobot_plot.png")

    plotter = Helper.LearningCurvePlot("Lunar Lander Learning Curve Graph", "Environment Step", "Episode Return")
    plotter.add_hline(200, "Solution Threshold")

    lunar_lander_mean, lunar_lander_std = smooth_curves(lunar_lander_res)
    plotter.add_curve(np.linspace(5000, 1000000, 200), lunar_lander_mean, "PPO", "y")
    plotter.ax.fill_between(np.linspace(5000, 1000000, 200), lunar_lander_mean-lunar_lander_std, lunar_lander_mean+lunar_lander_std, color="y", alpha=0.2)
    plotter.save("lunar_lander_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole', help='Environment')

    args = parser.parse_args()

    if args.env == "cartpole":
        print("Starting cartpole experiment")
        cartpole_experiment()
    elif args.env == "other_envs":
        print("Starting other environment experiment")
        other_envs_experiment(args.env)
    else:
        print("Experiment not defined, please choose cartpole or other_envs")