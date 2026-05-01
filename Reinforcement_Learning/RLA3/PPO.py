import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import tensor
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium.wrappers as gym_wrappers

#device selection (try to use gpu if available)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Agent(nn.Module):
    '''
    Agent class, contains the actor and critic networks and used to interact with them for the most part
    '''
    def __init__(self, input_size, neurons_per_layer=32, lr = 0.0003, output_size=2):
        '''
        Creates the agent, initializes both networks and the optimizer
        '''
        super(Agent, self).__init__()
        self.input_size = input_size
        self.critic = nn.Sequential(nn.Linear(input_size, neurons_per_layer), 
                                     nn.ReLU(),
                                     nn.Linear(neurons_per_layer, neurons_per_layer),
                                     nn.ReLU(),
                                     nn.Linear(neurons_per_layer, 1))
        self.actor = nn.Sequential(nn.Linear(input_size, neurons_per_layer), 
                                     nn.ReLU(),
                                     nn.Linear(neurons_per_layer, neurons_per_layer),
                                     nn.ReLU(),
                                     nn.Linear(neurons_per_layer, output_size))
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        
    def get_state_value(self, state):
        '''
        Evaluates the current state observation using the critic network
        '''
        return self.critic(state)
    
    def get_action(self, state, action=None):
        '''
        Used to either sample an action from the policy and get the log probability given an observation
        or to retrieve the log probability of a previously taken action to compute the surrogate objective 
        '''
        net_out = self.actor(state)
        distrib = torch.distributions.categorical.Categorical(logits=net_out)
        if action is None:
            action = distrib.sample()
        return action, distrib.log_prob(action), distrib.entropy()

def train(agent, env_id="CartPole-v1", max_steps=1000000, steps_per_descent=512, report_freq=5000, num_minibatches=8, gamma=0.99, lamb=0.95, num_epochs=4, clip_coef=0.2, entropy_coef=0.01, flatten=False):
    '''
    Main training logic
    '''
    batch_size = steps_per_descent // num_minibatches
    tot_s = 0
    env = gym.make(env_id)
    env = gym_wrappers.RecordEpisodeStatistics(env)
    env = gym_wrappers.NumpyToTorch(env)
    if flatten:
        env = gym_wrappers.FlattenObservation(env)
    last_r = 0
    results = []

    while tot_s < max_steps:
        obs = torch.zeros((steps_per_descent, agent.input_size), device=device)
        actions = torch.zeros(steps_per_descent, device=device, dtype=torch.int64)
        probs = torch.zeros(steps_per_descent, device=device)
        ep_over = torch.zeros(steps_per_descent, dtype=torch.bool, device=device)
        rewards = torch.zeros(steps_per_descent, device=device)
        values = torch.zeros(steps_per_descent, device=device)

        observation, _ = env.reset()
        observation = observation.to(device)
        episode_over = False
        for t in range(steps_per_descent):
            '''
            samples steps_per_descent steps from the environment following the current policy
            '''
            obs[t] = observation
            ep_over[t] = episode_over
            with torch.no_grad():
                val = agent.get_state_value(observation)
                values[t] = val
                action, prob, _ = agent.get_action(observation)
            actions[t] = action
            probs[t] = prob
            observation, reward, terminated, truncated, info = env.step(action.cpu())
            rewards[t] = reward
            episode_over = terminated or truncated

            if "episode" in info.keys():
                last_r = info["episode"]['r']

            if (t + tot_s) % report_freq == 0 and tot_s+t != 0:
                results.append(last_r)
                print(f'step {tot_s+t}, reward: {last_r}')

            if episode_over:
                observation, _ = env.reset()
            observation = observation.to(device)
                
        tot_s += steps_per_descent

        with torch.no_grad():
            '''
            Generalized Advantage Estimation (GAE) computation from the sample steps
            '''
            next_val = agent.get_state_value(observation)
            last_adv = 0
            advantages = torch.zeros(steps_per_descent, device=device)
            for t in reversed(range(steps_per_descent)):
                if t == steps_per_descent - 1:
                    nonterminal = int(not episode_over)
                else:
                    nonterminal = int(not ep_over[t+1])
                    next_val = values[t+1]
                delta = rewards[t] + gamma * next_val * nonterminal - values[t]
                last_adv = delta + gamma * lamb * nonterminal * last_adv
                advantages[t] = last_adv
            returns = values + advantages
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.arange(steps_per_descent)
        for epoch in range(num_epochs):
            '''
            Main gradient descent logic
            '''
            np.random.shuffle(indices) #shuffle the samples
            for minibatch in range(num_minibatches):
                minibatch_indices = indices[minibatch*batch_size:(minibatch+1)*batch_size] #get the minibatch's samples for each tensor used
                batch_states = obs[minibatch_indices]
                batch_actions = actions[minibatch_indices]
                batch_adv = advantages[minibatch_indices]
                batch_probs = probs[minibatch_indices]
                batch_returns = returns[minibatch_indices]

                new_value = agent.get_state_value(batch_states)
                _, new_probs, entropy = agent.get_action(batch_states, batch_actions)

                ratio = torch.exp(new_probs - batch_probs)
                clipped_ratio = torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
                policy_loss = -torch.minimum(ratio * batch_adv, clipped_ratio * batch_adv).mean() #compute clip objective

                value_loss = (new_value - batch_returns).pow(2).mean()

                loss = policy_loss + value_loss - entropy_coef * entropy.mean()

                agent.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                agent.optimizer.step()
        
    env.close()
    return results

if __name__ == "__main__":
    input_size = 8
    agent = Agent(lr=7.5e-5, input_size=input_size, output_size=4).to(device)

    res = train(agent, env_id="LunarLander-v3", steps_per_descent=1024, num_minibatches=8, num_epochs=4)
    plt.plot(res)
    plt.show()