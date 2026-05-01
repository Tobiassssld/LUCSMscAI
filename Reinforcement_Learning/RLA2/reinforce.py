import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import argparse

# Create results folder
os.makedirs("results", exist_ok=True)


# Setup GPU
def s4213211_setup_gpu():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s).")

            if len(gpus) > 0:
                try:
                    mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
                    print("Mixed precision training enabled")
                except:
                    print("Mixed precision not supported")
        except RuntimeError as e:
            print(f"Error with GPU: {e}")
    else:
        print("Using CPU.")


# Policy network
def s4213211_make_policy_network(state_dim, action_dim, hidden_size=[128, 64]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(state_dim,)))

    for units in hidden_size:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        model.add(tf.keras.layers.Dense(action_dim, activation='softmax', dtype='float32'))
    else:
        model.add(tf.keras.layers.Dense(action_dim, activation='softmax'))

    return model


# REINFORCE Agent
class s4213211_REINFORCEAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 hidden_size=[128, 64],
                 normalize_returns=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.normalize_returns = normalize_returns

        # using s4213211_make_policy_network func() to build models
        self.policy = s4213211_make_policy_network(state_dim, action_dim, hidden_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def get_action_probs(self, states):
        return self.policy(states)

    def get_action(self, state, deterministic=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.get_action_probs(state_tensor)[0].numpy()

        if deterministic:
            return np.argmax(action_probs)
        else:
            return np.random.choice(self.action_dim, p=action_probs)

    def evaluate(self, env, num_episodes=5):
        """Evaluate current policy performance"""
        eval_rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Use deterministic policy for evaluation
                action = self.get_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                state = next_state

            eval_rewards.append(total_reward)

        return np.mean(eval_rewards)

    def calculate_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        discounted_sum = 0

        # Going backwards through rewards
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + self.gamma * discounted_sum
            returns[t] = discounted_sum

        # Normalize if needed
        if self.normalize_returns and len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        return returns

    @tf.function
    def train_step(self, states, actions, returns):
        with tf.GradientTape() as tape:
            action_probs = self.policy(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-10)
            loss = -tf.reduce_mean(log_probs * returns)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return loss

    def learn(self, states, actions, rewards):
        returns = self.calculate_returns(rewards)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        loss = self.train_step(states, actions, returns)
        return loss.numpy()


# Helper for environments
def s4213211_make_env(env_name):
    def _init():
        return gym.make(env_name)

    return _init


# Main training function
def s4213211_train_reinforce(env_name='CartPole-v1',
                             max_env_steps=100000,
                             learning_rate=0.001,
                             gamma=0.99,
                             hidden_size=[128, 64],
                             normalize_returns=True,
                             num_envs=4,
                             repetitions=5,
                             render=False,
                             eval_interval=5000,  # Evaluate every 5000 steps
                             eval_episodes=5):  # Run 5 episodes for each evaluation
    print(f"\n=== Training REINFORCE ===")
    print(f"Environment: {env_name}")
    print(f"Steps: {max_env_steps}, Repetitions: {repetitions}, Parallel envs: {num_envs}")

    all_eval_rewards = []  # Store evaluation rewards
    all_eval_steps = []  # Store evaluation steps
    total_train_time = 0

    for rep in range(repetitions):
        print(f"Repetition {rep + 1}/{repetitions}")

        # Setup training environment
        if num_envs > 1 and not render:
            env = AsyncVectorEnv([s4213211_make_env(env_name) for _ in range(num_envs)])
            temp_env = gym.make(env_name)
            state_size = temp_env.observation_space.shape[0]
            action_size = temp_env.action_space.n
            temp_env.close()
        else:
            env = gym.make(env_name, render_mode="human" if render else None)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

        # Create evaluation environment
        eval_env = gym.make(env_name)

        # Create agent
        agent = s4213211_REINFORCEAgent(
            state_dim=state_size,
            action_dim=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            hidden_size=hidden_size,
            normalize_returns=normalize_returns
        )

        eval_rewards = []
        eval_steps = []
        total_env_steps = 0
        episode = 0
        next_eval_step = eval_interval
        start_time = time.time()

        # Vectorized training
        if num_envs > 1 and not render:
            trajectories = [[] for _ in range(num_envs)]
            states, _ = env.reset()
            dones = np.zeros(num_envs, dtype=bool)
            episode_rewards_vec = np.zeros(num_envs)

            while total_env_steps < max_env_steps:
                actions = np.array([agent.get_action(states[i]) for i in range(num_envs)])
                next_states, rewards, terminated, truncated, _ = env.step(actions)
                dones = terminated | truncated

                for i in range(num_envs):
                    trajectories[i].append((states[i], actions[i], rewards[i]))
                    episode_rewards_vec[i] += rewards[i]

                states = next_states
                total_env_steps += num_envs

                # Process completed episodes
                for i in range(num_envs):
                    if dones[i]:
                        states_batch, actions_batch, rewards_batch = [], [], []
                        for state, action, reward in trajectories[i]:
                            states_batch.append(state)
                            actions_batch.append(action)
                            rewards_batch.append(reward)

                        if len(states_batch) > 0:
                            loss = agent.learn(states_batch, actions_batch, rewards_batch)

                        trajectories[i] = []
                        episode_rewards_vec[i] = 0
                        episode += 1

                # Periodic evaluation
                if total_env_steps >= next_eval_step:
                    eval_reward = agent.evaluate(eval_env, num_episodes=eval_episodes)
                    eval_rewards.append(eval_reward)
                    eval_steps.append(total_env_steps)
                    next_eval_step += eval_interval

                    print(f"Step: {total_env_steps}, Evaluation reward: {eval_reward:.1f}")

                # Print progress
                if episode % 10 == 0 and episode > 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode: {episode}, Steps: {total_env_steps}/{max_env_steps}")

        else:
            # Single environment training
            while total_env_steps < max_env_steps:
                state, _ = env.reset()
                done = False

                states_batch, actions_batch, rewards_batch = [], [], []
                episode_reward = 0

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    states_batch.append(state)
                    actions_batch.append(action)
                    rewards_batch.append(reward)

                    state = next_state
                    episode_reward += reward
                    total_env_steps += 1

                    # Periodic evaluation
                    if total_env_steps >= next_eval_step:
                        eval_reward = agent.evaluate(eval_env, num_episodes=eval_episodes)
                        eval_rewards.append(eval_reward)
                        eval_steps.append(total_env_steps)
                        next_eval_step += eval_interval

                        print(f"Step: {total_env_steps}, Evaluation reward: {eval_reward:.1f}")

                    if total_env_steps >= max_env_steps:
                        break

                if len(states_batch) > 0:
                    loss = agent.learn(states_batch, actions_batch, rewards_batch)

                episode += 1

                if episode % 10 == 0:
                    print(f"Episode: {episode}, Steps: {total_env_steps}/{max_env_steps}")

        episode_time = time.time() - start_time
        total_train_time += episode_time
        all_eval_rewards.append(eval_rewards)
        all_eval_steps.append(eval_steps)
        print(f"Repetition {rep + 1} finished in {episode_time:.2f} seconds")

        eval_env.close()

    env.close()
    print(f"Total training time: {total_train_time:.2f} seconds")

    return all_eval_rewards, all_eval_steps


# Process data for plotting
def s4213211_process_plot_data(rewards, env_steps):
    # No interpolation needed since data is already at fixed intervals
    processed_data = {
        'REINFORCE': {
            'step_points': [],
            'scores': []
        }
    }

    for rep_idx in range(len(rewards)):
        if rep_idx >= len(env_steps):
            continue

        # Collect step points from first repetition
        if len(processed_data['REINFORCE']['step_points']) == 0 and len(env_steps[rep_idx]) > 0:
            processed_data['REINFORCE']['step_points'] = np.array(env_steps[rep_idx])

        processed_data['REINFORCE']['scores'].append(rewards[rep_idx])

    if processed_data['REINFORCE']['scores']:
        processed_data['REINFORCE']['scores'] = np.array(processed_data['REINFORCE']['scores'])
    else:
        processed_data['REINFORCE']['scores'] = np.array([[0]])
        print("Warning: No valid data for plotting")

    return processed_data


# Plot results
def s4213211_plot_results(processed_data, env_name='CartPole-v1', smoothing=5):
    plt.figure(figsize=(14, 8))

    data = processed_data['REINFORCE']

    # Calculate mean and std
    mean_scores = np.mean(data['scores'], axis=0)
    std_scores = np.std(data['scores'], axis=0)

    # Smooth if needed
    if smoothing > 0 and len(mean_scores) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        mean_scores = np.convolve(mean_scores, kernel, mode='same')
        std_scores = np.convolve(std_scores, kernel, mode='same')

        # Fix edges
        mean_scores[:smoothing // 2] = mean_scores[smoothing // 2]
        std_scores[:smoothing // 2] = std_scores[smoothing // 2]
        mean_scores[-smoothing // 2:] = mean_scores[-smoothing // 2 - 1]
        std_scores[-smoothing // 2:] = std_scores[-smoothing // 2 - 1]

    # Plot
    x = data['step_points'] / 1000
    plt.plot(x, mean_scores, label='REINFORCE', linewidth=2)
    plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

    plt.title(f"REINFORCE Learning Curve on {env_name}", fontsize=16)
    plt.xlabel('Environment Steps (K)', fontsize=14)
    plt.ylabel('Average Return', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/reinforce_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Main function
if __name__ == "__main__":
    # Setup GPU
    s4213211_setup_gpu()

    # Parse arguments
    parser = argparse.ArgumentParser(description='REINFORCE Algorithm Training')
    parser.add_argument('--steps', type=int, default=1000000, help='Maximum environment steps')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')

    args = parser.parse_args()

    # Train REINFORCE
    rewards, env_steps = s4213211_train_reinforce(
        max_env_steps=args.steps,
        repetitions=args.reps,
        num_envs=args.num_envs
    )

    # Plot results
    processed_data = s4213211_process_plot_data(rewards, env_steps)
    s4213211_plot_results(processed_data)

    print("Training complete.")