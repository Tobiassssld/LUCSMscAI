import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import os
import argparse

# Make results folder
os.makedirs("results", exist_ok=True)


# GPU setup function
def s4213211_setup_gpu():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")

            # Note: To avoid mixed precision issues, we stick to float32
            print("Using float32 precision for all operations")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")
    else:
        print("No GPU found. Using CPU.")

    print("Devices:", tf.config.list_logical_devices())


# Actor-Critic Network class
class s4213211_ActorCriticNetwork:
    def __init__(self, state_size, action_size, actor_learning_rate=0.001, critic_learning_rate=0.001, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size

        # Policy network (Actor)
        self.actor_model = self._build_actor_model(hidden_size)
        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)

        # Value network (Critic)
        self.critic_model = self._build_critic_model(hidden_size)
        self.critic_optimizer = Adam(learning_rate=critic_learning_rate)

    def _build_actor_model(self, hidden_size):
        model = Sequential([
            Dense(hidden_size, activation='relu', input_dim=self.state_size),
            Dense(hidden_size, activation='relu'),
            Dense(self.action_size, activation='softmax')  # output action probabilities
        ])
        return model

    def _build_critic_model(self, hidden_size):
        model = Sequential([
            Dense(hidden_size, activation='relu', input_dim=self.state_size),
            Dense(hidden_size, activation='relu'),
            Dense(1)  # output state value
        ])
        return model

    @tf.function
    def get_actor_policy_batch(self, states):
        return self.actor_model(states)

    @tf.function
    def get_critic_value_batch(self, states):
        return self.critic_model(states)

    @tf.function
    def train_actor_batch(self, states, actions, td_errors):
        with tf.GradientTape() as tape:
            # Get policy probabilities
            policy_probs = self.actor_model(states)

            # Create action masks
            action_masks = tf.one_hot(actions, self.action_size)

            # Calculate log probability of selected actions
            selected_probs = tf.reduce_sum(policy_probs * action_masks, axis=1)
            # Add small constant to avoid log(0)
            log_probs = tf.math.log(selected_probs + 1e-10)

            # Policy loss: maximize log(π(a|s)) * TD-error
            actor_loss = -tf.reduce_mean(log_probs * td_errors)

        # Calculate and apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        return actor_loss

    @tf.function
    def train_critic_batch(self, states, targets):
        with tf.GradientTape() as tape:
            # Get value predictions
            values = self.critic_model(states)

            # Ensure consistent data types
            values = tf.cast(values, dtype=tf.float32)
            targets = tf.cast(targets, dtype=tf.float32)

            # Calculate MSE loss
            critic_loss = tf.reduce_mean(tf.square(targets - values))

        # Calculate and apply gradients
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))
        return critic_loss


# Actor-Critic Agent
class s4213211_ACAgent:
    def __init__(self, state_size, action_size, gamma=0.99,
                 actor_lr=0.001, critic_lr=0.001,
                 hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Create Actor-Critic network
        self.network = s4213211_ActorCriticNetwork(
            state_size,
            action_size,
            actor_lr,
            critic_lr,
            hidden_size
        )

    def get_actions(self, states, deterministic=False):
        """Choose actions based on states (batch version)"""
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        policy_probs = self.network.get_actor_policy_batch(states_tensor).numpy()

        if deterministic:
            actions = np.argmax(policy_probs, axis=1)
        else:
            actions = np.array([np.random.choice(self.action_size, p=probs) for probs in policy_probs])

        return actions

    def train_batch(self, states, actions, rewards, next_states, dones):
        """Train Actor and Critic using batch data"""
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # Get current and next state values
        current_values = self.network.get_critic_value_batch(states_tensor).numpy().flatten()
        next_values = self.network.get_critic_value_batch(next_states_tensor).numpy().flatten()

        # Calculate TD targets
        targets = rewards + (1 - dones.astype(float)) * self.gamma * next_values

        # Calculate TD errors
        td_errors = targets - current_values

        # Train Critic network
        critic_loss = self.network.train_critic_batch(
            states_tensor,
            tf.convert_to_tensor(targets.reshape(-1, 1), dtype=tf.float32)
        )

        # Train Actor network
        actor_loss = self.network.train_actor_batch(
            states_tensor,
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(td_errors, dtype=tf.float32)
        )

        return actor_loss, critic_loss

    def evaluate(self, env, num_episodes=5):
        """Evaluate current policy performance"""
        eval_rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Use deterministic policy for evaluation
                action = self.get_actions(np.expand_dims(state, axis=0), deterministic=True)[0]
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                state = next_state

            eval_rewards.append(total_reward)

        return np.mean(eval_rewards)


# Helper function to create vectorized environments
def s4213211_make_env(env_name):
    """Helper function to create environment"""

    def _init():
        return gym.make(env_name)

    return _init


# Experiment run function
def s4213211_run_ac_experiment(actor_lr=0.001,
                               critic_lr=0.001,
                               gamma=0.99,
                               hidden_size=128,
                               max_env_steps=100000,
                               repetitions=5,
                               env_name='CartPole-v1',
                               num_envs=8,
                               render=False,
                               eval_interval=5000,
                               eval_episodes=5):
    print(f"\n=== Running Basic Actor-Critic Algorithm ===")
    print(f"Actor LR: {actor_lr}, Critic LR: {critic_lr}")
    print(f"Hidden size: {hidden_size}, Gamma: {gamma}")
    print(f"Using {num_envs} parallel environments")
    print(f"Evaluation every {eval_interval} steps, {eval_episodes} episodes per evaluation")

    all_eval_rewards = []  # Store evaluation rewards
    all_eval_steps = []  # Store evaluation steps
    all_train_rewards = []  # Store training episode rewards
    all_train_steps = []  # Store training episode steps

    total_train_time = 0

    for rep in range(repetitions):
        print(f"Repetition {rep + 1}/{repetitions}")

        # Create vectorized environment for training
        if num_envs > 1 and not render:
            env = AsyncVectorEnv([s4213211_make_env(env_name) for _ in range(num_envs)])
            # Create a single environment to get state/action dimensions
            temp_env = gym.make(env_name)
            state_size = temp_env.observation_space.shape[0]
            action_size = temp_env.action_space.n
            temp_env.close()
        else:
            # Use single environment if rendering or num_envs=1
            env = gym.make(env_name, render_mode="human" if render else None)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

        # Create evaluation environment
        eval_env = gym.make(env_name)

        # Create Agent
        agent = s4213211_ACAgent(
            state_size=state_size,
            action_size=action_size,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            hidden_size=hidden_size
        )

        train_rewards = []
        train_steps = []
        eval_rewards = []
        eval_steps = []

        total_env_steps = 0
        completed_episodes = 0
        next_eval_step = eval_interval

        start_time = time.time()

        # Initialize tracking variables for vectorized environment
        if num_envs > 1 and not render:
            states, _ = env.reset()
            episode_rewards = np.zeros(num_envs)
            episode_steps = np.zeros(num_envs, dtype=np.int32)

            while total_env_steps < max_env_steps:
                # Choose actions for all environments
                actions = agent.get_actions(states)

                # Execute actions in all environments
                next_states, rewards, terminateds, truncateds, _ = env.step(actions)
                dones = np.logical_or(terminateds, truncateds)

                # Update episode tracking
                episode_rewards += rewards
                episode_steps += 1
                total_env_steps += num_envs

                # Train agent on batch
                agent.train_batch(states, actions, rewards, next_states, dones)

                # Process completed episodes
                for i in range(num_envs):
                    if dones[i]:
                        train_rewards.append(episode_rewards[i])
                        train_steps.append(total_env_steps)

                        episode_rewards[i] = 0
                        episode_steps[i] = 0
                        completed_episodes += 1

                        if completed_episodes % 10 == 0:
                            elapsed_time = time.time() - start_time
                            steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                            print(f"Episodes: {completed_episodes}, Steps: {total_env_steps}/{max_env_steps}, "
                                  f"Speed: {steps_per_sec:.1f} steps/s")

                # Periodic evaluation
                if total_env_steps >= next_eval_step:
                    eval_reward = agent.evaluate(eval_env, num_episodes=eval_episodes)
                    eval_rewards.append(eval_reward)
                    eval_steps.append(total_env_steps)
                    next_eval_step += eval_interval

                    print(f"Step: {total_env_steps}, Evaluation reward: {eval_reward:.1f}")

                states = next_states

        else:
            # Single environment training loop
            while total_env_steps < max_env_steps:
                state, _ = env.reset()
                done = False
                score = 0
                episode_steps = 0

                while not done:
                    # Choose action
                    action = agent.get_actions(np.expand_dims(state, axis=0))[0]

                    # Execute action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Train agent
                    agent.train_batch(
                        np.expand_dims(state, axis=0),
                        np.array([action]),
                        np.array([reward]),
                        np.expand_dims(next_state, axis=0),
                        np.array([done])
                    )

                    state = next_state
                    score += reward
                    episode_steps += 1
                    total_env_steps += 1

                    # Periodic evaluation
                    if total_env_steps >= next_eval_step:
                        eval_reward = agent.evaluate(eval_env, num_episodes=eval_episodes)
                        eval_rewards.append(eval_reward)
                        eval_steps.append(total_env_steps)
                        next_eval_step += eval_interval

                        print(f"Step: {total_env_steps}, Evaluation reward: {eval_reward:.1f}")

                    if done or total_env_steps >= max_env_steps:
                        break

                # Record training episode results
                train_rewards.append(score)
                train_steps.append(total_env_steps)
                completed_episodes += 1

                if completed_episodes % 10 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episodes: {completed_episodes}, Score: {score}, Steps: {total_env_steps}/{max_env_steps}, "
                          f"Speed: {steps_per_sec:.1f} steps/s")

        episode_time = time.time() - start_time
        total_train_time += episode_time

        all_train_rewards.append(train_rewards)
        all_train_steps.append(train_steps)
        all_eval_rewards.append(eval_rewards)
        all_eval_steps.append(eval_steps)

        print(
            f"Repetition {rep + 1} done in {episode_time:.2f} seconds, speed: {total_env_steps / episode_time:.1f} steps/sec")

        env.close()
        eval_env.close()

    print(f"Total training time: {total_train_time:.2f} seconds")
    return all_eval_rewards, all_eval_steps, all_train_rewards, all_train_steps


# Process plot data function
def s4213211_process_plot_data(eval_rewards, eval_steps, train_rewards=None, train_steps=None):
    processed_data = {}

    # Find maximum steps across all runs
    max_steps = 0
    for run_steps in eval_steps:
        if run_steps and len(run_steps) > 0:
            max_steps = max(max_steps, run_steps[-1])

    step_interval = 5000
    step_points = np.arange(0, max_steps + step_interval, step_interval)

    # Process evaluation data
    eval_interpolated_scores = []
    for rep_idx in range(len(eval_rewards)):
        # Skip if data is missing
        if rep_idx >= len(eval_steps) or not eval_steps[rep_idx] or not eval_rewards[rep_idx]:
            continue

        rep_scores = eval_rewards[rep_idx]
        rep_steps = eval_steps[rep_idx]

        # Linear interpolation
        try:
            interp_scores = np.interp(
                step_points,
                rep_steps,
                rep_scores,
                left=0,
                right=rep_scores[-1]
            )
            eval_interpolated_scores.append(interp_scores)
        except Exception as e:
            print(f"Evaluation interpolation error for rep {rep_idx}: {e}")

    if eval_interpolated_scores:
        processed_data['Evaluation'] = {
            'step_points': step_points,
            'scores': np.array(eval_interpolated_scores)
        }

    # Process training data if provided
    if train_rewards and train_steps:
        train_interpolated_scores = []

        for rep_idx in range(len(train_rewards)):
            if rep_idx >= len(train_steps) or not train_steps[rep_idx] or not train_rewards[rep_idx]:
                continue

            rep_scores = train_rewards[rep_idx]
            rep_steps = train_steps[rep_idx]

            # Smooth training data with window
            window_size = 10
            smoothed_scores = []
            smoothed_steps = []

            for i in range(0, len(rep_scores), window_size):
                end = min(i + window_size, len(rep_scores))
                if end > i:
                    smoothed_scores.append(np.mean(rep_scores[i:end]))
                    smoothed_steps.append(rep_steps[min(end - 1, len(rep_steps) - 1)])

            if not smoothed_scores or not smoothed_steps:
                continue

            try:
                interp_scores = np.interp(
                    step_points,
                    smoothed_steps,
                    smoothed_scores,
                    left=0,
                    right=smoothed_scores[-1]
                )
                train_interpolated_scores.append(interp_scores)
            except Exception as e:
                print(f"Training data interpolation error for rep {rep_idx}: {e}")

        if train_interpolated_scores:
            processed_data['Training'] = {
                'step_points': step_points,
                'scores': np.array(train_interpolated_scores)
            }

    return processed_data


# Plot learning curves
def s4213211_plot_learning_curves(eval_rewards, eval_steps, train_rewards=None, train_steps=None,
                                  env_name='CartPole-v1', smoothing=5):
    # Process evaluation data
    processed_data = s4213211_process_plot_data(eval_rewards, eval_steps, None, None)

    if not processed_data or 'Evaluation' not in processed_data:
        print("No valid evaluation data to plot!")
        return

    plt.figure(figsize=(14, 8))

    #Get evaluation data
    data = processed_data['Evaluation']

    # Calculate mean and standard deviation
    mean_scores = np.mean(data['scores'], axis=0)
    if data['scores'].shape[0] > 1:
        std_scores = np.std(data['scores'], axis=0)
    else:
        std_scores = np.zeros_like(mean_scores)

    # Apply smoothing
    if smoothing > 0 and len(mean_scores) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        mean_scores = np.convolve(mean_scores, kernel, mode='same')
        std_scores = np.convolve(std_scores, kernel, mode='same')

        # Fix edge effects
        if len(mean_scores) > smoothing:
            mean_scores[:smoothing // 2] = mean_scores[smoothing // 2]
            std_scores[:smoothing // 2] = std_scores[smoothing // 2]
            mean_scores[-smoothing // 2:] = mean_scores[-smoothing // 2 - 1]
            std_scores[-smoothing // 2:] = std_scores[-smoothing // 2 - 1]

    # Plot line and shaded area
    x = data['step_points'] / 1000
    plt.plot(x, mean_scores, label="Evaluation", linewidth=2, color='#E41A1C')
    plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='#E41A1C')

    # Beautify chart
    plt.title(f"Basic Actor-Critic Performance on {env_name}", fontsize=16)
    plt.xlabel('Environment Steps (K)', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save chart
    plt.tight_layout()
    plt.savefig(f"results/basic_ac_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Main experiment function
def s4213211_run_experiment(actor_lr=0.001,
                            critic_lr=0.001,
                            gamma=0.99,
                            hidden_size=128,
                            max_env_steps=100000,
                            repetitions=5,
                            num_envs=8,
                            render=False):
    # Setup GPU
    s4213211_setup_gpu()

    # Run AC experiment
    eval_rewards, eval_steps, train_rewards, train_steps = s4213211_run_ac_experiment(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        hidden_size=hidden_size,
        max_env_steps=max_env_steps,
        repetitions=repetitions,
        num_envs=num_envs,
        render=render,
        eval_interval=5000,
        eval_episodes=5
    )

    # Plot learning curves
    s4213211_plot_learning_curves(eval_rewards, eval_steps, train_rewards, train_steps, smoothing=5)

    return eval_rewards, eval_steps, train_rewards, train_steps


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run basic Actor-Critic experiment for CartPole')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Maximum environment steps per run (default: 1000000)')
    parser.add_argument('--reps', type=int, default=5,
                        help='Number of repetitions (default: 5)')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel environments (default: 8)')
    parser.add_argument('--actor_lr', type=float, default=0.00005,
                        help='Actor learning rate (default: 0.00001)')
    parser.add_argument('--critic_lr', type=float, default=0.0001,
                        help='Critic learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden layer size (default: 128)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (will disable vectorization)')

    args = parser.parse_args()

    print("Running basic Actor-Critic training...")
    s4213211_run_experiment(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        hidden_size=args.hidden,
        max_env_steps=args.steps,
        repetitions=args.reps,
        num_envs=args.num_envs,
        render=args.render
    )
    print("Training complete. Results saved to 'results/basic_ac_performance.png'")