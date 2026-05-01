import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import os
import argparse
from gymnasium.vector import AsyncVectorEnv

# make a folder for results
os.makedirs("results", exist_ok=True)


def s4213211_setup_gpu():
    # setup gpu stuff
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")

            if len(gpus) > 0:
                try:
                    mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
                    print("Mixed precision training enabled")
                except:
                    print("Mixed precision training not supported")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")
    else:
        print("No GPU found. Using CPU.")

    print("Devices:", tf.config.list_logical_devices())


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, network_size=[64, 64],
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 update_to_data_ratio=1, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size

        # hyperparams
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.network_size = network_size
        self.update_to_data_ratio = update_to_data_ratio
        self.batch_size = batch_size

        # stuff for training
        self.steps_since_update = 0
        self.temp_buffer = []

        # make the Q-Network
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        # build nn model
        model = Sequential()

        # first layer
        model.add(Dense(self.network_size[0], input_dim=self.state_size, activation='relu'))

        # hidden layers
        for units in self.network_size[1:]:
            model.add(Dense(units, activation='relu'))

        # output layer
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            model.add(Dense(self.action_size, activation='linear', dtype='float32'))
        else:
            model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    @tf.function
    def predict_batch(self, states):
        # predict with model
        return self.model(states)

    @tf.function
    def train_step(self, states, actions, targets):
        # do one step of training
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.action_size)
            current_q = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - current_q))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def act(self, state):
        # choose an action (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        act_values = self.predict_batch(state_tensor)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        # train model on experience
        self.steps_since_update += 1

        self.temp_buffer.append((state, action, reward, next_state, done))

        if self.steps_since_update >= self.update_to_data_ratio:
            if len(self.temp_buffer) >= self.batch_size:
                self._batch_train()
            else:
                state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
                next_state_tensor = tf.convert_to_tensor(np.expand_dims(next_state, axis=0), dtype=tf.float32)

                q_values = self.predict_batch(state_tensor)[0].numpy()

                if done:
                    target = reward
                else:
                    next_q_values = self.predict_batch(next_state_tensor)[0].numpy()
                    target = reward + self.gamma * np.max(next_q_values)

                q_values[action] = target

                self.train_step(
                    state_tensor,
                    tf.convert_to_tensor([action], dtype=tf.int32),
                    tf.convert_to_tensor([target], dtype=tf.float32)
                )

            self.steps_since_update = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _batch_train(self):
        # train with batch of experiences
        batch_size = min(self.batch_size, len(self.temp_buffer))
        minibatch = random.sample(self.temp_buffer, batch_size)

        states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = float(done)

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        current_q_values = self.predict_batch(states_tensor).numpy()
        next_q_values = self.predict_batch(next_states_tensor).numpy()

        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        target_qs = current_q_values.copy()
        for i in range(batch_size):
            target_qs[i, actions[i]] = targets[i]

        self.train_step(
            states_tensor,
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(targets, dtype=tf.float32)
        )

        self.temp_buffer = []


def s4213211_make_env(env_name):
    # helper for vectorized envs
    def _init():
        return gym.make(env_name)

    return _init


def s4213211_run_experiment(learning_rate, network_size, epsilon_decay, update_to_data_ratio,
                   max_env_steps=100000, repetitions=5, env_name='CartPole-v1', batch_size=64,
                   num_envs=4):
    # runs experiment with given hyperparams
    all_scores = []
    all_env_steps = []

    for rep in range(repetitions):
        print(f"Repetition {rep + 1}/{repetitions}")

        if num_envs > 1:
            env = AsyncVectorEnv([s4213211_make_env(env_name) for _ in range(num_envs)])
            temp_env = gym.make(env_name)
            state_size = temp_env.observation_space.shape[0]
            action_size = temp_env.action_space.n
            temp_env.close()
        else:
            env = gym.make(env_name)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            network_size=network_size,
            epsilon_decay=epsilon_decay,
            update_to_data_ratio=update_to_data_ratio,
            batch_size=batch_size
        )

        scores = []
        env_steps_list = []
        total_env_steps = 0

        start_time = time.time()
        episode = 0

        if num_envs > 1:
            # vector env stuff
            states, _ = env.reset()
            env_dones = np.zeros(num_envs, dtype=bool)
            current_scores = np.zeros(num_envs)
            episode_steps = np.zeros(num_envs, dtype=int)

            while total_env_steps < max_env_steps:
                actions = np.array([agent.act(states[i]) for i in range(num_envs)])
                next_states, rewards, terminateds, truncateds, _ = env.step(actions)
                dones = terminateds | truncateds

                for i in range(num_envs):
                    agent.train(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    current_scores[i] += rewards[i]
                    episode_steps[i] += 1

                    if dones[i]:
                        scores.append(current_scores[i])
                        env_steps_list.append(total_env_steps)
                        current_scores[i] = 0
                        episode_steps[i] = 0
                        episode += 1

                states = next_states
                total_env_steps += num_envs

                if episode % 10 == 0 and episode > 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    avg_score = np.mean(scores[-min(10, len(scores)):]) if scores else 0
                    print(f"Episode: {episode}, Avg Score (last 10): {avg_score:.1f}, "
                          f"Steps: {total_env_steps}/{max_env_steps}, "
                          f"Epsilon: {agent.epsilon:.3f}, Speed: {steps_per_sec:.1f} steps/s")

                if total_env_steps >= max_env_steps:
                    break
        else:
            # single env stuff
            while total_env_steps < max_env_steps:
                state, _ = env.reset()
                done = False
                score = 0
                episode_steps = 0

                while not done:
                    action = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    agent.train(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    episode_steps += 1
                    total_env_steps += 1

                    if done or total_env_steps >= max_env_steps:
                        break

                scores.append(score)
                env_steps_list.append(total_env_steps)

                if episode % 10 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode: {episode}, Score: {score}, Steps: {total_env_steps}/{max_env_steps}, "
                          f"Epsilon: {agent.epsilon:.3f}, Speed: {steps_per_sec:.1f} steps/s")

                episode += 1

        all_scores.append(scores)
        all_env_steps.append(env_steps_list)

        print(f"Repetition {rep + 1} done in {time.time() - start_time:.1f} seconds")
        env.close()

    return all_scores, all_env_steps


def s4213211_run_ablation(param_type, max_env_steps=100000, repetitions=5,
                       batch_size=64, num_envs=4):
    # run ablation test for one param
    results = {}
    env_steps_data = {}

    # default values
    base_lr = 0.001
    base_network = [64, 64]
    base_epsilon_decay = 0.995
    base_update_ratio = 1

    if param_type == 'lr':
        # learning rate test
        learning_rates = [0.01, 0.001, 0.0001]
        print("\n=== Learning Rate Ablation ===")
        for lr in learning_rates:
            print(f"Testing learning rate: {lr}")
            scores, env_steps = s4213211_run_experiment(
                learning_rate=lr,
                network_size=base_network,
                epsilon_decay=base_epsilon_decay,
                update_to_data_ratio=base_update_ratio,
                max_env_steps=max_env_steps,
                repetitions=repetitions,
                batch_size=batch_size,
                num_envs=num_envs
            )
            results[f'lr_{lr}'] = scores
            env_steps_data[f'lr_{lr}'] = env_steps

        s4213211_plot_results(results, env_steps_data, 'lr')

    elif param_type == 'network':
        # network size test
        network_sizes = [[24], [64], [128, 64]]
        print("\n=== Network Size Ablation ===")
        for size in network_sizes:
            size_str = '_'.join(map(str, size))
            print(f"Testing network size: {size}")
            scores, env_steps = s4213211_run_experiment(
                learning_rate=base_lr,
                network_size=size,
                epsilon_decay=base_epsilon_decay,
                update_to_data_ratio=base_update_ratio,
                max_env_steps=max_env_steps,
                repetitions=repetitions,
                batch_size=batch_size,
                num_envs=num_envs
            )
            results[f'network_{size_str}'] = scores
            env_steps_data[f'network_{size_str}'] = env_steps

        s4213211_plot_results(results, env_steps_data, 'network')

    elif param_type == 'epsilon':
        # epsilon decay test
        epsilon_decays = [0.99, 0.995, 0.999]
        print("\n=== Epsilon Decay Ablation ===")
        for decay in epsilon_decays:
            print(f"Testing epsilon decay: {decay}")
            scores, env_steps = s4213211_run_experiment(
                learning_rate=base_lr,
                network_size=base_network,
                epsilon_decay=decay,
                update_to_data_ratio=base_update_ratio,
                max_env_steps=max_env_steps,
                repetitions=repetitions,
                batch_size=batch_size,
                num_envs=num_envs
            )
            results[f'epsilon_{decay}'] = scores
            env_steps_data[f'epsilon_{decay}'] = env_steps

        s4213211_plot_results(results, env_steps_data, 'epsilon')

    elif param_type == 'update':
        # update ratio test
        update_ratios = [1, 4, 10]
        print("\n=== Update-to-Data Ratio Ablation ===")
        for ratio in update_ratios:
            print(f"Testing update ratio: {ratio}")
            scores, env_steps = s4213211_run_experiment(
                learning_rate=base_lr,
                network_size=base_network,
                epsilon_decay=base_epsilon_decay,
                update_to_data_ratio=ratio,
                max_env_steps=max_env_steps,
                repetitions=repetitions,
                batch_size=batch_size,
                num_envs=num_envs
            )
            results[f'update_{ratio}'] = scores
            env_steps_data[f'update_{ratio}'] = env_steps

        s4213211_plot_results(results, env_steps_data, 'update')

    elif param_type == 'batch':
        # batch size test
        batch_sizes = [32, 64, 128]
        print("\n=== Batch Size Ablation ===")
        for b_size in batch_sizes:
            print(f"Testing batch size: {b_size}")
            scores, env_steps = s4213211_run_experiment(
                learning_rate=base_lr,
                network_size=base_network,
                epsilon_decay=base_epsilon_decay,
                update_to_data_ratio=base_update_ratio,
                max_env_steps=max_env_steps,
                repetitions=repetitions,
                batch_size=b_size,
                num_envs=num_envs
            )
            results[f'batch_{b_size}'] = scores
            env_steps_data[f'batch_{b_size}'] = env_steps

        s4213211_plot_results(results, env_steps_data, 'batch')

    return results, env_steps_data


def s4213211_process_data(scores, env_steps):
    # process data for plotting
    step_interval = 5000
    max_steps = max([max(steps[-1] for steps in run) for run in env_steps.values()])
    step_points = np.arange(0, max_steps + step_interval, step_interval)

    processed_data = {}

    for config_name in scores.keys():
        interpolated_scores = []

        for rep_idx in range(len(scores[config_name])):
            rep_scores = scores[config_name][rep_idx]
            rep_steps = env_steps[config_name][rep_idx]

            interp_scores = np.interp(
                step_points,
                rep_steps,
                rep_scores,
                left=0,
                right=rep_scores[-1]
            )

            interpolated_scores.append(interp_scores)

        processed_data[config_name] = {
            'step_points': step_points,
            'scores': np.array(interpolated_scores)
        }

    return processed_data


def s4213211_plot_results(scores, env_steps, param_name, env_name='CartPole-v1', smoothing=5):
    # plot the result graphs
    processed_data = s4213211_process_data(scores, env_steps)

    plt.figure(figsize=(14, 8))

    for name, data in processed_data.items():
        # average over repetitions
        mean_scores = np.mean(data['scores'], axis=0)
        std_scores = np.std(data['scores'], axis=0)

        # smoothing
        if smoothing > 0:
            kernel = np.ones(smoothing) / smoothing
            mean_scores = np.convolve(mean_scores, kernel, mode='same')
            std_scores = np.convolve(std_scores, kernel, mode='same')

            # fix edge cases
            mean_scores[:smoothing // 2] = mean_scores[smoothing // 2]
            std_scores[:smoothing // 2] = std_scores[smoothing // 2]
            mean_scores[-smoothing // 2:] = mean_scores[-smoothing // 2 - 1]
            std_scores[-smoothing // 2:] = std_scores[-smoothing // 2 - 1]

        # get param value from name
        param_value = name.split('_')[1]

        # make nice label
        if param_name == 'lr':
            display_name = f"Learning Rate: {param_value}"
        elif param_name == 'network':
            display_name = f"Network Size: {param_value.replace('_', '-')}"
        elif param_name == 'epsilon':
            display_name = f"Epsilon Decay: {param_value}"
        elif param_name == 'update':
            display_name = f"Update Ratio: {param_value}"
        elif param_name == 'batch':
            display_name = f"Batch Size: {param_value}"
        else:
            display_name = f"{param_name}: {param_value}"

        # plot line with shaded area
        x = data['step_points'] / 1000
        plt.plot(x, mean_scores, label=display_name, linewidth=2)
        plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

    # make plot look nice
    plt.title(f"Effect of {param_name.title()} on Learning Performance in {env_name}", fontsize=16)
    plt.xlabel('Environment Steps (K)', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # save figure
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"results/{param_name}_ablation.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # set up GPU
    s4213211_setup_gpu()

    # parse arguments
    parser = argparse.ArgumentParser(description='Run DQN hyperparameter ablation study with GPU optimization')
    parser.add_argument('--param', type=str, choices=['lr', 'network', 'epsilon', 'update', 'batch'],
                        help='Hyperparameter to ablate (lr, network, epsilon, update, batch)')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Maximum environment steps per run (default: 100000)')
    parser.add_argument('--reps', type=int, default=5,
                        help='Number of repetitions per configuration (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--num_envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')

    args = parser.parse_args()

    if args.param:
        # run the ablation study
        print(
            f"Running ablation study for {args.param} with {args.steps} environment steps, "
            f"{args.reps} repetitions, and {args.num_envs} parallel environments")
        start_time = time.time()
        results, _ = s4213211_run_ablation(
            args.param,
            max_env_steps=args.steps,
            repetitions=args.reps,
            batch_size=args.batch_size,
            num_envs=args.num_envs
        )
        elapsed_time = time.time() - start_time
        print(
            f"Ablation study for {args.param} done in {elapsed_time / 60:.2f} minutes. Results saved to 'results' directory.")
    else:
        print("\nPlease specify a parameter to ablate using --param. Options: lr, network, epsilon, update, batch")
        print("Example: python HyperparamatersTest.py --param lr --steps 100000 --reps 5 --batch_size 64 --num_envs 4")