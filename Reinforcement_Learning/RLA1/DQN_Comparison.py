import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
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

# make results folder
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


class ReplayBuffer:
    def __init__(self, capacity=10000, state_size=4):
        # initialize replay buffer
        self.capacity = capacity
        self.state_size = state_size

        # preallocate memory
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        # add experience to buffer
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # sample a batch
        indices = np.random.choice(self.size, min(self.size, batch_size), replace=False)

        states = tf.convert_to_tensor(self.states[indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions[indices], dtype=tf.int32)
        rewards = tf.convert_to_tensor(self.rewards[indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_states[indices], dtype=tf.float32)
        dones = tf.convert_to_tensor(self.dones[indices], dtype=tf.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        # get buffer size
        return self.size


class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001,
                 network_size=[128],
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.999,
                 epsilon_min=0.01,
                 update_to_data_ratio=1,
                 use_target_network=False,
                 target_update_freq=10,
                 use_experience_replay=False,
                 memory_size=10000,
                 batch_size=64):
        # init DQN agent
        self.state_size = state_size
        self.action_size = action_size

        # params
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.network_size = network_size
        self.update_to_data_ratio = update_to_data_ratio
        self.steps_since_update = 0

        # features config
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.target_update_counter = 0

        self.use_experience_replay = use_experience_replay
        self.batch_size = batch_size

        # optimizer
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # main Q-network
        self.model = self._build_model()

        # target network if enabled
        if self.use_target_network:
            self.target_model = self._build_model()
            self.update_target_network()

        # replay buffer if enabled
        if self.use_experience_replay:
            self.memory = ReplayBuffer(capacity=memory_size, state_size=state_size)

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

    def update_target_network(self):
        # copy weights to target network
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def predict_q_values(self, state):
        # predict q values with main network
        return self.model(state)

    @tf.function
    def predict_target_q_values(self, state):
        # predict q values with target network
        return self.target_model(state)

    @tf.function
    def train_step(self, states, actions, targets):
        # do one training step
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.action_size)
            current_q = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - current_q))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def act(self, state):
        # choose action (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        act_values = self.predict_q_values(state_tensor)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        # train the agent
        self.steps_since_update += 1

        # store experience in replay buffer if using ER
        if self.use_experience_replay:
            self.memory.add(state, action, reward, next_state, done)

            # wait for enough samples
            if len(self.memory) < self.batch_size:
                return

        # only update based on ratio
        if self.steps_since_update >= self.update_to_data_ratio:
            if self.use_experience_replay:
                # sample batch
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

                # compute targets
                if self.use_target_network:
                    next_q_values = self.predict_target_q_values(next_states)
                else:
                    next_q_values = self.predict_q_values(next_states)

                max_next_q = tf.reduce_max(next_q_values, axis=1)
                targets = rewards + (1.0 - dones) * self.gamma * max_next_q

                # train the network
                self.train_step(states, actions, targets)

            else:
                # single sample training
                state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
                next_state_tensor = tf.convert_to_tensor(np.expand_dims(next_state, axis=0), dtype=tf.float32)
                action_tensor = tf.convert_to_tensor([action], dtype=tf.int32)

                if self.use_target_network:
                    next_q_values = self.predict_target_q_values(next_state_tensor)
                else:
                    next_q_values = self.predict_q_values(next_state_tensor)

                max_next_q = tf.reduce_max(next_q_values, axis=1)
                target = reward + (1.0 - float(done)) * self.gamma * max_next_q.numpy()[0]

                self.train_step(state_tensor, action_tensor, tf.convert_to_tensor([target], dtype=tf.float32))

            # update target network if using TN
            if self.use_target_network:
                self.target_update_counter += 1
                if self.target_update_counter >= self.target_update_freq:
                    self.update_target_network()
                    self.target_update_counter = 0

            # reset counter
            self.steps_since_update = 0

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def s4213211_make_env(env_name):
    # helper for vectorized envs
    def _init():
        return gym.make(env_name)

    return _init


def s4213211_run_experiment(use_target_network=False,
                   use_experience_replay=False,
                   learning_rate=0.001,
                   network_size=[128],
                   gamma=0.99,
                   epsilon_decay=0.999,
                   update_to_data_ratio=1,
                   target_update_freq=10,
                   memory_size=10000,
                   batch_size=64,
                   max_env_steps=100000,
                   repetitions=5,
                   env_name='CartPole-v1',
                   num_envs=4,
                   render=False):
    # run experiment with given config
    config_name = "Naive"
    if use_target_network and use_experience_replay:
        config_name = "TN + ER"
    elif use_target_network:
        config_name = "Target Network Only"
    elif use_experience_replay:
        config_name = "Experience Replay Only"

    print(f"\n=== Running {config_name} Configuration ===")
    print(f"Learning rate: {learning_rate}, Network size: {network_size}")
    print(f"Epsilon decay: {epsilon_decay}, Update ratio: {update_to_data_ratio}")
    if use_target_network:
        print(f"Target update frequency: {target_update_freq}")
    if use_experience_replay:
        print(f"Memory size: {memory_size}, Batch size: {batch_size}")
    print(f"Using {num_envs} parallel environments")

    all_scores = []
    all_env_steps = []
    total_train_time = 0

    for rep in range(repetitions):
        print(f"Repetition {rep + 1}/{repetitions}")

        # setup env
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

        # make agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            network_size=network_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            update_to_data_ratio=update_to_data_ratio,
            use_target_network=use_target_network,
            target_update_freq=target_update_freq,
            use_experience_replay=use_experience_replay,
            memory_size=memory_size,
            batch_size=batch_size
        )

        scores = []
        env_steps_list = []
        total_env_steps = 0
        episode = 0
        start_time = time.time()

        # vectorized environments
        if num_envs > 1 and not render:
            states, _ = env.reset()
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
            # single environment
            while total_env_steps < max_env_steps:
                state, _ = env.reset()
                done = False
                score = 0
                episode_steps = 0

                while not done:
                    # get action
                    action = agent.act(state)

                    # do action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # train agent
                    agent.train(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    episode_steps += 1
                    total_env_steps += 1

                    if done or total_env_steps >= max_env_steps:
                        break

                scores.append(score)
                env_steps_list.append(total_env_steps)
                episode += 1

                if episode % 10 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode: {episode}, Score: {score}, Steps: {total_env_steps}/{max_env_steps}, "
                          f"Epsilon: {agent.epsilon:.3f}, Speed: {steps_per_sec:.1f} steps/s")

        episode_time = time.time() - start_time
        total_train_time += episode_time
        all_scores.append(scores)
        all_env_steps.append(env_steps_list)
        print(f"Repetition {rep + 1} done in {episode_time:.2f} seconds")

    env.close()
    print(f"Total training time: {total_train_time:.2f} seconds")
    return config_name, all_scores, all_env_steps


def s4213211_compare_exps(learning_rate=0.001,
                               network_size=[128],
                               epsilon_decay=0.999,
                               update_ratio=1,
                               max_env_steps=100000,
                               repetitions=5,
                               num_envs=4):
    # run all 4 configurations
    print(
        f"Running experiments with {max_env_steps} steps, {repetitions} reps, and {num_envs} parallel envs")
    print(
        f"Using params: LR={learning_rate}, Network={network_size}, Epsilon decay={epsilon_decay}, Update ratio={update_ratio}")

    results = {}
    env_steps_data = {}

    # Config 1: Naive (no TN, no ER)
    config1, scores1, steps1 = s4213211_run_experiment(
        use_target_network=False,
        use_experience_replay=False,
        learning_rate=learning_rate,
        network_size=network_size,
        epsilon_decay=epsilon_decay,
        update_to_data_ratio=update_ratio,
        max_env_steps=max_env_steps,
        repetitions=repetitions,
        num_envs=num_envs
    )
    results[config1] = scores1
    env_steps_data[config1] = steps1

    # Config 2: Target Network only
    config2, scores2, steps2 = s4213211_run_experiment(
        use_target_network=True,
        use_experience_replay=False,
        learning_rate=learning_rate,
        network_size=network_size,
        epsilon_decay=epsilon_decay,
        update_to_data_ratio=update_ratio,
        target_update_freq=10,
        max_env_steps=max_env_steps,
        repetitions=repetitions,
        num_envs=num_envs
    )
    results[config2] = scores2
    env_steps_data[config2] = steps2

    # Config 3: Experience Replay only
    config3, scores3, steps3 = s4213211_run_experiment(
        use_target_network=False,
        use_experience_replay=True,
        learning_rate=learning_rate,
        network_size=network_size,
        epsilon_decay=epsilon_decay,
        update_to_data_ratio=update_ratio,
        memory_size=10000,
        batch_size=64,
        max_env_steps=max_env_steps,
        repetitions=repetitions,
        num_envs=num_envs
    )
    results[config3] = scores3
    env_steps_data[config3] = steps3

    # Config 4: Both TN and ER
    config4, scores4, steps4 = s4213211_run_experiment(
        use_target_network=True,
        use_experience_replay=True,
        learning_rate=learning_rate,
        network_size=network_size,
        epsilon_decay=epsilon_decay,
        update_to_data_ratio=update_ratio,
        target_update_freq=10,
        memory_size=10000,
        batch_size=64,
        max_env_steps=max_env_steps,
        repetitions=repetitions,
        num_envs=num_envs
    )
    results[config4] = scores4
    env_steps_data[config4] = steps4

    # plot the comparison
    s4213211_plot_comparison(results, env_steps_data, smoothing=5)

    return results, env_steps_data


def s4213211_process_plot_data(scores, env_steps):
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


def s4213211_plot_comparison(results, env_steps_data, env_name='CartPole-v1', smoothing=5):
    # plot the comparison graph
    processed_data = s4213211_process_plot_data(results, env_steps_data)

    plt.figure(figsize=(14, 8))

    # colors for each config
    config_colors = {
        'Naive': '#E41A1C',
        'Target Network Only': '#377EB8',
        'Experience Replay Only': '#4DAF4A',
        'TN + ER': '#984EA3'
    }

    for name, data in processed_data.items():
        # get mean and std
        mean_scores = np.mean(data['scores'], axis=0)
        std_scores = np.std(data['scores'], axis=0)

        # smooth if needed
        if smoothing > 0:
            kernel = np.ones(smoothing) / smoothing
            mean_scores = np.convolve(mean_scores, kernel, mode='same')
            std_scores = np.convolve(std_scores, kernel, mode='same')

            # fix edge cases
            mean_scores[:smoothing // 2] = mean_scores[smoothing // 2]
            std_scores[:smoothing // 2] = std_scores[smoothing // 2]
            mean_scores[-smoothing // 2:] = mean_scores[-smoothing // 2 - 1]
            std_scores[-smoothing // 2:] = std_scores[-smoothing // 2 - 1]

        # plot line with shaded area
        x = data['step_points'] / 1000
        color = config_colors.get(name, None)
        plt.plot(x, mean_scores, label=name, linewidth=2, color=color)
        plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color=color)

    # make plot look nice
    plt.title(f"Comparison of DQN Configurations on {env_name}", fontsize=16)
    plt.xlabel('Environment Steps (K)', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # save figure
    plt.tight_layout()
    plt.savefig(f"results/configuration_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def s4213211_demo_run(learning_rate=0.001,
                    network_size=[128],
                    epsilon_decay=0.999,
                    update_ratio=1,
                    use_target_network=True,
                    use_experience_replay=True,
                    max_episodes=10):
    # run single demo with rendering
    s4213211_run_experiment(
        use_target_network=use_target_network,
        use_experience_replay=use_experience_replay,
        learning_rate=learning_rate,
        network_size=network_size,
        epsilon_decay=epsilon_decay,
        update_to_data_ratio=update_ratio,
        max_env_steps=100000,
        repetitions=1,
        num_envs=1,
        render=True
    )


if __name__ == "__main__":
    # setup gpu
    s4213211_setup_gpu()

    # parse arguments
    parser = argparse.ArgumentParser(description='Run DQN experiments for CartPole')
    parser.add_argument('--mode', type=str, choices=['compare', 'single'], default='compare',
                        help='Mode: compare (all configs) or single (best config)')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Maximum environment steps per run (default: 100000)')
    parser.add_argument('--reps', type=int, default=5,
                        help='Number of repetitions (default: 5)')
    parser.add_argument('--num_envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (only for single mode)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 64)')

    args = parser.parse_args()

    # best params from ablation
    best_lr = 0.001
    best_network = [128]
    best_epsilon_decay = 0.999
    best_update_ratio = 1

    if args.mode == 'compare':
        print("Running all four configs...")
        results, _ = s4213211_compare_exps(
            learning_rate=best_lr,
            network_size=best_network,
            epsilon_decay=best_epsilon_decay,
            update_ratio=best_update_ratio,
            max_env_steps=args.steps,
            repetitions=args.reps,
            num_envs=args.num_envs
        )
        print("Experiment complete. Results saved to 'results/configuration_comparison.png'")

    elif args.mode == 'single':
        print("Running with best params...")
        s4213211_demo_run(
            learning_rate=best_lr,
            network_size=best_network,
            epsilon_decay=best_epsilon_decay,
            update_ratio=best_update_ratio,
            use_target_network=True,
            use_experience_replay=True,
            max_episodes=10 if args.render else 100
        )
        print("Complete.")