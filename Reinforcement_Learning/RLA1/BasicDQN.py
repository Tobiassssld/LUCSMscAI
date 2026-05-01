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


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size

        # params
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        # make the model
        self.model = self._build_model()

    def _build_model(self):
        # simple nn for q-learning
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # choose action (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        # train on one step
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        # get current q
        target = self.model.predict(state, verbose=0)

        # get next q
        if done:
            target[0][action] = reward
        else:
            t = self.model.predict(next_state, verbose=0)
            target[0][action] = reward + self.gamma * np.amax(t[0])

        # do training
        self.model.fit(state, target, epochs=1, verbose=0)

        # decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# run experiment and get results
def s4213211_run_dqn(episodes=1000, max_steps=100000, repetitions=5, render=False):
    all_scores = []
    all_env_steps = []

    for rep in range(repetitions):
        print(f"Starting repetition {rep + 1}/{repetitions}")

        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        scores = []
        env_steps = []
        total_steps = 0

        for e in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])[0]
            done = False
            score = 0

            while not done:
                if render:
                    env.render()

                # get action
                action = agent.act(state)

                # do action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # train agent
                agent.train(state, action, reward, next_state, done)

                state = next_state
                score += reward
                total_steps += 1

                if done:
                    break

                # check steps limit
                if total_steps >= max_steps:
                    break

            scores.append(score)
            env_steps.append(total_steps)

            # show progress
            if e % 10 == 0:
                avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                print(
                    f"Rep {rep + 1}, Episode: {e}/{episodes}, Score: {score}, Avg Score: {avg_score:.2f}, Steps: {total_steps}/{max_steps}, Epsilon: {agent.epsilon:.2f}")

            # break if max steps reached
            if total_steps >= max_steps:
                print(f"Reached maximum steps ({max_steps}) at episode {e}")
                break

        env.close()
        all_scores.append(scores)
        all_env_steps.append(env_steps)

    return all_scores, all_env_steps


# plot the learning curve
def s4213211_plot_curve(scores, env_steps, smoothing=10, title="Learning Curve"):
    plt.figure(figsize=(12, 8))

    # plot each rep
    for i in range(len(scores)):
        scores_array = np.array(scores[i])
        steps_array = np.array(env_steps[i])

        # smooth if needed
        if smoothing > 0 and len(scores_array) > smoothing:
            smooth_scores = []
            for j in range(len(scores_array)):
                start_idx = max(0, j - smoothing + 1)
                smooth_scores.append(np.mean(scores_array[start_idx:j + 1]))
            smooth_scores = np.array(smooth_scores)
        else:
            smooth_scores = scores_array

        # plot this rep
        plt.plot(steps_array, smooth_scores, alpha=0.3, label=f'Rep {i + 1}' if i == 0 else "_nolegend_")

    # calculate average over all reps
    max_steps = max([env_steps[i][-1] for i in range(len(env_steps))])
    step_interval = 1000  # every 1000 steps
    common_steps = np.arange(0, max_steps + step_interval, step_interval)

    interpolated_scores = []
    for i in range(len(scores)):
        # interpolate at common points
        interp_scores = np.interp(common_steps, env_steps[i], scores[i])
        interpolated_scores.append(interp_scores)

    interpolated_scores = np.array(interpolated_scores)

    # get mean and std
    mean_scores = np.mean(interpolated_scores, axis=0)
    std_scores = np.std(interpolated_scores, axis=0)

    # smooth mean and std
    if smoothing > 0 and len(mean_scores) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        mean_scores = np.convolve(mean_scores, kernel, mode='same')
        std_scores = np.convolve(std_scores, kernel, mode='same')

    # plot mean with std area
    plt.plot(common_steps, mean_scores, 'b-', linewidth=2, label='Mean Score')
    plt.fill_between(common_steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

    plt.title(title)
    plt.xlabel('Environment Steps')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    # run the dqn
    print("Running basic DQN...")
    all_scores, all_env_steps = s4213211_run_dqn(episodes=1000, max_steps=100000, repetitions=5)
    s4213211_plot_curve(all_scores, all_env_steps, smoothing=10, title="Basic DQN - CartPole")