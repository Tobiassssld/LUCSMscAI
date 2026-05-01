# REINFORCE

## Requirements

- Python 3.8+
- TensorFlow 2.x
- gymnasium
- numpy
- matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```
## Usage

Run the training script:

```bash
python reinforce.py --steps 100000 --reps 5 --num_envs 8
```
You can specify the environment steps, the numbers of repetitions and the number of parallel environments by using the `--num_envs` /  `--steps` / `--reps` parameter.



# Actor-Critic Agent

This repository contains an improved implementation of the Actor-Critic (AC) reinforcement learning algorithm using TensorFlow and Gymnasium. The agent is trained to solve the `CartPole-v1` environment.

## Features

- Actor-Critic architecture with separate networks
- Experience replay and target critic network
- Huber loss and entropy regularization
- Generalized Advantage Estimation (GAE)
- Parallelized environment simulation

## Requirements

- Python 3.8+
- TensorFlow 2.x
- gymnasium
- numpy
- matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```
## Usage

Run the training script:

```bash
python basic_ac_agent.py --steps 100000 --reps 5 --num_envs 8
```
You can specify the number of parallel environments by using the `--num_envs` parameter

## Output
Results are saved in the `results` directory.

# Advantage Actor Critic (A2C) Agent

## Requirements

- Python 3.8+
- PyTorch
- gymnasium
- Numpy

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python a2c.py
```

## Output

Results are saved in the current directory.

# Experiment

## Requirements

- Python 3.8+
- PyTorch
- gymnasium
- Numpy
- Matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

Train 4 agents (DQN, REINFORCE, AC and A2C) in cartpole and plot their learning curves:

```bash
python experiment.py
```

## Output

Results are saved in the current directory.