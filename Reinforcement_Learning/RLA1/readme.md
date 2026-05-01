# Deep Q-Network (DQN) for CartPole Environment

This repository contains implementations of Deep Q-Network (DQN) algorithms for solving the CartPole environment from OpenAI Gymnasium. The code demonstrates different variations of DQN, including enhancements like Target Networks and Experience Replay.

## Important


## Overview

### 1. Basic_DQN.py
A simple implementation of the DQN algorithm that serves as a baseline. This version includes:
- Basic neural network architecture
- Epsilon-greedy exploration
- Simple training loop

Usage:
```
python BasicDQN.py
```

### 2. HyperparametersTest.py
An improved version of DQN with GPU optimization and hyperparameter ablation studies. Features:
- GPU acceleration with TensorFlow
- Hyperparameter tuning capabilities
- Improved training efficiency

Usage:
```
python HyperparametersTest.py --param [lr/network/epsilon/update/batch] --steps 100000 --reps 5 --batch_size 64 --num_envs 4
```

Parameters:
- `--param`: Hyperparameter to ablate (learning rate, network size, epsilon decay, etc.)
- `--steps`: Maximum environment steps per run
- `--reps`: Number of repetitions per configuration
- `--batch_size`: Batch size for training
- `--num_envs`: Number of parallel environments

### 3. DQN_Comparison.py
The most advanced implementation with support for Target Networks and Experience Replay. This version implements:
- Target Networks for stable learning
- Experience Replay for better sample efficiency
- Vectorized environments for parallel training
- GPU optimizations

Usage:
```
python DQN_Comparison.py --mode [compare/single] --steps 1000000 --reps 5 --num_envs 4 --render
```

Parameters:
- `--mode`: Running mode
  - `compare`: Compare all configurations (Naive, TN only, ER only, TN+ER)
  - `single`: Run a single demo with best configuration
- `--steps`: Maximum environment steps per run
- `--reps`: Number of repetitions
- `--num_envs`: Number of parallel environments
- `--render`: Enable rendering (visualization)
- `--batch_size`: Batch size for training

## Key Features

### Target Network (TN)
Provides stability during training by keeping a separate network for generating target values, which is periodically updated with the weights from the main network.

### Experience Replay (ER)
Stores and reuses past experiences to break correlations between consecutive samples and improve data efficiency.

### Hyperparameter Optimization
The code includes tools for ablation studies on various hyperparameters:
- Learning rate
- Network architecture
- Exploration rate decay
- Update-to-data ratio
- Batch size

## Requirements & Environment Setup

### Environment
- Python 3.6+
- **Conda environment is required** for proper dependency management
- CUDA-compatible GPU is required for HyperparametersTest.py and DQN_Comparison.py

### Creating Conda Environment
```bash
# Create conda environment
conda create -n dqn_env python=3.8
conda activate dqn_env

# Install dependencies
conda install tensorflow-gpu  # For GPU support
pip install gymnasium matplotlib numpy
```

### Dependencies
- TensorFlow 2.x (with GPU support highly recommended)
- Gymnasium
- NumPy
- Matplotlib

> **Note**: The Basic_DQN.py can run on CPU, but HyperparametersTest.py and DQN_Comparison.py are optimized for and require GPU acceleration for reasonable performance.

## Results
Results are saved in the `results` directory.