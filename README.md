# Project Overview

This project implements a Deep Q-Network (DQN) agent to play the Atari Assault game using Stable Baselines3 and Gymnasium. The aim is to train an intelligent agent capable of learning optimal actions in the Assault-v5 environment through trial and error using reinforcement learning.

## Environment

- Game: Atari Assault (ALE/Assault-v5)
- Learning Algorithm: Deep Q-Network (DQN)
- Policy Network: Convolutional Neural Network (CnnPolicy)

## Policy Choice

We selected CnnPolicy over MlpPolicy because:

- The agent receives pixel-based image input, just like a human player, making CNNs more suitable than fully connected MLPs.
- CNNs extract relevant spatial features, improving the agent’s situational awareness and performance.
- According to the original DeepMind DQN paper, CNNs outperform MLPs in Atari environments.

##  Hyperparameter Exploration

Below is a summary of experimental configurations and their performance:

### Experiment Results

| Experiment         | γ (Gamma) | Learning Rate | Epsilon (Start → End) | Notable Observations                                                   |
|--------------------|-----------|----------------|------------------------|------------------------------------------------------------------------|
| CNN_Optimized_1    | 0.98      | 2.5e-4         | 0.01 → 0.01            | Low exploration limited diversity, decent stability, reward = **361.20** |
| CNN_Optimized_2    | 0.99      | 5e-4           | 0.1 → 0.1              | Slightly more exploration helped, reward = **386.40**                  |
| CNN_Optimized_5    | 0.99      | 2.5e-4         | 0.01 → 0.01            | Best result so far, stable + focused learning, reward = **407.40**     |

## Why CNN_Optimized_5 Succeeded

- Balanced Long-Term Learning
  - High gamma (0.99) encouraged planning beyond immediate rewards.
- Conservative Exploration
  - Epsilon was low and constant at 0.01, allowing the agent to refine strategies early and exploit them efficiently.
- Stable Policy Updates
  - Learning rate of 2.5e-4 allowed the agent to learn steadily without erratic updates or overfitting.

## Key Components

### train.py
- Trains the DQN agent using Stable Baselines3.
- Uses Atari wrappers and frame stacking for temporal context.
- Includes callbacks for evaluation and early stopping.
- Logs training data for visualization.
### play.py
- Loads the trained model.
- Evaluates the agent's performance.
- Optionally renders gameplay for inspection.

## Prerequisites

- Python 3.8+
- Libraries:
`pip install stable-baselines3[extra] gymnasium[atari] ale-py opencv-python numpy`

## Training Process

- Environment setup and preprocessing
- DQN agent initialization with optimized hyperparameters
- Training for 100,000 timesteps (more in extended runs)
- Evaluation with EvalCallback and saving best model
- Logging via TensorBoard for reward tracking

## Results Visualization
- The trained agent achieves over 400 mean reward in Assault-v5.
- TensorBoard logs show stable learning curves.
- Agent learns to:
- Fire strategically
- Dodge enemies
- Maximize score within limited exploration range

Additional logs on reward and loss are stored in the tensorboard_logs folder.

## Contributions

Nina Mwangi (Play script and Training, Documentation)
Purity Kihiu (Train Script, Training, Video)
