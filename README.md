# FDI Detection in Power Consumption Using PPO

This repository implements a reinforcement learning-based False Data Injection (FDI) attack detection system for smart home energy data using the Proximal Policy Optimization (PPO) algorithm. The agent learns to dynamically adjust the detection threshold based on time series patterns, self-attention predictions, and belief scores.

## Overview

- **Goal**: Detect anomalies or FDI attacks in power consumption data.
- **Approach**: Custom OpenAI Gym environment + PPO agent with adaptive thresholding.
- **Features**:
  - BiLSTM prediction module.
  - Multi-head self-attention-based Belief vector integration for attack probability.
  - PPO implemented using PyTorch with custom state vector including power, prediction, error, and temporal features.

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/SimaHamedifar/fdi-detection-ppo.git
   cd FDI-Attack-Detection-PPO

2. Create a virtual environment and install dependencies:
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt

## How to Run.
1. Dataset: Download the dataset in data/ folder.
2. Train the predictor model.
3. Test the predictor model to make the data for the next steps.
4. Train the Belief network.
5. Test the Belief network to make the data for the next steps.
6. Train the PPO agent.
   python train.py --log True --plot True 
8. Test the PPO agent.
   python test.py --model_path checkpoints/ppo_agent.pth

## Model Structure
•	Predictor.py: BiLSTM-based time series predictor.
• Belief_model.py: Multi-head attention-based belief network.
•	agent.py: PPO actor-critic.
•	environment.py: Custom Gym environment for attack detection.
•	networks.py: Neural networks for actor and critic.

