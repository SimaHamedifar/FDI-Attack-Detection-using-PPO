# FDI Detection in Power Consumption Using PPO

This repository implements a reinforcement learning-based False Data Injection (FDI) detection system for smart home energy data using the Proximal Policy Optimization (PPO) algorithm. The agent learns to dynamically adjust the detection threshold based on time series patterns, self-attention predictions, and belief scores.

## Overview

- **Goal**: Detect anomalies or FDI attacks in power consumption data.
- **Approach**: Custom OpenAI Gym environment + PPO agent with adaptive thresholding.
- **Features**:
  - Self-attention prediction module.
  - Belief vector integration for attack probability.
  - PPO with custom state vector including power, prediction, error, and temporal features.

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/fdi-detection-ppo.git
   cd fdi-detection-ppo
