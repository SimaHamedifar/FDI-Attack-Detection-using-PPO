# python train.py --log --plot
import pandas as pd
import random
import numpy as np
import torch as T

from environment import AnomalyDetectionEnv
from agent import Agent
from utils.plotting import plot_rewards

import argparse
parser = argparse.ArgumentParser(description="Train PPO for FDI detection")
parser.add_argument("--plot", action="store_true", help="Plot training results")
parser.add_argument("--log", action="store_true", help="Enable logging")
args = parser.parse_args()
if args.log:
    from utils.logger import setup_logger
    logger = setup_logger("ppo_fdi", "logs/training.log")
    logger.info("Logging is enabled.")
else:
    logger = None

data = pd.read_csv ('/data/Belief_Data_Seq_20.csv', parse_dates=True)
data.head()

#%%
train_data = data.iloc[:-1440].copy()

#%%
Predictions = train_data.Prediction
Measurements = train_data.Scaling_1
t_h = train_data.hour_cosin
t_hh = train_data.hour_sin
t_m = train_data.minute_cosin
t_mm = train_data.minute_sin
Beliefs = train_data[['NonAttack_Belief', 'Attack_Belief']]
train_data['Attacked'] = train_data.Targets
train_data['Non_Attacked'] = abs(train_data['Attacked'] -1)
true_labels = train_data[['Non_Attacked', 'Attacked']]
#%%
alpha = random.random() # FNR
beta = random.random()  # FPR
env = AnomalyDetectionEnv(t_h, t_m, Predictions, Measurements, Beliefs, true_labels, alpha, beta)
#%%
N = 500
batch_size = 32
n_epochs = 10
alpha = 0.00001

agent = Agent(n_actios=1, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)

n_games = 1000
n_var = 15

best_score = - 100000
score_hist = []
learn_iters = 0
avg_score = 0
n_steps = 0

Var = random.random()
error = []

for i in range(n_games):
    observation, _ = env.reset()
    terminated = False
    score = 0
    
    while not terminated:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action, Var)
        
        e_t = np.abs(observation_[2]-observation_[3])
        error.append(e_t)
        if n_steps % n_var == 0:
            error_sig = pd.Series(error)
            Var = error_sig.var(ddof=0) 
            error = []
           
            
        n_steps += 1
        score += reward
        agent.remember(observation, val, action, prob, reward, terminated)
        observation = observation_
        
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
            break
        
    score_hist.append(score)
    avg_score = np.mean(score_hist[-100:])
    
    if score > best_score:
        best_score = score
        agent.save_models()

    if logger:
        logger.info(f"Episode {i}, Reward: {score}")
    else:
        print(f"Episode {i}, Reward: {score}")
    
score_hist_s = pd.Series(score_hist, name="Reward")
rolling_mean = score_hist_s.rolling(window=50).mean()
score_hist_s.to_csv("rewards.csv", index_label="Episode")

if args.plot:
    from utils.plotting import plot_rewards
    plot_rewards(score_hist, save_path="plots/reward_plot.png")

T.save(agent.actor.state_dict(), "actor_weights_thre.pth")
T.save(agent.critic.state_dict(), "critic_weights_thre.pth")
