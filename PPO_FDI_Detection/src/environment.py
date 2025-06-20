import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import random


class AnomalyDetectionEnv(gym.Env):
    """
    Custom FDI attack detection system in power consumption measurements. 
    """
    def __init__(self, t_h, t_m, Predictions, Measurements, Beliefs, true_labels, alpha, beta):
        super().__init__()
        self.t_h = t_h
        self.t_m = t_m
        self.Predictions = Predictions
        self.Measurements = Measurements
        self.Beliefs = Beliefs
        self.true_labels = true_labels
        self.current_index = 0
        self.episodic_reward = 0
        self.steps = 0
        self.alpha = alpha
        self.beta = beta
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)  # action: threshold in [0, 1]
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.current_index = 0
        self.current_index = random.randint(0, len(self.t_h)-501)
        self.episodic_reward = 0
        self.steps = 0
        observation = self._get_state()
        info = {'Initial State':observation}
        return observation, info
    
    def _get_state(self):
        t_h = self.t_h[self.current_index]
        t_m = self.t_m[self.current_index]
        Prediction_t = self.Predictions[self.current_index]
        Measurement_t = self.Measurements[self.current_index]
        Belief_0_t = self.Beliefs.iloc[self.current_index, 0]
        Belief_1_t = self.Beliefs.iloc[self.current_index, 1]
        return np.array([t_h, t_m, Prediction_t, Measurement_t, Belief_0_t, Belief_1_t])
    
    def step(self, action, Var):
        self.current_index += 1
        self.steps += 1
        
        observation = self._get_state()
        reward = self._compute_reward(action, Var)
        self.episodic_reward += reward
        
        terminated = self.current_index >= len(self.t_h) - 1
        truncated = False
        
        info = {"True_Label": np.argmax(self.true_labels.iloc[self.current_index])}

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, action, Var):
        if action < Var :
            a = 0
        else:
            a = 1
        # print(f"Action: {a}")
        
        e = abs(self.Measurements[self.current_index] - self.Predictions[self.current_index])
            
        R_e = -1 * (e-Var)**2
        # print(f"R_e: {R_e}")
        
        R_b = self.Beliefs.iloc[self.current_index, int(a)] 
        # print(f"R_b: {R_b}")
        
        R_b_n = -1 * self.Beliefs.iloc[self.current_index, int(1-a)] 
        # print(f"R_b_n: {R_b_n}")
        
        R_u = abs(self.Beliefs.iloc[self.current_index, 0] - 0.5)
             
        if action < Var :  # Non-Attacked
            reward1 =  self.Beliefs.iloc[self.current_index, 0] - 1 * self.Beliefs.iloc[self.current_index, 1] #- self.alpha
            
        if action >= Var: # Attacked
            reward1 = self.Beliefs.iloc[self.current_index, 1] - 1 * self.Beliefs.iloc[self.current_index, 0] #- self.beta
            
        reward = R_b + R_b_n + reward1 + R_e
        return reward
    
    def render(self, mode='human'):
        print(f'Step: {self.current_index}, Current Power: {self.Measurements[self.current_index]}, Predicted Power: {self.Predictions[self.current_index]}')
        
    def close(self):
        pass        