from networks import ActorNetwork
from networks import CriticNetwork
from memory import PPOmemory

import torch as T
import numpy as np

class Agent:
    def __init__(self, n_actios, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95, 
                 policy_clip=0.2, N=2048, batch_size=64, n_epochs=10):
        
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        self.actor = ActorNetwork(n_actios, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOmemory(batch_size)
        
        
    def remember(self, state, vals, action, probs, reward, done):
        self.memory.store_memory(state, vals, action, probs, reward, done)
        
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        print('... loading models ... ')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        mu, log_sigma = self.actor(state)
        sigma = log_sigma.exp()
        
        dist = T.distributions.Normal(mu, sigma)
        
        action = dist.sample()
        
        probs = dist.log_prob(action).item()
        
        action = action.item()
        
        # action = T.clamp(action, 0, 1).item()
        
        value = self.critic(state)
        
        # probs = T.squeeze(dist.log_prob(action)).item()
        # action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        
        return action, probs, value

    def learn(self):        
        for _ in range(self.n_epochs):
            state_arr, vals_arr , action_arr, old_probs_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * vals_arr[k+1] * (1 - int(dones_arr[k])) - vals_arr[k])
                    discount *= self.gamma * self.gae_lambda
                advantage [t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            
            values = T.tensor(vals_arr).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                
                
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                
                mu, log_sigma = self.actor(states)
                sigma = log_sigma.exp()
                dist = T.distributions.Normal(mu, sigma)
                new_probs = dist.log_prob(actions)
                
                # new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp() 
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor.optimizer.zero_grad() # set all gradients to zero. 
                self.critic.optimizer.zero_grad()
                
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()    
        