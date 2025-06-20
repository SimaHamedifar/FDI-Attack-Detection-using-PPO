import torch as T
import torch.nn as nn 
import torch.optim as optim
import totch.nn.functional as F
import os

class ActorNetwork (nn.Module):
    """
    Defining the actor network of the PPO algorithm.
    """
    def __init__(self, n_actios, input_dims, alpha, fc1_dims = 256, fc2_dims = 256, chkpt_dir="tmp/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, 1)
        self.log_sigma = nn.Parameter(T.zeros(1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Get mean (mu)
        mu = T.sigmoid(self.mu(x))
        
        return mu, self.log_sigma

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
class CriticNetwork(nn.Module):
    """
    Defining the critic network of the PPO algorithm.
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        