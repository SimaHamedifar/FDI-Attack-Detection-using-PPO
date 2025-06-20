import numpy as np

class PPOmemory:
    """
    memory for storing the experiences of the agent. 
    """
    def __init__(self, batch_size):
        self.states = []
        self.vals = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def generate_batches (self): 
        # shuffle indices and make the chunk of batches according to the batch_size. 
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.vals), np.array(self.actions), np.array(self.probs), np.array(self.rewards), np.array(self.dones),batches
                            
    def store_memory (self, state, vals, action, probs, reward, done):
        self.states.append(state)
        self.vals.append(vals)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)
        
    # a function to clear the memory at the end of the trajectory. 
    def clear_memory (self):
        self.states = []
        self.vals = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []