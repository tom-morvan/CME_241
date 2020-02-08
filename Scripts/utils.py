import numpy as np

class State:
    
    def __init__(self, index: int, reward: float = None):
        self.index = index
        self.reward = reward
        
    