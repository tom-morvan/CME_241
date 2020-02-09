import numpy as np
from numpy.linalg import inv
#from utils import State
from MP import MP, State

class MRP(MP):
    
    def __init__(self, 
                 states_list: list, 
                 transitions: np.ndarray, 
                 disc_fact: float):
        
        assert(disc_fact >= 0 and disc_fact <= 1),"Discount Factor must be in [0,1]"
        super().__init__(states_list, transitions)
        self.disc_fact = disc_fact
        self.reward_vec = self.get_reward_vec()
        self.value_vec = self.get_value_vec()
    
    
    def get_state_reward(self, 
                         state: State):
        reward = 0
        for successor in self.successors[state]:
            reward += successor[0].reward*successor[1]
        return(reward)
        
    def get_reward_vec(self):
        return(np.array([self.get_state_reward(state) for state in self.states_list]))

            
    def get_value_vec(self):
        return(np.dot(inv(np.eye(self.nb_states - self.disc_fact*self.transistions)),
                           np.transpose(self.reward_vec)))
        
        