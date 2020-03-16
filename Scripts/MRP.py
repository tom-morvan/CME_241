import numpy as np
from numpy.linalg import inv

from MP import MP, State

class MRP(MP):
        
    def __init__(self,
                 states_list: list, 
                 disc_fact: float,
                 transitions: np.ndarray  = np.array([None]), 
                 reward_vec: np.ndarray = np.array([None])):
        """
        Parameters
        ----------
        states_list : list of State
            Collection of all the states in the MRP.
            
        transitions : Numpy 2D Array
            State transition probability matrix.
            transitions[i,j] == Probability of transitionning from State i to State j.
        
        disc_fact : float
            Reward Discount Factor
        
        reward_vec : Numpy 1D Array (optional Argument)
            Reward function represented as a vector
            
        
        Attributes
        ----------
        Inherited from MP: states_list, nb_states, transitions, successors
        
        disc_fact : float
            Reward Discount Factor
            
        reward_vec : Numpy 1D Array
            Reward function represented as a vector
            
        value_vec : Numpy 1D Array
            Value function represented as a vector
        """
        assert(disc_fact >= 0 and disc_fact <= 1),"Discount Factor must be in [0,1]"
        super().__init__(states_list, transitions)
        self.disc_fact = disc_fact
        self.reward_vec = reward_vec 
        self.value_vec = np.array([None])
    
    
    def state_reward(self, 
                     state: State):
        """
        Parameters
        ----------
        state : State
            State on which reward function is computed
            
        Returns
        -------
        float
            Value of State reward function
        """
        reward = 0
        for successor in self.successors[state].keys():
            reward += successor.reward*self.successors[state][successor]
        return(reward)
        
    def set_state_reward(self):
        """            
        Sets
        ----
        Numpy 1D Array of float
            State reward function
        """
        self.reward_vec = np.array([self.state_reward(state) for state in self.states_list])

            
    def set_state_value_function(self):
        """            
        Sets
        ----
        Numpy 1D Array of float
            State Value Function
        """
        self.value_vec = np.dot(inv(np.eye(self.nb_states) - self.disc_fact*self.transitions),
                           self.reward_vec)
        
        
if __name__ == "__main__":
    
    
# =============================================================================
#     data = {
#         1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
#         2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
#         3: ({3: 1.0}, 0.0)
#     }
# =============================================================================
    
    state_list = [State(1, 7), State(2, 10), State(3, 0)]
    transitions = np.array([[0.6, 0.3, 0.1],
                            [0.1, 0.2, 0.7],
                            [0., 0., 1]])
    disc_fact = 1
    
    chain = MRP(state_list, disc_fact, transitions)
    print(chain.successors)
    chain.set_state_reward()
    print(chain.reward_vec)
    #chain.set_state_value_function()
    #print(chain.value_vec)
    
    