import numpy as np
from numpy.linalg import inv
from MP import State
import MRP

class Action(MRP):
    
    def __init__(self, 
                 index: int,
                 states_list: list,
                 state_transitions: np.ndarray,
                 disc_fact: float):
        
        super().__init__(states_list, state_transitions, disc_fact)
        self.index = index
    
# =============================================================================
#     def get_successors(self):
#         successors = dict()
#         for state in self.states_list:
#             if (state not in successors.keys()):
#                 successors[state] = []
#             for j in range(0, self.nb_states):
#                 prob = self.transistions[state.index][j]
#                 if prob != 0:
#                     successors[state].append((self.states_list[j], prob))
#         return(successors)
# =============================================================================
    
# =============================================================================
#     def get_state_reward(self, 
#                          state: State):
#         reward = 0
#         for successor in self.successors[state]:
#             reward += successor[0].reward*successor[1]
#         return(reward)
# =============================================================================
        
# =============================================================================
#     def get_reward_vec(self):
#         return(np.array([self.get_state_reward(state) for state in self.states_list]))
# =============================================================================

            
# =============================================================================
#     def get_value_vec(self):
#         return(np.dot(inv(np.eye(self.nb_states - self.disc_fact*self.transistions)),
#                            self.reward_vec))
# =============================================================================

class MDP:
    
    def __init__(self, 
                 states_list: list,  
                 disc_fact: float,
                 actions_list: list,
                 policy: np.ndarray):
        
        assert(disc_fact >= 0 and disc_fact <= 1),"Discount Factor must be in [0,1]"
        self.states_list = states_list
        self.nb_states= len(states_list)
        self.disc_fact = disc_fact        
        self.actions_list = actions_list
        
        self.policy = policy 
        
        self.transition_policy = self.get_transition_policy()
        self.reward_policy_vec = self.get_reward_policy_vec()
        
    
    
    def get_transition_policy(self):
        transition_policy = np.zeros((self.nb_states, self.nb_states))
        for start_state in self.states_list:
            for end_state in self.states_list:        
                for action in self.actions_list:
                    transition_policy[start_state.index, end_state.index] +=  \
                    self.policy[start_state.index, action.index] * \
                    action.state_transitions[start_state.index, end_state.index]
        return(transition_policy)
    
    def get_reward_policy_vec(self):
        reward_policy_vec = np.zeros(self.nb_states)
        for state in self.states_list:
            for action in self.actions_list:
                reward_policy_vec[state.index] += self.policy[state.index, action.index] * \
                action.reward_vec[state.index]
        return(reward_policy_vec)
        
    def get_value_policy_vec(self):
        return(np.dot(inv(np.eye(self.nb_states - self.disc_fact*self.transistion_policy)),
                           np.transpose(self.reward_policy_vec)))
    
    
        
        
        
        
        
        