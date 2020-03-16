import numpy as np
from numpy.linalg import inv

from MP import MP, State
from MRP import MRP

#Forward decleration
class MDP:
    pass

class Action(MRP):
    
    #Static Indexing
    _mtx_index = 0
    
    def __init__(self, 
                 name,
                 MDP: MDP,
                 transitions: np.ndarray):
        """
        Parameters
        ----------
        name: Hashable type
            Name of the Action
        
        states_list : list of State
            Collection of all the states in the MRP.
            
        transitions : Numpy 2D Array
            State transition probability matrix.
            transitions[i,j] == Probability of transitionning from State i to State j.
        
        disc_fact : float
            Reward Discount Factor
            
        
        Attributes
        ----------
        Inherited from MRP: states_list, nb_states, transitions, successors, 
                            disc_factor, reward_vec, value_vec
        
        name: hashable type
            Name of the Action
            
        """
        self.MDP = MDP
        super().__init__(MDP.states_list, transitions, MDP.disc_fact)
        self.index = Action._mtx_index
        self.name = name
        
        Action._mtx_index += 1
    
    #Overload equality method
    def __eq__(self, 
               action):
        return (self.name == action.name)
    
    def __hash__(self):
        return hash((self.name))
    
    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
    
    def __repr__(self):
        return(str(self.name))


class MDP:
    
    def __init__(self, 
                 states_list: list,  
                 disc_fact: float,
                 actions_list: list = [],
                 policy: np.ndarray = None):
        """
        Parameters
        ----------
        states_list : list of State
            Collection of all the states in the MDP.
        
        disc_fact : float
            Reward Discount Factor
            
        actions_list : list of Action (default empty)
            Collection of all the actions in the MDP.
        
        policy: Numpy 2D Array (default None)
            Probability policy matrix
            policy[i,j] == Probability of observing Action j at State i
            
        
        Attributes
        ----------
        states_list : list of State
            Collection of all the states in the MDP.
        
        nb_states : int
            Number of states in the MDP.
            
        disc_fact : float
            Reward Discount Factor
            
        actions_list : list of Action
            Collection of all the actions in the MDP.
            
        reward_vec : Numpy 1D Array
            Reward function represented as a vector
                        
        policy : Numpy 2D Array
            Policy probability matrix
            policy[i,j] == Probability of observing Action j at State i
        
        MRP : MRP
            MRP representation of MDP
        """
        
        assert(disc_fact >= 0 and disc_fact <= 1),"Discount Factor must be in [0,1]"
        self.states_list = states_list
        self.nb_states= len(states_list)
        self.disc_fact = disc_fact        
        self.actions_list = actions_list
        self.policy = policy 
        
        self.MRP = None
    
    
    def set_policy(self, policy: np.ndarray):
        """
        Parameters
        ----------
        policy: Numpy 2D Array (default None)
            Probability policy matrix
            policy[i,j] == Probability of observing Action j at State i
        """
        self.policy = policy
        
    def add_action(self, action: Action):
        """
        Parameters
        ----------
        action: Action
            action to add to the MDP
        """
        self.actions_list.append(action)
        
    def MP_transition_policy(self):
        """
        Returns
        -------
        Numpy 2D Array
            Policy State transition probability matrix.
        """
        transition_policy = np.zeros((self.nb_states, self.nb_states))
        for start_state in self.states_list:
            for end_state in self.states_list:        
                for action in self.actions_list:
                    transition_policy[start_state.index, end_state.index] +=  \
                    self.policy[start_state.index, action.index] * \
                    action.transitions[start_state.index, end_state.index]
        return(transition_policy)
    
    def MP_reward_policy_vec(self):
        """
        Returns
        -------
        Numpy 1D Array
            Policy State reward vector.
        """
        reward_policy_vec = np.zeros(self.nb_states)
        for state in self.states_list:
            for action in self.actions_list:
                reward_policy_vec[state.index] += self.policy[state.index, action.index] * \
                action.reward_vec[state.index]
        return(reward_policy_vec)
        
    def set_MRP(self):
        """
        Brief
        ----------
        Sets the MRP policy view of the MDP. 
        """
        self.MRP = MRP(states_list, self.MP_transition_policy(), self.disc_fact,
                       self.MP_reward_policy_vec())
        
    def policy_evaluation(self):
        """
        Returns
        -------
        Numpy 1D Array
            Policy State Value Function vector.
        """
        v_old = self.MRP.value_vec
        v_new = self.MRP.reward_vec + self.disc_fact * np.dot(self.MRP.transitions,v_old)
        while sum(abs(v_old-v_new)) > 1e-3:
            v_old = v_new
            v_new = self.MRP.reward_vec + self.disc_fact * np.dot(self.MRP.transitions,v_old)
        return(v_new)
    
    def update_state_value(self, value_vec):
        """
        Parameters
        ----------
        value_vec : Numpy 1D Array
            Current State Value Function vector
        """
        v_new = value_vec[:]
        for state in self.states_list:
            temp = 0
            for action in self.actions_list:
                temp = max(temp, 
                           action.reward_vec[state.index] + \
                           self.disc_fact * np.dot(action.transitions[state.index,:],value_vec))
            v_new[state.index] = temp
        return(v_new)
    
    def value_iteration(self):
        """
        Returns
        -------
        Numpy 1D Array
            Optimal State Value Function vector.
        """
        v_old = self.MRP.value_vec
        v_new = self.update_state_value(v_old)
        while sum(abs(v_old-v_new)) > 1e-3:
            v_old = v_new
            v_new = self.update_state_value(v_old)
        return(v_new)
   

     
if __name__ == "__main__":
    
# =============================================================================
#     policy_data = {
# 
#             1: {'a': 0.4, 'b': 0.6},
#             2: {'a': 0.7, 'c': 0.3},
#             3: {'a' : 0.5, 'b': 0.5}
#     }
# 
#     data = {
#         1: {
#             'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
#             'b': ({2: 0.3, 3: 0.7}, 2.8),
#             'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
#         },
#         2: {
#             'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 5.0),
#             'c': ({1: 0.2, 2: 0.6, 3: 0.2}, -7.2)
#         },
#         3: {
#             'a': ({1:0.5, 3: 0.5}, 1.0),
#             'b': ({2: 0.5, 3:0.5}, 10)
#         }
#     }
# =============================================================================
    
    
    states_list = [State(1,5.), State(2,10) , State(3,-7.2)]
    disc_fact = 1
    chain = MDP(states_list, disc_fact)
    
    transitions_a = np.array([[0.3, 0.6, 0.1],
                               [0.1, 0.6, 0.3],
                               [0.5, 0 , 0.5]])
    a = Action("a", chain, transitions_a)    
    
    transitions_b = np.array([[0, 0.3, 0.7],
                               [0, 0, 0],
                               [0, 0.5 , 0.5]])
    b = Action("b", chain, transitions_b)
    
    transitions_c = np.array([[0.2, 0.4, 0.4],
                               [0.2, 0.6, 0.2],
                               [0, 0 , 0]])
    c = Action("c", chain, transitions_c)
    
    chain.add_action(a)
    chain.add_action(b)
    chain.add_action(c)
    
    policy = np.array([[0.4, 0.6 , 0], 
                       [0, 0.3, 0.7],
                       [0.5, 0.5, 0]])
    
    chain.set_policy(policy)
    chain.set_MRP()
    
    print(chain.MRP.reward_vec)
    print(chain.policy_evaluation())
    print(chain.value_iteration())
    
    