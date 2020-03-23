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
                 transitions: np.ndarray  = np.array([None]),
                 reward_vec: np.ndarray = np.array([None])):
        """
        Parameters
        ----------
        name: Hashable type
            Name of the Action
        
        MDP : MDP
            Associated MDP
            
        transitions : Numpy 2D Array
            State transition probability matrix under current action.
            transitions[i,j] == Probability of transitionning from State i to State j.
        
        reward_vec: Numpy 1D Array
            Reward for taking this action at state i.
            
        
        Attributes
        ----------
        Inherited from MRP: states_list, nb_states, transitions, successors, 
                            disc_factor, reward_vec, value_vec
        
        name: hashable type
            Name of the Action
            
        """
        self.MDP = MDP
        super().__init__(MDP.states_list, MDP.disc_fact, transitions, reward_vec)
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
        self.nb_actions = len(actions_list)
        self.policy = policy 
        
        self.MRP = None
        
    
    # ----------------
    # MDP Manipulation
    # ----------------
    
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
        self.nb_actions += 1
        
        
        
    # ------------------------------
    # MRP view of MDP given a policy
    # ------------------------------
    
    def MRP_transition_policy(self, policy):
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
                    policy[start_state.index, action.index] * \
                    action.transitions[start_state.index, end_state.index]            
        return(transition_policy)
    
    def MRP_reward_policy_vec(self, policy):
        """
        Returns
        -------
        Numpy 1D Array
            Policy State reward vector.
        """
        reward_policy_vec = np.zeros(self.nb_states)
        for state in self.states_list:
            for action in self.actions_list:
                reward_policy_vec[state.index] += policy[state.index, action.index] * \
                action.reward_vec[state.index]
        return(reward_policy_vec)
        
    def set_MRP(self, policy = np.array([None])):
        """
        Brief
        ----------
        Sets the MRP policy view of the MDP. 
        """
        policy = self.policy if policy.any() == None else policy
        self.MRP = MRP(self.states_list, self.disc_fact, 
                       self.MRP_transition_policy(policy), 
                       self.MRP_reward_policy_vec(policy))
     
        
        
        
    # -----------------
    # Policy Evaluation 
    # -----------------   
    
    def policy_evaluation(self, policy = np.array([None])):
        """
        Returns
        -------
        Numpy 1D Array
            Policy State Value Function vector.
        """
        policy = self.policy if policy.any() == None else policy
        self.set_MRP(policy)
        v_old = np.zeros(self.nb_states)
        v_new = self.MRP.reward_vec + self.disc_fact * np.dot(self.MRP.transitions,v_old)
        while abs(v_old-v_new).any() > 1e-5:
            v_old = v_new[:]
            v_new = self.MRP.reward_vec + self.disc_fact * np.dot(self.MRP.transitions,v_old)
        return(v_new)
    
    
    
    
    
    # ----------------
    # Policy Iteration
    # ----------------
    
    def improve_policy(self, policy, epsilon = 0.0):
        """
        Returns
        -------
        Numpy 2D Array
            Policy improved through Bellman optimality Equation
        """
        value_vec = self.policy_evaluation(policy)
        new_policy = np.ones((self.nb_states, self.nb_actions))*epsilon/self.nb_actions
        for state in self.states_list:
            temp_value = float("-inf")
            temp_index = -1
            for action in self.actions_list:
                q_value = action.reward_vec[state.index] + \
                           self.disc_fact * np.dot(action.transitions[state.index,:],value_vec)
                if temp_value < q_value:
                    temp_value = q_value                          
                    temp_index = action.index
            new_policy[state.index][temp_index] += 1.0 - epsilon
        return(new_policy)


    def policy_iteration(self, policy = np.array([None])):
        """
        Returns
        -------
        Numpy 2D Array
            Policy assiociated with value function
        """
        old_policy = self.policy if policy.any() == None else policy
        new_policy = self.improve_policy(old_policy)
        while abs(old_policy-new_policy).any() > 1e-5:
            old_policy = new_policy[:,:]
            new_policy = self.improve_policy(old_policy)
        return(new_policy)
    
    
    
    
    # ---------------
    # Value Iteration
    # ---------------
    
    def update_state_value(self, value_vec):
        """
        Parameters
        ----------
        value_vec : Numpy 1D Array
            Current State Value Function vector
            
        Parameters
        ----------
        v_new : Numpy 1D Array
            New State Value Function vector
        """
        v_new = value_vec[:]
        for state in self.states_list:
            temp = float("-inf")
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
        v_old = np.zeros(self.nb_states)
        v_new = self.update_state_value(v_old)
        while abs(v_old-v_new).any() > 1e-5:
            v_old = v_new[:]
            v_new = self.update_state_value(v_old)
        return(v_new)
        
        
    
    def retrieve_policy(self, state_value_vec):
        """
        Returns
        -------
        Numpy 2D Array
            Policy assiociated with value function
        """
        pass
   
    
    def get_Q_policy(self, Q_value : np.ndarray,  epsilon = 0.0):
        """
        Returns
        -------
        Numpy 2D Array
            Policy based on estimated Q_value
        """
        policy = np.ones((self.nb_states, self.nb_actions))*epsilon/self.nb_actions
        for state in self.states_list:
            temp_value = float("-inf")
            temp_index = -1
            for action in self.actions_list:
                q_value = Q_value[state.index][action.index]
                if temp_value < q_value:
                    temp_value = q_value                          
                    temp_index = action.index
            policy[state.index][temp_index] += 1.0 - epsilon
        return(policy)
     
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
    
    # ---- States ----
    states_list = [State(1,5.), State(2,10) , State(3,-7.2)]
    
    # ---- MDP ----
    disc_fact = 0.5
    chain = MDP(states_list, disc_fact)
    
    # ---- Actions ----
    
    # ----
    transitions_a = np.array([[0.3, 0.6, 0.1],
                              [0.1, 0.6, 0.3],
                              [0.5, 0.0, 0.5]])
    reward_a = np.array([5.0, 5.0, 1.0])
    a = Action("a", chain, transitions_a, reward_a)    
    # ----
    
    # ----
    transitions_b = np.array([[0.0, 0.3, 0.7],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.5, 0.5]])
    reward_b = np.array([2.8, 0.0, 10.0])
    b = Action("b", chain, transitions_b, reward_b)
    # ----
    
    # ----
    transitions_c = np.array([[0.2, 0.4, 0.4],
                              [0.2, 0.6, 0.2],
                              [0.0, 0.0, 0.0]])
    reward_c = np.array([-7.2, -7.2, 0.0])
    c = Action("c", chain, transitions_c,reward_c)
    # ----
    
    # --- Add Actions to MDP ----
    chain.add_action(a)
    chain.add_action(b)
    chain.add_action(c)
    
    # ---- Define and set Policy ----
    policy = np.array([[0.4, 0.6, 0.0], 
                       [0.7, 0.0, 0.3],
                       [0.5, 0.5, 0.0]])
    
    chain.set_policy(policy)
    
    # ---- Testing ----  
    
    #chain.set_MRP()
    #print(chain.MRP.reward_vec)
    print(chain.policy_evaluation())
    print(chain.policy_iteration())
# =============================================================================
#     print(chain.value_iteration())
#     chain.policy = chain.policy_iteration()
#     chain.set_MRP()
#     print(chain.MRP.reward_vec)
# =============================================================================
    