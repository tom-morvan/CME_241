import numpy as np
import random
from numpy.linalg import inv

from Environment import Environment
from MP import MP, State
from MRP import MRP
from MDP import Action, MDP



# ---------------
# EPISODE METHODS
# ---------------
    
# ----------
# DEPRECATED
# ----------
# =============================================================================
# def generate_episode(process: MDP, 
#                      policy: np.ndarray = np.array([None]), 
#                      max_len: int = 200):
#     states = []
#     actions = []
#     returns = []
#     policy = process.policy if policy.any() == None else policy
#     start_state = process.states_list[random.randint(0, process.nb_states-1)]
#     states.append(start_state)
#     finished = False
#     len_ep = max_len #random.randint(0,max_len)
#     counter = 1
#     while not finished:
#         
#         # ---- Generate next Action ----
#         p = random.uniform(0,1)
#         curr_state = states[-1]
#         cdf = 0.0
#         action_index = -1
#         while cdf < p:
#             action_index += 1
#             cdf += policy[curr_state.index][action_index]
#         action = process.actions_list[action_index]
#         actions.append(action)
#         
#         # ---- Get Reward associated ----
#         returns.append(action.reward_vec[curr_state.index])
#         
#         # ---- Predict next state ----
#         p = random.uniform(0,1)
#         cdf = 0.0
#         next_state_index = -1
#         while cdf < p:
#             next_state_index += 1
#             cdf += action.transitions[curr_state.index,next_state_index]
#         next_state = process.states_list[next_state_index]
#         states.append(next_state)
#         
#         # ---- Stop Condition ----
#         counter += 1
#         if next_state.terminal:
#             finished = True
#         if counter > len_ep:
#             finished = True
#             
#     return(states,actions,returns)
# =============================================================================

# ------------------
# MONTE CARLO MEHODS
# ------------------
    
# ---- PERDICTION | First Visit ----

def First_Visit_Monte_Carlo(process: MDP,
                            env: Environment,
                            n_iter: int = 5000):
    
    state_value = np.zeros(process.nb_states)
    count_state = np.zeros(process.nb_states)
    for i in range(n_iter):
        states, actions, returns = env.generate_episode()
        G = 0
        for j in range(len(returns)-1, 0, -1):
            G = process.disc_fact*G + returns[j]
            current_state = states[j]
            if current_state not in states[:j-1]:
                count_state[current_state.index] += 1
                state_value[current_state.index] += (G - state_value[current_state.index]) \
                                                    /count_state[current_state.index]
    return state_value

# ---- PERDICTION | Every Visit ----

def Every_Visit_Monte_Carlo(process: MDP,
                            env: Environment,
                            n_iter: int = 5000):

    state_value = np.zeros(process.nb_states)
    count_state = np.zeros(process.nb_states)
    for i in range(n_iter):
        states, actions, returns = env.generate_episode()
        G = 0
        for j in range(len(returns)-1, 0, -1):
            G = process.disc_fact*G + returns[j]
            current_state = states[j]
            count_state[current_state.index] += 1
            state_value[current_state.index] += (G - state_value[current_state.index]) \
                                                /count_state[current_state.index]
    return state_value

# ---- CONTROL | On Policy First Visit ----
    
def First_Visit_Control_Monte_Carlo(process: MDP,
                                    env: Environment,
                                    n_iter: int = 5000,
                                    eps: float = 0.01):
    policy = np.ones((process.nb_states, process.nb_actions))/process.nb_actions
    state_value = np.zeros(process.nb_states)
    count_state = np.zeros(process.nb_states)
    
    for i in range(n_iter):
        states, actions, returns = env.generate_episode()
        G = 0
        for j in range(len(returns)-1, 0, -1):
            G = process.disc_fact*G + returns[j]
            current_state = states[j]
            if current_state not in states[:j-1]:
                count_state[current_state.index] += 1
                state_value[current_state.index] += (G - state_value[current_state.index]) \
                                                    /count_state[current_state.index]
    return state_value



# ------------------
# MONTE CARLO MEHODS
# ------------------
    
# ---- PERDICTION | TD(0) ----
    
def TD_0(process: MDP, 
         env: Environment,
         alpha : float = 0.01,
         n_iter: int = 5000,
         max_ep_len: int = 200):
    
    state_value = np.zeros(process.nb_states)
    for i in range(n_iter):
        current_state = env.get_random_state()
        counter = 1
        ep_finished = False
        while not ep_finished:
            
            # ---- MDP stepping ----
            action = env.generate_action(current_state)
            reward = env.generate_return(current_state,action)
            next_state = env.step(current_state,action)
            
            # ---- Updating state value function ----
            state_value[current_state.index] += alpha * (reward + 
                        process.disc_fact * state_value[next_state.index] - 
                        state_value[current_state.index])
            
            # ---- Stop Condition ----
            counter += 1
            if next_state.terminal:
                ep_finished = True
            if counter > max_ep_len:
                ep_finished = True
    return state_value    


# ---- PERDICTION | Forward TD(lambda) ----

def TD_lambda_forward(process: MDP, 
                      env: Environment,
                      lambda_: float = 0.7,
                      alpha : float = 0.01,
                      n_iter: int = 1000):
    
    state_value = np.zeros(process.nb_states)
    for i in range(n_iter):
        states, actions, returns = env.generate_episode(200)
        ep_len = len(returns)
        for t in range(0, ep_len):
            current_state = states[t]
            cumulated_return = 0
            G_t = 0
            for n in range (t, ep_len):
                cumulated_return +=  process.disc_fact**(n-t) * returns[n]
                G_n = cumulated_return + process.disc_fact**(n+1-t) * state_value[states[n+1].index]
                G_t += lambda_**(n-t) * G_n
            G_t *= (1-lambda_)
            
            state_value[current_state.index] += alpha * (G_t - state_value[current_state.index])
            
    return state_value    


# ---- PERDICTION | Backward TD(lambda) ----

def TD_lambda_backward(process: MDP, 
                      env: Environment,
                      lambda_: float = 0.7,
                      alpha : float = 0.01,
                      n_iter: int = 5000,
                      max_ep_len: int = 200):
    
    state_value = np.zeros(process.nb_states)
    for i in range(n_iter):
        E_t = np.zeros(process.nb_states)
        current_state = env.get_random_state()
        counter = 1
        ep_finished = False
        while not ep_finished:
            
            # ---- MDP stepping ----
            action = env.generate_action(current_state)
            reward = env.generate_return(current_state,action)
            next_state = env.step(current_state,action)
            
            # ---- Updating Eligibility Trace ----
            E_t *= process.disc_fact * lambda_
            E_t[current_state.index] += 1
            
            # ---- Updating state value function ----
            error = reward + process.disc_fact*state_value[next_state.index] -\
                    state_value[current_state.index]
            state_value += alpha * error * E_t
            
            # ---- Stop Condition ----
            counter += 1
            if next_state.terminal:
                ep_finished = True
            if counter > max_ep_len:
                ep_finished = True
            
    return state_value   



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
    
    
    # ------------
    # MDP CREATION
    # ------------
    
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
    
    
    # -----------
    # ENVIRONMENT
    # -----------
    
    env = Environment(chain,policy)
        
    
    # -------
    # TESTING
    # -------
# =============================================================================
#     #First visit Monte Carlo:
#     mc = First_Visit_Monte_Carlo(chain)
#     print(mc)
# =============================================================================
    
# =============================================================================
#     #Every visit Monte Carlo
#     mc = Every_Visit_Monte_Carlo(chain, env)
#     print(mc)
# =============================================================================
    
# =============================================================================
#     # TD_0
#     TD = TD_0(chain, env)
#     print(TD)
# =============================================================================
    
# =============================================================================
#     # TD_lambda Forward
#     TD = TD_lambda_forward(chain, env)
#     print(TD)
# =============================================================================
    
# =============================================================================
#     # TD_lambda Backward
#     TD = TD_lambda_backward(chain, env)
#     print(TD)
# =============================================================================

