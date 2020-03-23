import numpy as np
import random
from numpy.linalg import inv

from MP import MP, State
from MRP import MRP
from MDP import Action, MDP

class Environment:
    
    def __init__(self,
                 process: MDP,
                 policy: np.ndarray = np.array([None])):
        
        self.process = process
        self.policy = process.policy if policy.any() == None else policy
    
    def get_random_state(self):
        """
        Returns
        -------
        State : random state of MDP 
        """
        return(self.process.states_list[random.randint(0, self.process.nb_states-1)])
    
    def generate_action(self,
                        state : State):
        """
        Parameters
        ----------
        state : State
            State on which action is sampled
            
        Returns
        -------
        Action : Sample of action performed under process policy
        """
        p = random.uniform(0,1)
        cdf = 0.0
        action_index = -1
        while cdf < p:
            action_index += 1
            cdf += self.policy[state.index][action_index]
        action = self.process.actions_list[action_index]
        return(action)
        
    def generate_return(self,
                        state : State,
                        action : Action):
        """
        Parameters
        ----------
        state : State
            State on which return is sampled
            
        action : Action
            Action on which return is sampled
            
        Returns
        -------
        float : Sample return of state action
        """
        return(action.reward_vec[state.index])
        
    def step(self,
             state : State,
             action : Action):
        """
        Parameters
        ----------
        state : State
            State on which next_state is sampled
            
        action : Action
            Action on which next_state is sampled
            
        Returns
        -------
        next_state : State
            State sampled from @a state and @a action
        """
        p = random.uniform(0,1)
        cdf = 0.0
        next_state_index = -1
        while cdf < p:
            next_state_index += 1
            cdf += action.transitions[state.index,next_state_index]
        next_state = self.process.states_list[next_state_index]
        return(next_state)
    
    def generate_episode(self, 
                         max_len: int = 200,
                         start_state: State = None):
        """
        Parameters
        ----------
        max_len : int (default = 200)
            Maximum length of episode if no termination state is encountered
            
        start_state : State (optional)
            State on which episode starts. Default is random amongst all states
            
        Returns
        -------
        states : list of States
            States visited during episode
            
        actions : list of Actions
            Actions performed during episode
            
        returns : list of float
            Returns perceived during episode
        """    
        states = []
        actions = []
        returns = []
        start_state = start_state if start_state != None else self.get_random_state()
        states.append(start_state)
        finished = False
        len_ep = max_len #random.randint(0,max_len)
        counter = 1
        while not finished:
            
            # ---- Retrieve current State ----
            curr_state = states[-1]
            
            # ---- Generate next Action ----
            action = self.generate_action(curr_state)
            actions.append(action)
            
            # ---- Get Reward associated ----
            returns.append(self.generate_return(curr_state, action))
            
            # ---- Predict next state ----
            next_state = self.step(curr_state, action)
            states.append(next_state)
            
            # ---- Stop Condition ----
            counter += 1
            if next_state.terminal:
                finished = True
            if counter > len_ep:
                finished = True
                
        return(states,actions,returns)
           