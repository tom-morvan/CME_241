import numpy as np
#from numpy.linalg import eig

class State:
    
    def __init__(self, 
                 index: int, 
                 reward: float = None):
        self.index = index
        self.reward = reward
        

class MP:
    
    def __init__(self, 
                 states_list: list, 
                 transitions: np.ndarray):
        self.states_list = states_list
        self.nb_states = len(states_list)
        self.transistions = transitions
        self.successors = self.get_successors()
        
    
    def get_sink_states(self):
        sink_states = []
        for i in range(0, self.nb_states):
            if self.transitions[i,i] == 1:
                sink_states.append(self.states_list[i])        
        return(sink_states)
        
    def get_successors(self):
        successors = dict()
        for state in self.states_list:
            if (state not in successors.keys()):
                successors[state] = []
            for j in range(0, self.nb_states):
                prob = self.transistions[state.index][j]
                if prob != 0:
                    successors[state].append((self.states_list[j], prob))
        return(successors)

        
    