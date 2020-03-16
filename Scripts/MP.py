import numpy as np
from numpy.linalg import eig, inv


def diag(A: np.ndarray):
    
    eig_val, eig_vec = eig(A)
    return(np.dot(np.dot(inv(eig_vec), A),eig_vec))

class State:
    
    #Static Indexing
    _mtx_index = 0
    
    def __init__(self, 
                 name, 
                 reward: float = None,
                 terminal : bool = False):
        self.name = name
        self.index = State._mtx_index
        self.reward = reward
        self.terminal = terminal
        State._mtx_index += 1
    
    #Overload equality method
    def __eq__(self, 
               state):
        return (self.name == state.name)
    
    def __hash__(self):
        return hash((self.name))
    
    def __ne__(self, other):
        return not(self == other)
    
    def __repr__(self):
        return(str(self.name))

class MP:
    
    def __init__(self, 
                 states_list: list, 
                 transitions: np.ndarray = np.array([None])):
        """
        Parameters
        ----------
        states_list : list of State
            Collection of all the states in the MP.
            
        transitions : Numpy 2D Array
            State transition probability matrix.
            transitions[i,j] == Probability of transitionning from State i to State j.
        
        Attributes
        ----------
        states_list : list of State
            Collection of all the states in the MP.
        
        nb_states : int
            Number of states in the MP.
            
        transitions : Numpy 2D Array
            State transition probability matrix.
            transitions[i,j] == Probability of transitionning from State i to State j.
            
        successors : dict
            Dictionnary:
                keys: State
                values: [(State, float), ...]
            Maps a State to its possible successor States and keeps track of 
            transition probability.
        """
        self.states_list = states_list
        self.nb_states = len(states_list)
        self.transitions = transitions
        self.successors = self.get_successors() if transitions.any() != None else None
        
    
    def get_sink_states(self):
        """
        Returns
        -------
        list of State
            List of the sinks states.
        """
        sink_states = []
        for i in range(0, self.nb_states):
            if self.transitions[i,i] == 1:
                sink_states.append(self.states_list[i])        
        return(sink_states)
        
    def get_successors(self):
        """
        Returns
        -------
        Dictionnary:
            keys: State
            values: Dict:{ State : float, ...]
            Maps a State to its possible successor States and keeps track of 
            transition probability.
        """
        successors = dict()
        for state in self.states_list:
            if (state not in successors.keys()):
                successors[state] = dict()
            for j in range(0, self.nb_states):
                prob = self.transitions[state.index][j]
                if prob != 0:
                    successors[state][self.states_list[j]] = prob
        return(successors)
        
    def get_stationnary(self):
        
        eig_vals, eig_vecs = eig(self.transistions)
        for i in range(0,self.nb_states):
            if(abs(eig_vals[i] - 1) < 1e-16):
                return(eig[i])



if __name__ == "__main__":
    
# =============================================================================
#     transitions = {
#         1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
#         2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
#         3: {1: 0.7, 2: 0.3},
#         4: {1: 0.3, 2: 0.5, 3: 0.2}
#     }
# =============================================================================
    
    state_list = [State(1), State(2), State(3), State(4)]
    transitions = np.array([[0.1, 0.6, 0.1, 0.2],
                            [0.25, 0.22, 0.24, 0.29],
                            [0.7, 0.3, 0, 0],
                            [0.3, 0.5, 0.2, 0]])
    
    chain = MP(state_list, transitions)
    print(chain.successors)
    
    