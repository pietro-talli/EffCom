import numpy as np

class MDP:
    '''
    Markov Decision Process (MDP) class
    
    Attributes:
    - n_states: int
        Number of states
    - n_actions: int
        Number of actions
    - P: np.ndarray
        Transition probability matrix of shape (n_actions, n_states, n_states)
    - R: np.ndarray
        Reward matrix of shape (n_actions, n_states, n_states)
    - gamma: float
        Discount factor
    '''
    def __init__(self, n_states: int, n_actions: int, P: np.ndarray, R: np.ndarray, gamma: float = 0.99):
        self.n_states = n_states
        self.N_s = n_states
        self.n_actions = n_actions
        self.N_a = n_actions
        self.gamma = gamma
        assert P.shape == (n_actions, n_states, n_states), f'P.shape = {P.shape}, expected {(n_actions, n_states, n_states)}'
        self.P = P
        assert R.shape == (n_actions, n_states, n_states), f'R.shape = {R.shape}, expected {(n_actions, n_states, n_states)}'
        self.R = R

        # density of the transition matrices of the MDP
        self.density = None

def create_randomized_mdps(N_states: int, N_actions: int, gamma: float, r_seed: int = 1234, reward_decay: float = 10.0):
    """
    This method create a list of random MDP starting from the same 
    deterministic structure and adding variability in the transition
    probabilities and rewards.
   
    Parameters:
    -----------
    N_states: int
        Number of states in the MDP (even integer)
    N_actions: int
        Number of actions in the MDP
    gamma: float
        Discount factor
    
    Returns:
    --------
    mdp_list: list
        List of random MDPs
    """
    # set the seed to do the same experiments with different communication costs
    print(N_states, N_actions, gamma, r_seed, reward_decay)

    np.random.seed(r_seed)

    # create a list of MDPs
    mdp_list = [MDP(N_states, N_actions,np.zeros((N_actions, N_states, N_states)), np.zeros((N_actions, N_states, N_states)), gamma) for _ in range(int(N_states/2))]

    optimal_state = np.random.randint(0,N_states)

    # from the first MDP, create the other MDPs by adding links between states
    for a in range(N_actions):
        for s in range(N_states):
            mdp_list[0].P[a][s] = np.zeros(N_states)
            random_idx = np.random.randint(N_states)
            mdp_list[0].P[a][s][random_idx] = 1

            for i in range(1, int(N_states/2)):
                offsets = [j for j in range(-i, i+1)]
                new_link_set = [(random_idx + j) % N_states for j in range(-i, i+1)]
                mdp_list[i].P[a][s] = np.zeros(N_states)
                for offset,idx in zip(offsets, new_link_set):
                    mdp_list[i].P[a][s][idx] = 0.5 - np.abs(offset)/len(new_link_set)

                mdp_list[i].P[a,s,random_idx] += 0.1
                mdp_list[i].P[a,s,:] /= np.sum(mdp_list[i].P[a,s,:])
                if not np.all(mdp_list[i].P[a,s,:] >= 0):
                    print(mdp_list[i].P[a,s,:])
                    assert False

        for idx, m in enumerate(mdp_list):
            m.R = mdp_list[0].R 
            m.density = (2*idx+1)/N_states
             
            for i in range(N_states):
                m.R[:,:,i] = 10*np.exp(-np.abs(i-optimal_state)*reward_decay)

    for mdp in mdp_list:
        assert np.all(np.isclose(np.sum(mdp.P, axis=2), 1.0))
        if not np.all(np.isclose(np.sum(mdp.P, axis=2), 1.0)):#, rtol=0.1,atol=0.1)):
            print(mdp.P)
            assert False
        
    return mdp_list
