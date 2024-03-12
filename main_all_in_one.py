import numpy as np
import copy
import pickle
import csv
import json
import os
from julia import Pkg
from julia import NativeSARSOP, POMDPs
import concurrent.futures

from julia.POMDPTools import SparseCat,Deterministic
from julia.QuickPOMDPs import DiscreteExplicitPOMDP
from quickpomdps import QuickPOMDP


def transition_quick(s, a, n_s, n_a, P):
    return SparseCat(range(n_s), P[a % n_a, s])

def observation_quick(a, sp, n_s, n_a):
    a_tx = a // n_a
    string_representation_of_a_tx = f'{a_tx:0{n_s}b}' 
    a_tx = [int(bit) for bit in string_representation_of_a_tx]
    if a_tx[sp] == 1:
        return Deterministic(sp)
    return Deterministic(n_s)

def reward_quick(s, a, R, P, gamma, t_cost, n_s, n_a):
    """one step reward, including the expected transmission costs"""
    a_tx = a // n_a
    string_representation_of_a_tx = f'{a_tx:0{n_s}b}' 
    a_tx = [int(bit) for bit in string_representation_of_a_tx]
    a_rx = a % n_a
    
    immediate_reward = R[s, a_rx] # immmediate reward from taking action a_rx in state a
    transmissions = np.dot(P[a_rx,s], a_tx) # probability of transmission
    return immediate_reward - gamma * transmissions * t_cost


class POMDP_model():
    """Define the POMDP which can be equivalently solved to solve the original problem. Each time step starts after the 
    transmitter's decision to either transmit or not. """
    def __init__(self, mdp, t_cost):
        self.n_states = mdp.n_states 
        self.n_actions = mdp.n_actions
        self.P = mdp.P
        self.R = mdp.R
        self.gamma = mdp.gamma
        self.t_cost = t_cost

    def decode_action(self, a):
        """The POMDP actions are in the form (0/1, 0/1, ..., 0/1, 0/1/2/3/../n_actions-1), where the last element is the
        action taken by the receiver and the first ones are the decision rule of the transmitter."""
        a_tx = a // self.n_actions
        string_representation_of_a_tx = f'{a_tx:0{self.n_states}b}' 
        a_tx = [int(bit) for bit in string_representation_of_a_tx]
        a_rx = a % self.n_actions
        return a_tx, a_rx
    
    def transition(self, s, a, s_new):
        """probability of transitionining to state s_new by taking action a in state s"""
        return self.P[a % self.n_actions, s, s_new]
    
    

    def observation(self, a, s_new, o):
        """probability of observing o if we are in s_new and we previously took action a"""
        a_tx, _ = self.decode_action(a)
        if o == self.n_states: # no message sent
            return int(a_tx[s_new] == 0)
        return int(a_tx[s_new] == 1 and s_new == o) # message sent
    
    
    def reward(self, s, a):
        """one step reward, including the expected transmission costs"""
        a_tx, a_rx = self.decode_action(a)
        immediate_reward = self.R[s, a_rx] # immmediate reward from taking action a_rx in state a
        transmissions = np.dot(self.P[a_rx,s], a_tx) # probability of transmission
        return immediate_reward - self.gamma * transmissions * self.t_cost
    
    
    def get_interface(self):
        """Define the POMDP using the interface required by NativeSARSOP."""
        if not np.all(np.isclose(np.sum(self.P, axis=2), 1.0)):
            raise Exception("2nd axis of P does not sum to 1")
        out = DiscreteExplicitPOMDP( range(self.n_states), # state space
                                     range((2 ** self.n_states) * self.n_actions), # action space
                                     range(self.n_states + 1), # observation space
                                     self.transition,
                                     self.observation,
                                     self.reward,
                                     self.gamma, 
                                     Deterministic(0) )
        
        # out = QuickPOMDP( states = range(self.n_states),
        #                   actions = range((2 ** self.n_states) * self.n_actions),
        #                   observations = range(self.n_states + 1),
        #                   initialstate = Deterministic(0),
        #                   discount = self.gamma, 
        #                   transition = lambda s, a: transition_quick(s, a, self.n_states, self.n_actions,self.P),
        #                   observation = lambda a, s_new: observation_quick(a, s_new, self.n_states, self.n_actions), 
        #                   reward = lambda s, a: reward_quick(s, a, self.R, self.P, self.gamma, self.t_cost, self.n_states, self.n_actions) )


        print("WARNING: the POMDP is defined with the initial condition of starting at state 0")
        return out



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




class RL_Algorithm():
    def __init__(self, MDP: MDP):
        self.mdp = MDP
        
    def run(self):
        pass

    def eval_perf(self):
        pass

class POMDP_solver(RL_Algorithm):
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.mdp.R = np.sum(self.mdp.R * self.mdp.P, axis=2).T # (a,s,s_new)
        self.n_states = mdp.n_states

    def run(self, beta):
        self.beta = beta # MAYBE NOT THE BEST WAY

        # create the POMDP interface and declare the solver
        if not np.all(np.isclose(np.sum(self.mdp.P, axis=2), 1.0)):
            raise Exception("2nd axis of P does not sum to 1")
        self.POMDP_model = POMDP_model(self.mdp, beta)
        POMDP_interface = self.POMDP_model.get_interface()
        solver = NativeSARSOP.SARSOPSolver( epsilon    = 0.01,
                                            precision  = 0.01,
                                            kappa      = 0.5,
                                            delta      = 0.0001,
                                            max_time   = 300.0,  
                                            verbose    = True   )

        # train model
        policy_JuliaObj = POMDPs.solve(solver, POMDP_interface) 

        # extract value function vectors and action map
        alpha_vectors = np.array(policy_JuliaObj.alphas)
        action_map = policy_JuliaObj.action_map

        # create auxiliary functions
        self.best_action = lambda b: get_action(b, alpha_vectors, action_map, self.mdp.n_states, self.mdp.n_actions, self.POMDP_model, self.mdp.gamma)
        self.update_b = lambda b, a_tx, a_rx, o: update_b(b, a_tx, a_rx, o, self.mdp.n_states, self.mdp.P)     


    def eval_perf(self, horizon, n_episodes):
        tot_total_reward = 0
        tot_total_raw_reward = 0
        tot_total_reward_undiscounted = 0
        tot_total_raw_reward_undiscounted = 0
        tot_total_ch_uti = 0
        AoIs = [[]] * self.n_states

        for episode in range(n_episodes):
            # declare agent objects
            sensor = Sensor(self.n_states, self.best_action, self.update_b)
            actuator = Actuator(self.n_states, self.best_action, self.update_b)

            # set initial state
            b0 = np.zeros((self.mdp.n_states,))
            b0[0] = 1
            print("WARNING: initial condition is hard coded to be that the initial state is always 0")
            state = np.random.choice(np.arange(self.n_states), p=b0)

            # run one step to initialize the beliefs
            message = sensor.step(state)
            action = actuator.step(message)
            state = np.random.choice(np.arange(self.n_states), p=self.mdp.P[action,state])

            total_reward = 0
            total_raw_reward = 0
            total_reward_undiscounted = 0
            total_raw_reward_undiscounted = 0
            total_ch_uti = 0
            current_AoI = 0
            active_state = 0
            for t in range(horizon):
                message = sensor.step(state)
                if message is None:
                    transmission_cost = 0
                    if current_AoI > 0: # current AoI is only 0 when no message has occurred yet (we use this to prevent a false value at the start)
                        current_AoI += 1
                else:
                    transmission_cost = self.beta
                    if current_AoI > 0:
                        AoIs[active_state].append(current_AoI)
                    current_AoI = 1
                    active_state = message

                action = actuator.step(message)

                # Get reward and add to total
                reward = self.mdp.R[state, action]
                combined_reward = reward - transmission_cost
                total_reward += (self.mdp.gamma ** t) * combined_reward / horizon
                total_raw_reward += reward / horizon
                total_reward_undiscounted += combined_reward / horizon
                total_raw_reward_undiscounted += reward / horizon
                total_ch_uti += int(message is not None) / horizon

                # Step into new time-step
                state = np.random.choice(np.arange(self.n_states), p=self.mdp.P[action,state])

            # Add results to total averages
            tot_total_reward += (total_reward / n_episodes)
            tot_total_raw_reward += (total_raw_reward / n_episodes)
            tot_total_reward_undiscounted += (total_reward_undiscounted / n_episodes)
            tot_total_raw_reward_undiscounted += (total_raw_reward_undiscounted / n_episodes)
            tot_total_ch_uti += (total_ch_uti / n_episodes)

        # print(tot_total_reward)
        # print(tot_total_raw_reward)
        # print(tot_total_reward_undiscounted)
        # print(tot_total_ch_uti)

        return tot_total_reward, tot_total_raw_reward, tot_total_reward_undiscounted, tot_total_raw_reward_undiscounted, tot_total_ch_uti, AoIs


class Sensor:
    def __init__(self, n_states, best_action, update_b):
        self.first = True
        self.best_action = best_action
        self.update_b = update_b
        self.n_s = n_states
      
    def step(self, s):
        self.s = s # observe the state
            
        # if first action of the episode, force a transmission
        if self.first: 
            self.first = False 
            y = self.s # transmitted symbol is the state
            self.b = np.zeros((self.n_s,1)) # update belief for the next step
            self.b[self.s] = 1
        else:
            a_tx, a_rx = self.best_action(self.b) # get best action given the belief

            if int(a_tx[self.s]) == 1: # check the decision rule
                y = self.s # if 1, transmit the state symbol
            else:
                y = None # if 0, transmit nothing (None symbol)

            self.b = self.update_b(self.b, a_tx, a_rx, y)
        return y # return the transmitted symbol



class Actuator: 
    def __init__(self, n_states, best_action, update_b):
        self.first = True
        self.best_action = best_action
        self.update_b = update_b
        self.n_s = n_states

    def step(self, y):
        if self.first:
            self.first = False
            self.b = np.zeros((self.n_s,1))
            self.b[y] = 1
        else:
            a_tx, a_rx = self.last_a
            self.b = self.update_b(self.b, a_tx, a_rx, y)

        self.last_a = self.best_action(self.b)
        _, a_rx = self.last_a
        return a_rx
        
    


def update_b(b, a_tx, a_rx, o, n_states, P):
    assert np.shape(b) == (n_states, 1)
    if o is None:
        b = (P[a_rx].T @ b) * np.expand_dims((1 - a_tx), axis=1)
        assert np.sum(b) > 0
        b = b / np.sum(b)            
    else:
        b = np.zeros((n_states,1))
        b[o] = 1
    return b


def get_action(b, alpha_vectors, action_map, n_states, n_actions, POMDP_model, gamma):
    assert np.shape(b) == (n_states, 1)      
    actions_values = alpha_vectors @ b
    best_action = action_map[np.argmax(actions_values)]
    a_tx = np.array([int(el) for el in f'{best_action // n_actions:0{n_states}b}'])
    a_rx = best_action % n_actions
    return a_tx, a_rx





class RemotePolicyIteration(RL_Algorithm):
    """
    Class that implements the policy iteration algorithm where the state is 
    communicated by a remote sensor with a cost. 

    Parameters:
    -----------
    mdp: MDP
        MDP object
    horizon: int
        Horizon of the problem
    t_max: int
        Maximum number of consecutive actions before sensing

    Attributes:
    -----------
    mdp: MDP
        MDP object
    H: int
        Horizon of the problem
    V: np.array
        Value function
    pi: np.array
        Policy
    t_max: int
        Maximum number of consecutive actions before sensing
    """
    def __init__(self, mdp: MDP, horizon: int, t_max: int):
        self.mdp = mdp
        self.H = horizon
        self.V_sensor = np.zeros((self.mdp.N_s, self.H, self.mdp.N_s))
        self.pi_sensor = np.random.randint(0,2,(self.mdp.N_s, self.H, self.mdp.N_s))

        for s in range(self.mdp.N_s):
            self.pi_sensor[:,:,s] = 0

        self.V_actuator = np.zeros((self.mdp.N_s, self.H))
        self.pi_actuator = np.random.randint(0,self.mdp.N_a,(self.mdp.N_s, self.H))

        self.t_max = t_max

        self.gsr = np.sqrt(self.mdp.gamma) # gamma square root

    def run(self, beta):
        policy_stable = False
        max_iter = 100
        i = 0
        while not policy_stable:
            self.policy_evaluation(beta)
            policy_stable = self.policy_improvement(beta)
            # print(self.eval_perf(100))
            # print(np.mean(self.V_actuator[:,0]))
            i += 1
            if i == max_iter:
                break

    def policy_evaluation(self, beta):
        delta = 1
        while delta > 1e-3:
            delta = 0
            for s in range(self.mdp.N_s):
                belief = np.zeros(self.mdp.N_s)
                belief[s] = 1

                for t in range(self.H):
                    old_value = self.V_actuator[s,t]
                    action = self.pi_actuator[s,t]
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(belief, self.mdp.R[action])

                    R = np.dot(belief_after_action, immediate_reward)
                    self.V_actuator[s,t] = R + self.gsr*(np.dot(self.V_sensor[s,t,:],belief_after_action))
                    delta = max(delta, np.abs(old_value - self.V_actuator[s,t]))

                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 0:
                            old_value_sensor = self.V_sensor[s,t,s_curr]

                            if t == self.H-1:
                                self.V_sensor[s,t,s_curr] = -beta + self.gsr*self.V_actuator[s_curr,0]
                                delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))
                            else:
                                #UPDATE
                                self.V_sensor[s,t,s_curr] = self.gsr*self.V_actuator[s,t+1]
                                delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))

                        else:
                            old_value_sensor = self.V_sensor[s,t,s_curr]

                            #UPDATE
                            self.V_sensor[s,t,s_curr] = -beta + self.gsr*self.V_actuator[s_curr,0]
                            delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))

                    belief = belief_after_action

    def policy_improvement(self, beta):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(min(self.t_max, self.H)):
                # Update the policy for the MDP action
                old_action = self.pi_actuator[s,t]
                values = np.zeros(self.mdp.N_a)
                for action in range(self.mdp.N_a):
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(np.dot(belief, self.mdp.R[action]), belief_after_action)
                    value = 0
                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 1:
                            value += belief_after_action[s_curr]*(immediate_reward - self.gsr*beta + self.mdp.gamma*self.V_actuator[s_curr,0])
                        else:
                            reward_simulated = self.V_sensor[s,t,s_curr] # self.simulate(s,t,s_curr,beta, t)
                            value += belief_after_action[s_curr]*(immediate_reward + self.mdp.gamma*reward_simulated)
                    values[action] = value
                    
                self.pi_actuator[s,t] = np.argmax(values)
                if old_action != self.pi_actuator[s,t]:
                    policy_stable = False

                # Update the policy of the sensor
                for s_curr in range(self.mdp.N_s):
                    old_c = self.pi_sensor[s,t,s_curr]

                    if t == self.t_max-1 or t == self.H - 1:
                        self.pi_sensor[s,t,s_curr] = 1
                    else:
                        values = np.zeros(2)
                        values[0] = self.gsr*self.V_actuator[s,t+1]
                        values[1] = -beta + self.gsr*self.V_actuator[s_curr,0]
                        self.pi_sensor[s,t,s_curr] = np.argmax(values)
                        if np.abs(values[0] - values[1]) < 1e-3:
                            self.pi_sensor[s,t,s_curr] = 1

                    if old_c != self.pi_sensor[s,t,s_curr]:
                        policy_stable = False

                belief = np.dot(belief, self.mdp.P[self.pi_actuator[s,t]])
        return policy_stable

    def tbaa(self,b,s,t):
        for s_curr in range(self.mdp.N_s):
            if self.pi_sensor[s,t,s_curr] == 1:
                b[s_curr] = 0
        return b/np.sum(b)

    def simulate(self, s, t, s_curr, beta, t_0):
        reward = 0
        if t == self.t_max-1:
            return -beta + self.gsr*self.V_actuator[s_curr,0]
        else:
            action = self.pi_actuator[s,t+1]
            
    
    def eval_perf(self, n_episodes):
        tot_r = 0
        tot_c = 0
        for _ in range(n_episodes):
            s = np.random.randint(self.mdp.N_s)
            s_last = s
            t = 0
            for h in range(self.H):
                action = self.pi_actuator[s_last,t]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi_sensor[s_last,t,s_next]
                if h != self.H-1: tot_c += c_t
                if c_t == 0:
                    t += 1
                else:
                    t = 0
                    s_last = s_next
                s = s_next
        return tot_r/(n_episodes*self.H), tot_c/(n_episodes*self.H)
    
    def get_PeakAoI(self, target_state: int, n_episodes: int = 1000):
        tot_r = 0
        tot_c = 0
        PAOI = []
        paoi = 0
        for _ in range(n_episodes):
            s = np.random.randint(self.mdp.N_s)
            s_last = s
            t = 0

            if s_last == target_state:
                paoi = 1
            else:
                paoi = 0

            for h in range(self.H):
                action = self.pi_actuator[s_last,t]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi_sensor[s_last,t,s_next]
                if h != self.H-1: tot_c += c_t
                if c_t == 0:
                    t += 1
                else:
                    if paoi == 1: PAOI.append(t+1)
                    t = 0
                    s_last = s_next

                    
                    if s_last == target_state:
                        paoi = 1
                    else:
                        paoi = 0

                s = s_next
        return PAOI
    









def load_mdps(name_of_file: str):
    """
    Create a list of randomized MDPs from the parameters in the file
    and a list of beta values.

    In the file there should be a params["mdp] containing, at least, the following keys:
    - N_states: int
        Number of states in the MDP (even integer)
    - N_actions: int
        Number of actions in the MDP
    - gamma: float
        Discount factor
    - r_seed: int
        Random seed
    - reward_decay: float
        Reward decay

    In the file there should be a params["beta"] containing, at least, the following keys:
    - min: float
        Minimum value of beta
    - max: float
        Maximum value of beta
    - n: int
    
    Parameters:
    -----------
    name_of_file: str
        Name of the file containing the parameters of the MDPs and the beta values

    Returns:
    --------
    mdps: list
        List of MDPs
    betas: numpy array
        Array of beta values
    """

    with open(name_of_file, 'r') as f:
        params = json.load(f)
    betas = np.linspace(params["beta"]["min"], params["beta"]["max"], params["beta"]["n"])
    return create_randomized_mdps(**params["mdp"]), betas, params




def create_folder(folder_name):
    if not os.path.exists(folder_name):  # Check if the folder already exists
        os.makedirs(folder_name)  # Create the folder if it doesn't exist
        return folder_name
    else:
        i = 0
        while True:
            new_folder_name = f"{folder_name}_{i}"  # Append a number to the folder name
            if not os.path.exists(new_folder_name):  # Check if the new folder name does not exist
                os.makedirs(new_folder_name)  # Create the new folder
                return new_folder_name
            i += 1












def run_one(p_list):
    pomdp, policy_iteration, beta, horizon, tmax, runs_per_instance, mdp_i, results_folder = p_list
    pomdp.run(beta)
    tot_reward, tot_raw_reward, tot_reward_undiscounted, tot_raw_reward_undiscounted, tot_ch_uti, AoIs = pomdp.eval_perf(horizon, runs_per_instance)

    policy_iteration.run(beta)
    r,c = policy_iteration.eval_perf(runs_per_instance)
    aois_policy_iteration = []
    for state in range(mdp.n_states):
        aois_policy_iteration.append(policy_iteration.get_PeakAoI(state, 10))

    with open("{}/{}_pomdp_aoi_{}".format(results_folder,str(mdp_i), beta), 'wb') as f:
        pickle.dump(AoIs, f)
    with open("{}/{}_politer_aoi_{}".format(results_folder,str(mdp_i), beta), 'wb') as f:
        pickle.dump(aois_policy_iteration, f)
    with open("{}/{}_results.csv".format(results_folder,str(mdp_i)), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([tot_reward, tot_raw_reward, tot_reward_undiscounted, tot_raw_reward_undiscounted, tot_ch_uti, r, c])

    return None



if __name__ == "__main__":
    # Activate the current Julia environment
    Pkg.activate(".")

    # Add or update package dependencies
    Pkg.add("POMDPs")
    Pkg.add("POMDPTools")
    Pkg.add("Distributions")
    Pkg.add(url="https://github.com/ed13santi/NativeSARSOP.jl")

    # Resolve and instantiate the environment
    Pkg.instantiate()



    # T = np.array([[0.4,0.0,0.0,0.0,1.0],
    #               [0.6,0.4,0.0,0.0,0.0],
    #               [0.0,0.6,0.4,0.0,0.0],
    #               [0.0,0.0,0.6,0.4,0.0],
    #               [0.0,0.0,0.0,0.6,0.0]])
    # P = np.expand_dims(T.T, 0)
    # P = np.tile(P, (5,1,1))
    # R = np.tile(np.expand_dims(np.eye(5), axis=2), (1,1,5))
    # mdp = MDP(5, 5, P, R, 0.99)

    # horizon = 100

    # type of experiment
    experiment_types = ["push_sparse_reward", "push_spread_reward","estimation"]

    params_ll = []

    for experiment_type in experiment_types:
        # load mdps
        mdps, betas, params = load_mdps("configs/{}.json".format(experiment_type))
        horizon = params["horizon"]
        tmax = params["t_max"]
        runs_per_instance = params["runs_per_instance"]

        # create folder for the results
        results_folder = create_folder(f"10_state_2/{experiment_type}")

        # run experiments
        for mdp_i, mdp in enumerate(mdps):
            for beta in betas:
                assert np.all(np.isclose(np.sum(mdp.P, axis=2), 1.0))

                pomdp = POMDP_solver(copy.deepcopy(mdp))
                policy_iteration = RemotePolicyIteration(copy.deepcopy(mdp), horizon, tmax)

                params_ll.append((pomdp, policy_iteration, beta, horizon, tmax, runs_per_instance, mdp_i, results_folder))
                
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     #results_and_Pis = list(executor.map(run_test, params))
    #     results = list(executor.map(run_one, params_ll))

    for p_el in params_ll:
        results = run_one(p_el)




