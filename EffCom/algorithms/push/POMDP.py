import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from POMDP_model import POMDP_model
from julia import NativeSARSOP, POMDPs

class PODMP_solver(RL_Algorithm):
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.n_states = mdp.n_states

    def run(self, beta):
        self.beta = beta # MAYBE NOT THE BEST WAY

        # create the POMDP interface and declare the solver
        self.POMDP_model = POMDP_model(self.mdp, beta)
        POMDP_interface = self.POMDP_model.get_interface()
        solver = NativeSARSOP.SARSOPSolver( epsilon    = 0.00001,
                                            precision  = 0.0001,
                                            kappa      = 0.5,
                                            delta      = 0.0001,
                                            max_time   = 40.0     )

        # train model
        policy_JuliaObj = POMDPs.solve(solver, POMDP_interface) 

        # extract value function vectors and action map
        alpha_vectors = np.array(policy_JuliaObj.alpha)
        action_map = policy_JuliaObj.action_map

        # create auxiliary functions
        self.best_action = lambda b: get_action(b, alpha_vectors, action_map, self.mdp.n_states, self.mdp.n_actions, self.POMDP_model, self.mdp.gamma)
        self.update_b = lambda b, a_tx, a_rx, o: update_b(b, a_tx, a_rx, o, self.mdp.n_states, self.mdp.P)     


    def eval_perf(self, horizon, n_episodes):
        for episode in range(n_episodes):
            # declare agent objects
            sensor = Sensor(self.best_action, self.best_actionupdate_b)
            actuator = Actuator(self.best_action, self.best_actionupdate_b)

            # set initial state
            b0 = np.zeros((self.mdp.n_states,))
            b0[0,0] = 1
            print("WARNING: initial condition is hard coded to be that the initial state is always 0")
            state = np.random.choices(np.arange(self.n_states), p=b0)

            # run one step to initialize the beliefs
            message = sensor.step(state)
            action = actuator.step(message)
            state = np.random.choices(np.arange(self.n_states), p=self.mdp.P[action,state])

            total_reward = 0
            for t in range(horizon):
                message = sensor.step(state)
                if message is None:
                    transmission_cost = 0
                else:
                    transmission_cost = self.beta
                action = actuator.step(message)

                # Get reward and add to total
                reward = self.mdp.R[state, action]
                combined_reward = reward - transmission_cost
                total_reward += (self.gamma ** t) * combined_reward 

                # Step into new time-step
                state = np.random.choices(np.arange(self.n_states), p=self.mdp.P[action,state])

            print(total_reward)


class Sensor:
    def __init__(self, best_action, update_b):
        self.first = True
        self.best_action = best_action
        self.update_b = update_b
      
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
    def __init__(self, best_action, update_b):
        self.first = True
        self.best_action = best_action
        self.update_b = update_b

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
    # T = POMDP_model.T
    # nA = (2 ** n_states) * n_actions
    # Q = np.zeros((nA, 1))
    # for a in range(nA):
    #     a_tx = a // n_states
    #     string_representation_of_a_tx = f'{a_tx:0{n_states}b}' 
    #     a_tx = np.array([int(bit) for bit in string_representation_of_a_tx])
    #     r_imms = np.zeros((n_states,))
    #     next_vals = np.zeros((n_states,))
    #     for s in range(n_states):
    #         r_imms[s] = POMDP_model.reward(s, a)
    #         next_s = T[:,s,a % n_states]
    #         next_bs = np.zeros((n_states, n_states))
    #         for s_new in range(n_states):
    #             if next_s[s_new] > 0:
    #                 y = a_tx[s_new]
    #                 if y == 1:
    #                     next_bs[s_new, s_new] = 1.0
    #                 else:
    #                     next_bs[s_new] = np.squeeze(T[:,:,a % n_states] @ b) * (1 - a_tx) 
    #                     if np.sum(next_bs[s_new]) > 0:
    #                         next_bs[s_new] = next_bs[s_new] / np.sum(next_bs[s_new])
    #         next_vals[s] = np.dot(np.max(alpha_vectors @ next_bs.T, axis=0), next_s)
    #     Q[a] = np.dot(np.squeeze(b), r_imms + gamma * next_vals)
    # action_taken = np.argmax(Q)
    # a_tx = action_taken // n_states
    # string_representation_of_a_tx = f'{a_tx:0{n_states}b}' 
    # a_tx = np.array([int(bit) for bit in string_representation_of_a_tx])
    # return a_tx, action_taken % n_states
            

    actions_values = alpha_vectors @ b
    best_action = action_map[np.argmax(actions_values)]
    a_tx = np.array([int(el) for el in f'{best_action // n_states:0{n_states}b}'])
    a_rx = best_action % n_states
    return a_tx, a_rx





