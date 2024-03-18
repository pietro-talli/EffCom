import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from EffCom.algorithms.push.POMDP_model import POMDP_model
from julia import NativeSARSOP, POMDPs


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
                                            verbose    = False   )

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
        AoIs = [[] for _ in range(self.n_states)]

        for episode in range(n_episodes):
            # declare agent objects
            sensor = Sensor(self.n_states, self.best_action, self.update_b)
            actuator = Actuator(self.n_states, self.best_action, self.update_b)

            # set initial state
            b0 = np.zeros((self.mdp.n_states,))
            b0[0] = 1
            # print("WARNING: initial condition is hard coded to be that the initial state is always 0")
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





