import numpy as np
from julia.POMDPTools import Deterministic
from julia.QuickPOMDPs import DiscreteExplicitPOMDP


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
        """The POMDP actions are in the form (0/1,0/1,...,0/1,0/1/2/3/../n_actions-1), where the last element is the
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
        out = DiscreteExplicitPOMDP( range(self.n_states), # state space
                                     range((2 ** self.n_states) * self.n_actions), # action space
                                     range(self.n_states + 1), # observation space
                                     self.transition,
                                     self.observation,
                                     self.reward,
                                     self.gamma, 
                                     Deterministic(0) )
        print("WARNING: the POMDP is defined with the initial condition of starting at state 0")
        return out
