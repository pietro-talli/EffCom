from EffCom.mdp import create_randomized_mdps, create_estimation_mdp, MDP
import numpy as np

class PUSH_API:
    def __init__(self, mdp: MDP, t_max: int, H: int, init_type = 'random'):
        self.mdp = mdp
        self.t_max = t_max
        self.H = H

        self.S = self.mdp.n_states
        self.A = self.mdp.n_actions
        self.P = self.mdp.P
        self.R = self.mdp.R
        self.gamma = self.mdp.gamma

        # Policy of the actuator 
        self.pi_a = np.random.randint(0, self.mdp.n_actions, size=(self.mdp.n_states, self.t_max))
        # Policy of the sensor
        if init_type == 'random':
            self.pi_s = np.random.randint(0, 2, size=(self.mdp.n_states, self.t_max, self.mdp.n_states))
        elif init_type == 'never':
            self.pi_s = np.zeros((self.mdp.n_states, self.t_max, self.mdp.n_states))
        elif init_type == 'always':
            self.pi_s = np.ones((self.mdp.n_states, self.t_max, self.mdp.n_states))
        else:
            raise ValueError('init_type must be random, never or always')

        # Value function of the actuator
        self.Q_a = np.zeros((self.mdp.n_states, self.t_max, self.mdp.n_actions))
        # Value function of the sensor
        self.Q_s = np.zeros((self.mdp.n_states, self.t_max, self.mdp.n_states, 2))

    def run(self, beta: float):
        # The algorithm is alternating policy iteration 
        # between the actuator and the sensor
        policy_stable = False
        changes_a = False
        changes_s = False
        while not policy_stable:
            # Policy iteration for the actuator
            changes_a = self.policy_iteration_actuator(beta)
            # Policy iteration for the sensor
            changes_s = self.policy_iteration_sensor(beta)

            if not changes_a and not changes_s:
                policy_stable = True
            
    def policy_iteration_actuator(self, beta: float):
        changes = False
        policy_stable = False
        old_pi = self.pi_a.copy()
        while not policy_stable:
            # Policy evaluation
            self.policy_evaluation_actuator()
            # Policy improvement
            policy_stable = self.policy_improvement_actuator(beta)
            if not np.array_equal(self.pi_a, old_pi):
                changes = True
        return changes
    
    def policy_evaluation_actuator(self):
        delta = 1
        while delta > 0.01:
            for s in range(self.S):
                old_value = self.V_a[s]
                action = self.pi_a[s,0]
                belief = self.P[action,s]
                self.V_a[s] = (self.R[action,s] + self.gamma * self.V_a).dot(belief)
                delta = max(delta, abs(old_value - self.V_a[s]))

    def policy_improvement_actuator(self,beta: float):
        policy_stable = True
        for s in range(self.S):
            belief = np.zeros(self.S)
            belief[s] = 1
            for t in range(self.t_max):
                old_action = self.pi_a[s,t]
                values = np.zeros(self.A)
                for a in range(self.A):
                    belief_next = np.dot(belief, self.P[a])
                    for s_ in range(self.S):
                        c = self.pi_s[s,t]
                        if c==0:
                            values[a] += (self.R[a,:,s_] + self.gamma * self.V_a[s_])*belief_next[s_]
                        else:
                            values[a] += (self.R[a,:,s_] + self.gamma * self.V_a[s_])*belief_next[s_] - beta
                self.pi_a[s,t] = np.argmax(values)
                if old_action != self.pi_a[s,t]:
                    policy_stable = False
                belief = np.dot(belief, self.P[self.pi_a[s,t]])
                #prune the belief
                for s_ in range(self.S):
                    if self.pi_s[s,t,s_] == 1:
                        belief[s_] = 0
        return policy_stable
    
    def policy_iteration_sensor(self, beta: float):
        changes = False
        policy_stable = False
        old_pi = self.pi_s.copy()
        while not policy_stable:
            # Policy evaluation
            self.policy_evaluation_sensor(beta)
            # Policy improvement
            policy_stable = self.policy_improvement_sensor(beta)
            if not np.array_equal(self.pi_s, old_pi):
                changes = True
        return changes
    
    def policy_evaluation_sensor(self, beta: float):
        delta = 1
        while delta > 0.01:
            for s in range(self.S):
                belief = np.zeros(self.S)
                belief[s] = 1
                for t in range(self.t_max):
                    action = self.pi_a[s,t]
                    belief_ = np.dot(belief, self.P[action])
                    for s_ in range(self.S):
                        c = self.pi_s[s,t,s_]
                        old_value = self.V_s[s,t,s_]
                        if c==0:
                            if t == self.t_max - 1:
                                self.V_s[s,t,s_] = np.dot(belief,self.R[action])[s_] -beta + self.gamma * self.V_s[s_,0].dot(belief_)
                            else:
                                self.V_s[s,t,s_] = np.dot(belief,self.R[action])[s_] + self.gamma * self.V_s[s,t+1].dot(self.P[self.pi_a[s,t+1],s_])
                        else:
                            self.V_s[s,t,s_] = (np.dot(belief,self.R[action]) + self.gamma * self.V_s[s_,0,]).dot(belief_) - beta