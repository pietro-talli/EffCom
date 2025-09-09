import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm

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
    def __init__(self, mdp: MDP, horizon: int, t_max: int, init_policy: int = 0):
        self.mdp = mdp
        self.H = horizon

        self.t_max = t_max

        self.V_sensor = np.zeros((self.mdp.N_s, self.t_max, self.mdp.N_s))
        self.pi_sensor = np.ones((self.mdp.N_s, self.t_max, self.mdp.N_s))*init_policy

        #for s in range(self.mdp.N_s):
        #    self.pi_sensor[:,:,s] = 0


        self.V_actuator = np.zeros((self.mdp.N_s, self.t_max))
        self.pi_actuator = np.random.randint(0,self.mdp.N_a,(self.mdp.N_s, self.t_max))

        self.gsr = np.sqrt(self.mdp.gamma) # gamma square root

    def run(self, beta):
        max_iter = 20
        i = 0
        prev_stable = False
        counter = 0
        while True:

            policy_stable1 = False
            mean_val_act = 0
            while not policy_stable1:
                
                self.policy_evaluation_actuator(beta)
                policy_stable1 = self.policy_improvement_actuator(beta)
                
                if mean_val_act > np.mean(self.V_actuator):
                    break
                mean_val_act = np.mean(self.V_actuator)
                #print(mean_val_act)


            #print('actuator done')

            policy_stable2 = False
            mean_val_sensor = 0 
            while not policy_stable2:
                self.policy_evaluation_sensor(beta)
                policy_stable2 = self.policy_improvement_sensor(beta)
                
                if mean_val_sensor > np.mean(self.V_sensor):
                    break
                mean_val_sensor = np.mean(self.V_sensor)
                print(counter, ',', np.mean(self.V_sensor[:,0,0]) * (1-self.mdp.gamma))
                counter += 1
                #print(mean_val_sensor)
            #print('sensor done')

            i += 1
            if i == max_iter:
                break

            if policy_stable1 and policy_stable2 and prev_stable:
                break
            prev_stable = policy_stable1 and policy_stable2
            
    def policy_evaluation_actuator(self, beta):
        delta = 1
        while delta > 1e-3:
            delta = 0
            for s in range(self.mdp.N_s):
                belief = np.zeros(self.mdp.N_s)
                belief[s] = 1

                for t in range(self.t_max):
                    old_value = self.V_actuator[s,t]
                    action = self.pi_actuator[s,t]
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(belief, self.mdp.R[action])

                    self.V_actuator[s,t] = 0 # reset the value function
                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 0:
                            if t == self.t_max-1:
                                self.V_actuator[s,t] += (-beta + self.mdp.gamma*self.V_actuator[s_curr,0] + immediate_reward[s_curr])*belief_after_action[s_curr]
                            else:
                                #UPDATE
                                self.V_actuator[s,t] += (immediate_reward[s_curr] + self.mdp.gamma*self.V_actuator[s,t+1])*belief_after_action[s_curr]
                        else:
                            #UPDATE
                            self.V_actuator[s,t] += (-beta + self.mdp.gamma*self.V_actuator[s_curr,0] + immediate_reward[s_curr])*belief_after_action[s_curr]
                    
                    belief = belief_after_action
                    belief = self.tbaa_actuator(belief,s,t)
                    delta = max(delta, np.abs(old_value - self.V_actuator[s,t]))

    def policy_evaluation_sensor(self, beta):
        delta = 1
        while delta > 1e-3:
            delta = 0
            for s in range(self.mdp.N_s):
                belief = np.zeros(self.mdp.N_s)
                belief[s] = 1

                for t in range(self.t_max):
                    action = self.pi_actuator[s,t]
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(belief, self.mdp.R[action])

                    for s_curr in range(self.mdp.N_s):
                        old_value = self.V_sensor[s,t,s_curr]
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 0:
                            if t == self.t_max-1:
                                self.V_sensor[s,t,s_curr] = (-beta + self.mdp.gamma*self.V_sensor[s_curr,0].dot(self.mdp.P[self.pi_actuator[s_curr,0],s_curr]) + immediate_reward[s_curr])
                            else:    
                                self.V_sensor[s,t,s_curr] = (immediate_reward[s_curr] + self.mdp.gamma*self.V_sensor[s,t+1].dot(self.mdp.P[self.pi_actuator[s,t+1],s_curr]))
                        else:
                            self.V_sensor[s,t,s_curr] = (-beta + self.mdp.gamma*self.V_sensor[s_curr,0].dot(self.mdp.P[self.pi_actuator[s_curr,0],s_curr]) + immediate_reward[s_curr])
                        delta = max(delta, np.abs(old_value - self.V_sensor[s,t,s_curr]))

                    belief = belief_after_action
                    belief = self.tbaa_sensor(belief,s,t)

    def policy_improvement(self, beta, check = 0):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(self.t_max):
                if check == 0:
                    # Update the policy for the MDP action
                    old_action = self.pi_actuator[s,t]
                    values = np.zeros(self.mdp.N_a)
                    for action in range(self.mdp.N_a):
                        belief_after_action = np.dot(belief, self.mdp.P[action])
                        immediate_reward = np.dot(belief, self.mdp.R[action])
                        value = 0
                        for s_curr in range(self.mdp.N_s):
                            c = self.pi_sensor[s,t,s_curr]
                            if c == 1:
                                value += belief_after_action[s_curr]*(immediate_reward[s_curr] - beta + self.mdp.gamma*self.V_actuator[s_curr,0])
                            else:
                                if t == self.t_max-1:
                                    value += belief_after_action[s_curr]*(-beta + self.mdp.gamma*self.V_actuator[s_curr,0] + immediate_reward[s_curr])
                                else: 
                                    value += belief_after_action[s_curr]*(immediate_reward[s_curr] + self.mdp.gamma*self.V_actuator[s,t+1])
                        values[action] = value
                        
                    self.pi_actuator[s,t] = np.argmax(values)
                    if old_action != self.pi_actuator[s,t]:
                        policy_stable = False
                if check == 1:
                    # Update the policy of the sensor
                    action_actuator = self.pi_actuator[s,t]
                    if t != self.t_max-1:
                        action_actuator_blind = self.pi_actuator[s,t+1]
                    belief_after_action = np.dot(belief, self.mdp.P[action_actuator])
                    R = np.dot(belief, self.mdp.R[action_actuator])
                    for s_curr in range(self.mdp.N_s):
                        old_c = self.pi_sensor[s,t,s_curr]

                        if t == self.t_max-1:
                            self.pi_sensor[s,t,s_curr] = 1
                        else:
                            values = np.zeros(2)
                            values[1] = (R[s_curr] -beta + self.mdp.gamma*self.V_sensor[s_curr,0].dot(self.mdp.P[self.pi_actuator[s_curr,0],s_curr]))
                            values[0] = (R[s_curr] + self.mdp.gamma*self.V_sensor[s,t+1].dot(self.mdp.P[action_actuator_blind,s_curr]))
                            self.pi_sensor[s,t,s_curr] = np.argmax(values)

                        if old_c != self.pi_sensor[s,t,s_curr]:
                            policy_stable = False

                belief = np.dot(belief, self.mdp.P[self.pi_actuator[s,t]])
                belief = self.tbaa(belief,s,t)
        return policy_stable

    def policy_improvement_actuator(self, beta):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(self.t_max):
                # Update the policy for the MDP action
                old_action = self.pi_actuator[s,t]
                values = np.zeros(self.mdp.N_a)
                for action in range(self.mdp.N_a):
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(belief, self.mdp.R[action])
                    value = 0
                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 1:
                            value += belief_after_action[s_curr]*(immediate_reward[s_curr] - beta + self.mdp.gamma*self.V_actuator[s_curr,0])
                        else:
                            if t == self.t_max-1:
                                value += belief_after_action[s_curr]*(-beta + self.mdp.gamma*self.V_actuator[s_curr,0] + immediate_reward[s_curr])
                            else: 
                                value += belief_after_action[s_curr]*(immediate_reward[s_curr] + self.mdp.gamma*self.V_actuator[s,t+1])
                    values[action] = value
                    
                self.pi_actuator[s,t] = np.argmax(values)
                if old_action != self.pi_actuator[s,t]:
                    policy_stable = False

                belief = np.dot(belief, self.mdp.P[self.pi_actuator[s,t]])
                belief = self.tbaa_actuator(belief,s,t)
        return policy_stable
    
    def policy_improvement_sensor(self, beta):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(self.t_max):
                # Update the policy of the sensor
                action_actuator = self.pi_actuator[s,t]
                if t != self.t_max-1:
                    action_actuator_blind = self.pi_actuator[s,t+1]
                baf = np.dot(belief, self.mdp.P[action_actuator])
                R = np.dot(baf, self.mdp.R[action_actuator])
                for s_curr in range(self.mdp.N_s):
                    old_c = self.pi_sensor[s,t,s_curr]

                    if t == self.t_max-1:
                        self.pi_sensor[s,t,s_curr] = 1
                    else:
                        values = np.zeros(2)
                        values[1] = (R[s_curr] -beta + self.mdp.gamma*self.V_sensor[s_curr,0].dot(self.mdp.P[self.pi_actuator[s_curr,0],s_curr]))
                        values[0] = (R[s_curr] + self.mdp.gamma*self.V_sensor[s,t+1].dot(self.mdp.P[action_actuator_blind,s_curr]))
                        self.pi_sensor[s,t,s_curr] = np.argmax(values)

                    if old_c != self.pi_sensor[s,t,s_curr]:
                        policy_stable = False

                belief = self.tbaa_sensor(baf,s,t)
        return policy_stable


    def tbaa_sensor(self,b,s,t):
        for s_curr in range(self.mdp.N_s):
            if t == self.t_max-1:
                b[s_curr] = 0
            elif self.pi_sensor[s,t,s_curr] == 1:
                b[s_curr] = 0
        return b/(np.sum(b)+1e-6)
    
    def tbaa_actuator(self,b,s,t):
        for s_curr in range(self.mdp.N_s):
            if t == self.t_max-1:
                b[s_curr] = b[s_curr]
            elif self.pi_sensor[s,t,s_curr] == 1:
                b[s_curr] = 0
        return b/(np.sum(b)+1e-6)

    def simulate(self,s,t,s_curr, beta):
        if t == self.t_max-1:
            return self.V_actuator[s_curr,0]
        action = self.pi_actuator[s,t+1]
        belief = self.mdp.P[action,s_curr]
        r = 0
        for s_next in range(self.mdp.N_s):
            c = self.pi_sensor[s,t+1,s_next]
            if c == 1:
                belief[s_next] = 0
                r += belief[s_next]*self.mdp.R[action,s_curr,s_next] + self.mdp.gamma*self.V_actuator[s_next,0]
            else:
                r += belief[s_next]*self.mdp.R[action,s_curr,s_next] + self.mdp.gamma*self.simulate(s,t+1,s_next)
        return r
    
    def eval_perf(self, n_episodes):
        states_visited = np.zeros(self.mdp.N_s)
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
                states_visited[s] += 1
        return tot_r/(n_episodes*self.H), tot_c/(n_episodes*self.H), states_visited/(n_episodes*self.H)
    
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