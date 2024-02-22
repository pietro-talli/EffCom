# Abstract class for RL algorithms
from EffCom.mdp import MDP

class RL_Algorithm():
    def __init__(self, MDP: MDP):
        self.mdp = MDP
        
    def run(self):
        pass

    def eval_perf(self):
        pass