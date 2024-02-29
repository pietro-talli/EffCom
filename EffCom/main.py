import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from POMDP_model import POMDP_model
from julia import NativeSARSOP, POMDPs

if __name__ == "__main__":
    P = np.random.uniform((5,5,5))
    P = P / np.sum(P, axis = 2, keepdims=True)
    R = np.eye(5)
    mdp = MDP(5, 5, P, R, 0.99)
    pomdp = PODMP_solver(mdp)

    pomdp.run(0.0)
    pomdp.eval_perf(100, 5)