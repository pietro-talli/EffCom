import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from EffCom.algorithms.push.POMDP import POMDP_solver
from julia import NativeSARSOP, POMDPs

if __name__ == "__main__":
    P = np.random.uniform(0,1,size=(5,5,5))
    P = P / np.sum(P, axis = 2, keepdims=True)
    R = np.tile(np.expand_dims(np.eye(5), axis=2), (1,1,5))
    mdp = MDP(5, 5, P, R, 0.99)
    pomdp = POMDP_solver(mdp)

    pomdp.run(0.0)
    pomdp.eval_perf(100, 5)