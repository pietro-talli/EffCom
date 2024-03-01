import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from EffCom.algorithms.push.POMDP import POMDP_solver

if __name__ == "__main__":
    T = np.array([[0.4,0.0,0.0,0.0,1.0],
                  [0.6,0.4,0.0,0.0,0.0],
                  [0.0,0.6,0.4,0.0,0.0],
                  [0.0,0.0,0.6,0.4,0.0],
                  [0.0,0.0,0.0,0.6,0.0]])
    P = np.expand_dims(T.T, 0)
    P = np.tile(P, (5,1,1))
    R = np.tile(np.expand_dims(np.eye(5), axis=2), (1,1,5))
    mdp = MDP(5, 5, P, R, 0.99)
    pomdp = POMDP_solver(mdp)

    pomdp.run(0.9)
    pomdp.eval_perf(100, 5)