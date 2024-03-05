import numpy as np
import copy
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from EffCom.algorithms.push.POMDP import POMDP_solver
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
from julia import Pkg

if __name__ == "__main__":
    # Activate the current Julia environment
    Pkg.activate(".")

    # Add or update package dependencies
    Pkg.add("POMDPs")
    Pkg.add("POMDPTools")
    Pkg.add(url="https://github.com/ed13santi/NativeSARSOP.jl")

    # Resolve and instantiate the environment
    Pkg.instantiate()



    T = np.array([[0.4,0.0,0.0,0.0,1.0],
                  [0.6,0.4,0.0,0.0,0.0],
                  [0.0,0.6,0.4,0.0,0.0],
                  [0.0,0.0,0.6,0.4,0.0],
                  [0.0,0.0,0.0,0.6,0.0]])
    P = np.expand_dims(T.T, 0)
    P = np.tile(P, (5,1,1))
    R = np.tile(np.expand_dims(np.eye(5), axis=2), (1,1,5))
    mdp = MDP(5, 5, P, R, 0.99)

    horizon = 100

    pomdp = POMDP_solver(copy.deepcopy(mdp))
    policy_iteration = RemotePolicyIteration(mdp, horizon, 1000)

    betas = [0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    n_episodes = 10

    for beta in betas:
        pomdp.run(beta)
        pomdp.eval_perf(100, n_episodes)

        policy_iteration.run(beta)
        r,c = policy_iteration.eval_perf(n_episodes)
        print(r)
        print(c)
        print("----")