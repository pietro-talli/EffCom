import numpy as np
import copy
import pickle
import csv
from EffCom.mdp import MDP
from EffCom.algorithms.BaseAlgorithm import RL_Algorithm
from EffCom.algorithms.push.POMDP import POMDP_solver
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
from utils import load_mdps, create_folder
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



    # T = np.array([[0.4,0.0,0.0,0.0,1.0],
    #               [0.6,0.4,0.0,0.0,0.0],
    #               [0.0,0.6,0.4,0.0,0.0],
    #               [0.0,0.0,0.6,0.4,0.0],
    #               [0.0,0.0,0.0,0.6,0.0]])
    # P = np.expand_dims(T.T, 0)
    # P = np.tile(P, (5,1,1))
    # R = np.tile(np.expand_dims(np.eye(5), axis=2), (1,1,5))
    # mdp = MDP(5, 5, P, R, 0.99)

    # horizon = 100

    # type of experiment
    experiment_types = ["push_sparse_reward", "push_spread_reward","estimation"]

    for experiment_type in experiment_types:
        # load mdps
        mdps, betas, params = load_mdps("configs/{}.json".format(experiment_type))
        horizon = params["horizon"]
        tmax = params["t_max"]
        runs_per_instance = params["runs_per_instance"]

        # create folder for the results
        results_folder = create_folder("results")

        # run experiments
        for mdp_i, mdp in enumerate(mdps):
            assert np.all(np.isclose(np.sum(mdp.P, axis=2), 1.0))

            pomdp = POMDP_solver(copy.deepcopy(mdp))
            policy_iteration = RemotePolicyIteration(copy.deepcopy(mdp), horizon, tmax)

            mdp_folder = create_folder("{}/{}".format(results_folder,str(mdp_i)))

            csv_file = "{}/{}_results.csv".format(mdp_folder,str(mdp_i))

            for beta in betas:
                pomdp.run(beta)
                tot_reward, tot_raw_reward, tot_reward_undiscounted, tot_raw_reward_undiscounted, tot_ch_uti, AoIs = pomdp.eval_perf(horizon, runs_per_instance)

                policy_iteration.run(beta)
                r,c = policy_iteration.eval_perf(runs_per_instance)
                aois_policy_iteration = []
                for state in range(mdp.n_states):
                    aois_policy_iteration.append(policy_iteration.get_PeakAoI(state, 10))

                with open("{}/{}_pomdp_aoi_{}".format(mdp_folder,str(mdp_i), beta), 'wb') as f:
                    pickle.dump(AoIs, f)
                with open("{}/{}_politer_aoi_{}".format(mdp_folder,str(mdp_i), beta), 'wb') as f:
                    pickle.dump(aois_policy_iteration, f)
                with open("{}/{}_results.csv".format(mdp_folder,str(mdp_i)), mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([tot_reward, tot_raw_reward, tot_reward_undiscounted, tot_raw_reward_undiscounted, tot_ch_uti, r, c])

            mdp_csv = "{}/{}".format(mdp_folder,str(mdp_i))


