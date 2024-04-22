from EffCom.mdp import create_randomized_mdps, create_estimation_mdp, create_randomized_mdps_estimation
from EffCom.algorithms.pull.policy_iteration import PolicyIterationWithSampling
import numpy as np

import os 
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--density', type=int)
argparser.add_argument('--beta', type=float)

args = argparser.parse_args()
idx = args.density
beta = args.beta

mdp_list = create_randomized_mdps_estimation(N_states=30,
                                             N_actions=30,
                                             gamma=0.9,
                                             r_seed=1234)

mdp = mdp_list[idx]

# always transmits at the beginning
rpi = PolicyIterationWithSampling(mdp, 100, 100)
rpi.run(beta=beta)

print('Terminated')

# save the results

if not os.path.exists('results_pull_estimation_30'):
    os.makedirs('results_pull_estimation_30')
name_of_policy = 'results_pull_estimation_30/policy_' + str(idx) + '_' + str(beta) + '.npy'
np.save(name_of_policy, rpi.pi)