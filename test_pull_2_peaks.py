from EffCom.mdp import create_randomized_mdps, create_estimation_mdp
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

mdp_list = create_randomized_mdps(N_states=30,
                                  N_actions=4,
                                  gamma=0.99,
                                  r_seed=1234,
                                  reward_decay=[10,0.7])

mdp = mdp_list[idx]

# always transmits at the beginning
rpi = PolicyIterationWithSampling(mdp, 100, 100)
rpi.run(beta=beta)

print('Terminated')

# save the results

if not os.path.exists('results_pull_2_peaks'):
    os.makedirs('results_pull_2_peaks')
name_of_policy = 'results_pull_2_peaks/policy_' + str(idx) + '_' + str(beta) + '.npy'
np.save(name_of_policy, rpi.pi)