from EffCom.mdp import create_randomized_mdps, create_estimation_mdp
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
import json
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
rpi = RemotePolicyIteration(mdp, 100, 100, 1)
rpi.run(beta=beta)

print('Terminated')

# save the results

name_of_policy = 'results/policy_sensor_' + str(idx) + '_' + str(beta) + '_always.npy'
np.save(name_of_policy, rpi.pi_sensor)
name_of_policy = 'results/policy_actuator_' + str(idx) + '_' + str(beta) + '_always.npy'
np.save(name_of_policy, rpi.pi_actuator)


# never transmits at the beginning
rpi = RemotePolicyIteration(mdp, 100, 100, 0)
rpi.run(beta=beta)

print('Terminated')

# save the results

name_of_policy = 'results/policy_sensor_' + str(idx) + '_' + str(beta) + '_never.npy'
np.save(name_of_policy, rpi.pi_sensor)
name_of_policy = 'results/policy_actuator_' + str(idx) + '_' + str(beta) + '_never.npy'
np.save(name_of_policy, rpi.pi_actuator)