from EffCom.mdp import create_randomized_mdps_estimation
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
import numpy as np
import os

mdp_list = create_randomized_mdps_estimation(30, 30, 0.99, 1234)


import numpy as np

import os 
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--density', type=int)
argparser.add_argument('--beta', type=float)

args = argparser.parse_args()
idx = args.density
beta = args.beta

mdp = mdp_list[idx]

# always transmits at the beginning
rpi = RemotePolicyIteration(mdp, 100, 100, 1)
rpi.run(beta=beta)

print('Terminated')

path_to_save = 'results_estimation_30'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

# save the results

name_of_policy = path_to_save+'/policy_sensor_' + str(idx) + '_' + str(beta) + '_always.npy'
np.save(name_of_policy, rpi.pi_sensor)
name_of_policy = path_to_save+'/policy_actuator_' + str(idx) + '_' + str(beta) + '_always.npy'
np.save(name_of_policy, rpi.pi_actuator)


# never transmits at the beginning
rpi = RemotePolicyIteration(mdp, 100, 100, 0)
rpi.run(beta=beta)

print('Terminated')

# save the results

name_of_policy = path_to_save+'/policy_sensor_' + str(idx) + '_' + str(beta) + '_never.npy'
np.save(name_of_policy, rpi.pi_sensor)
name_of_policy = path_to_save+'/policy_actuator_' + str(idx) + '_' + str(beta) + '_never.npy'
np.save(name_of_policy, rpi.pi_actuator)