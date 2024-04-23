from EffCom.mdp import create_randomized_mdps, create_randomized_mdps_estimation
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
import numpy as np

import os

result_path = 'results_estimation_30'

mdps = create_randomized_mdps(30,
                                4,
                                0.9,
                                1234,
                                [10,0.7])

mdps = create_randomized_mdps_estimation(N_states=30, N_actions=30, gamma=0.99, r_seed=1234)

rpi = RemotePolicyIteration(mdps[1], 100, 100)
beta = 1.5
rpi.pi_sensor = np.load(result_path+'/policy_sensor_' + str(1) +'_' + str(beta) + '_never.npy')
rpi.pi_actuator = np.load(result_path+'/policy_actuator_' + str(1) +'_' + str(beta) + '_never.npy')
final_mat = np.zeros((20,30))

for s in range(30):
    PAOI = rpi.get_PeakAoI(s)
    # obtain a distribution of the AoI
    dist_aoi = np.bincount(PAOI, minlength=20)
    # normalize the distribution
    dist_aoi = dist_aoi / np.sum(dist_aoi)

    print('State: {}'.format(s))
    print(dist_aoi)
    final_mat[:,s] = dist_aoi

np.save(result_path+'/AoI.npy', final_mat)

import matplotlib.pyplot as plt
import tikzplotlib
plt.figure()
plt.imshow(final_mat, aspect='auto', origin='lower')
plt.colorbar()
plt.savefig(result_path+'/AoI.png')
tikzplotlib.save(result_path+'/AoI.tex')
plt.show()