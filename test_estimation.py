from EffCom.mdp import create_estimation_mdp
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
import numpy as np
import os

mdp = create_estimation_mdp(gamma=0.99)[0]

betas = np.linspace(0.0, 2, 21)

save_path = 'results_estimation_edoardo'

if not os.path.exists(save_path):
    os.makedirs(save_path)
for beta in betas:
    # always transmits at the beginning
    rpi = RemotePolicyIteration(mdp, 100, 100, 1)
    rpi.run(beta=beta)

    print('Terminated')

    # save the results

    name_of_policy = save_path+'/policy_sensor_' + str(beta) + '_always.npy'
    np.save(name_of_policy, rpi.pi_sensor)
    name_of_policy = save_path+'/policy_actuator_' + str(beta) + '_always.npy'
    np.save(name_of_policy, rpi.pi_actuator)


    # never transmits at the beginning
    rpi = RemotePolicyIteration(mdp, 100, 100, 0)
    rpi.run(beta=beta)

    print('Terminated')

    # save the results

    name_of_policy = save_path+'/policy_sensor_' + str(beta) + '_never.npy'
    np.save(name_of_policy, rpi.pi_sensor)
    name_of_policy = save_path+'/policy_actuator_' + str(beta) + '_never.npy'
    np.save(name_of_policy, rpi.pi_actuator)