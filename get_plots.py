import numpy as np
import pandas as pd
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
from EffCom.algorithms.pull.policy_iteration import PolicyIterationWithSampling
from EffCom.mdp import MDP, create_randomized_mdps, create_estimation_mdp, create_randomized_mdps_estimation
import matplotlib.pyplot as plt
import os
import tikzplotlib
result_path = 'results_pull_estimation_30'

mdps = create_randomized_mdps(30,
                              4,
                              0.99,
                              1234,
                              [10,0.7])

#mdps = create_randomized_mdps_estimation(N_states=30, N_actions=30, gamma=0.99, r_seed=1234)

betas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
#betas = np.linspace(0.0, 2, 21)
for i, mdp in enumerate(mdps):
    for beta in betas:
        
        # rpi = RemotePolicyIteration(mdp, 100, 100, 1)
        # name_of_policy = result_path+'/policy_sensor_' + str(i) +'_' + str(beta) + '_always.npy'
        # rpi.pi_sensor = np.load(name_of_policy)
        # name_of_policy = result_path+'/policy_actuator_' + str(i) +'_' + str(beta) + '_always.npy'
        # rpi.pi_actuator = np.load(name_of_policy)
        # r,c,d = rpi.eval_perf(1000)

        # name_of_policy = result_path+'/policy_sensor_' + str(i) +'_' + str(beta) + '_never.npy'
        # rpi.pi_sensor = np.load(name_of_policy)
        # name_of_policy = result_path+'/policy_actuator_' + str(i) +'_' + str(beta) + '_never.npy'
        # rpi.pi_actuator = np.load(name_of_policy)
        # r2,c2,d2 = rpi.eval_perf(1000)

        # df = pd.DataFrame({'density': [i],
        #                     'beta': [beta],
        #                     'r_a': [r],
        #                     'r_n': [r2],
        #                     'c_a': [c],
        #                     'c_n': [c2],})
        # df.to_csv(result_path+'/results.csv', mode='a', header=False)
        # print('Done with density {} and beta {}'.format(i, beta))

        # if not os.path.exists(result_path+'/'+str(i)):
        #     os.makedirs(result_path+'/'+str(i))

        # plt.figure()
        # plt.plot(d, label='always')
        # plt.plot(d2, label='never')
        # plt.xlabel('States')
        # tikzplotlib.save(result_path+'/'+str(i)+'/plot_density_{}_beta_{}.tex'.format(i, beta))
        # plt.legend()
        # plt.savefig(result_path+'/'+str(i)+'/plot_density_{}_beta_{}.png'.format(i, beta))
        # plt.close()
        #     #print('Error with density {} and beta {}'.format(i, beta))
        rpi = PolicyIterationWithSampling(mdp, 100, 100)
        r,c = rpi.eval_perf(1000)
        df = pd.DataFrame({'density': [i],
                            'beta': [beta],
                            'r': [r],
                            'c': [c]})
        df.to_csv(result_path+'/results.csv', mode='a', header=False)
        print('Done with density {} and beta {}'.format(i, beta))
print('Done with all')

