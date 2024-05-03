import numpy as np
import time 

from EffCom.mdp import MDP, create_randomized_mdps_estimation, create_randomized_mdps
from EffCom.algorithms.pull.policy_iteration import PolicyIterationWithSampling
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration

betas = np.linspace(0,2,100)

max_n = 10

time_pull = np.zeros((max_n,len(betas)))
time_push = np.zeros((max_n,len(betas)))

for i in range(max_n):
    dim = i+2
    mdp_list = create_randomized_mdps(dim, 4, 0.99, r_seed=1234)
    print(mdp_list)
    if dim <= 3: mdp = mdp_list[0]
    else: mdp = mdp_list[1]
    for j, beta in enumerate(betas):
        pi = PolicyIterationWithSampling(mdp,100,100)
        start = time.time()
        pi.run(beta)
        end = time.time()
        time_pull[i,j] = end-start
        pi = RemotePolicyIteration(mdp,100,100)
        start = time.time()
        pi.run(beta)
        end = time.time()
        time_push[i,j] = end-start

# get mean time nd std
mean_pull = np.mean(time_pull,axis=1)
mean_push = np.mean(time_push,axis=1)
std_pull = np.std(time_pull,axis=1)
std_push = np.std(time_push,axis=1)

# save data in csv 
np.savetxt("pull_time.csv", time_pull, delimiter=",")
np.savetxt("push_time.csv", time_push, delimiter=",")
