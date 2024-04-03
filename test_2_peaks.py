from EffCom.mdp import create_randomized_mdps, create_estimation_mdp
from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
import json
import numpy as np

from joblib import Parallel, delayed

config = json.load(open('configs/push_sparse_reward.json', 'r'))

mdp_list = create_randomized_mdps(N_states=30,
                                  N_actions=4,
                                  gamma=0.99,
                                  r_seed=1234,
                                  reward_decay=[10,0.7])

rpi = RemotePolicyIteration(mdp_list[2], 100, 10, 1)

rpi.run(beta=0.8)

print('Terminated')

rpi2 = RemotePolicyIteration(mdp_list[2], 100, 10, 0)

rpi2.run(beta=0.8)

print('Terminated')

r1,c1,ro_1 = rpi.eval_perf(1000)
r2,c2,ro_2 = rpi2.eval_perf(1000)

print(r1,c1)
print(r2,c2)

ro_1 += 1e-10
ro_2 += 1e-10

# TV distance
print(np.mean(np.abs(ro_1-ro_2)))

import matplotlib.pyplot as plt
plt.plot(ro_1, label='rpi')
plt.plot(ro_2, label='rpi2')
plt.plot(mdp_list[1].R[0,0,:]/15, label='R')
plt.legend()
plt.show()

print(np.array_equal(rpi.pi_actuator, rpi2.pi_actuator))
print(np.array_equal(rpi.pi_sensor, rpi2.pi_sensor))
