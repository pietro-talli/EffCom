import numpy as np
from EffCom.mdp import MDP
from EffCom.algorithms.pull.policy_iteration import PolicyIterationWithSampling


R = np.tile(np.expand_dims(np.eye(10),axis=2), (1,1,10))

P = np.array( [[0.4,
0.6,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0],

                  [0.0,
0.4,
0.6,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0],

                  [0.0,
0.0,
0.4,
0.6,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0],

                  [0.0,
0.0,
0.0,
0.4,
0.6,
0.0,
0.0,
0.0,
0.0,
0.0],

                  [0.0,
0.0,
0.0,
0.0,
0.4,
0.6,
0.0,
0.0,
0.0,
0.0],

                  [0.0,
0.0,
0.0,
0.0,
0.0,
0.4,
0.6,
0.0,
0.0,
0.0],

                  [0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.4,
0.6,
0.0,
0.0],

                  [0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.4,
0.6,
0.0],

                  [0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.4,
0.6],

                  [1.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0]] )

P = np.tile(np.expand_dims(P,axis=0),(10,1,1))
gamma = 0.99
mdp = MDP(10,10,P,R,gamma)

betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 ,1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

result_r = []
result_c = []

from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration

for beta in betas:
    print("Beta: ", beta)
    pi = RemotePolicyIteration(mdp, 100, 100, 0)
    pi.run(beta)
    r,c,_ = pi.eval_perf(1000)
    result_r.append(r)
    result_c.append(c)

# svae results on csv
import csv
with open('results_4_edo_push_always.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Beta", "Reward", "Cost"])
    for i in range(len(betas)):
        writer.writerow([betas[i], result_r[i], result_c[i]])


result_r = []
result_c = []

from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration

for beta in betas:
    print("Beta: ", beta)
    pi = RemotePolicyIteration(mdp, 100, 100, 1)
    pi.run(beta)
    r,c,_ = pi.eval_perf(1000)
    result_r.append(r)
    result_c.append(c)

# svae results on csv
import csv
with open('results_4_edo_push_never.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Beta", "Reward", "Cost"])
    for i in range(len(betas)):
        writer.writerow([betas[i], result_r[i], result_c[i]])
