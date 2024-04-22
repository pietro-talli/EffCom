import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib

df = pd.read_csv('results_estimation_30/results.csv')

# add name of columns 
df.columns = ['index', 'density', 'beta', 'r_a', 'r_n', 'c_a', 'c_n']

reward_always = np.zeros((14,21))
reward_never = np.zeros((14,21))
cost_always = np.zeros((14,21))
cost_never = np.zeros((14,21))

betas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
for i in range(1,15):
    reward_always[i-1] = df[(df['density'] == i)]['r_a'].values
    reward_never[i-1] = df[(df['density'] == i)]['r_n'].values
    cost_always[i-1] = df[(df['density'] == i)]['c_a'].values
    cost_never[i-1] = df[(df['density'] == i)]['c_n'].values

plt.figure()
plt.subplot(2,2,1)
plt.imshow(reward_always, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.subplot(2,2,2)
plt.imshow(reward_never, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.subplot(2,2,3)
plt.imshow(cost_always, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.subplot(2,2,4)
plt.imshow(cost_never, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

reward_star = np.zeros((14,21))
cost_star = np.zeros((14,21))
idx_star = np.zeros((14,21))
gap = np.zeros((14,21))

for i in range(1,15):
    for j, beta in enumerate(betas):
        phi_a = reward_always[i-1,j] - beta*cost_always[i-1,j]
        phi_n = reward_never[i-1,j] - beta*cost_never[i-1,j]
        delta = np.abs(phi_a - phi_n)
        if delta < 1e-3:
            reward_star[i-1,j] = reward_always[i-1,j]
            cost_star[i-1,j] = cost_always[i-1,j]
            idx_star[i-1,j] = 0
            gap[i-1,j] = 0
        elif phi_a > phi_n:
            reward_star[i-1,j] = reward_always[i-1,j]
            cost_star[i-1,j] = cost_always[i-1,j]
            idx_star[i-1,j] = 1
            gap[i-1,j] = phi_a - phi_n
        elif phi_a < phi_n:
            reward_star[i-1,j] = reward_never[i-1,j]
            cost_star[i-1,j] = cost_never[i-1,j]
            idx_star[i-1,j] = -1
            gap[i-1,j] = phi_n - phi_a

plt.figure()
plt.imshow(reward_star, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')
plt.savefig('figs/estimation_30_reward_star.png')
tikzplotlib.save('figs/estimation_30_reward_star.tex')

plt.figure()
plt.imshow(cost_star, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')
plt.savefig('figs/estimation_30_cost_star.png')
tikzplotlib.save('figs/estimation_30_cost_star.tex')

plt.figure()
plt.imshow(idx_star, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')
plt.savefig('figs/estimation_30_idx_star.png')
tikzplotlib.save('figs/estimation_30_idx_star.tex')

plt.figure()
plt.imshow(gap, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')
plt.savefig('figs/estimation_30_gap.png')
tikzplotlib.save('figs/estimation_30_gap.tex')

plt.show()
