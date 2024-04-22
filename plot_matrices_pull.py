import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib

df = pd.read_csv('results_pull_2_peaks/results.csv')

# add name of columns 
df.columns = ['index', 'density', 'beta', 'r', 'c']

reward = np.zeros((14,21))
cost = np.zeros((14,21))

betas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
for i in range(1,15):
    reward[i-1] = df[(df['density'] == i)]['r'].values
    cost[i-1] = df[(df['density'] == i)]['c'].values

plt.figure()
plt.imshow(reward, aspect='auto', origin='lower')
plt.savefig('results_pull_2_peaks/reward_pull_2_peaks.png')
tikzplotlib.save('results_pull_2_peaks/reward_pull_2_peaks.tex')
plt.colorbar()

plt.figure()
plt.imshow(cost, aspect='auto', origin='lower')
plt.savefig('results_pull_2_peaks/cost_pull_2_peaks.png')
tikzplotlib.save('results_pull_2_peaks/cost_pull_2_peaks.tex')
plt.colorbar()


df = pd.read_csv('results_pull_estimation_30/results.csv')

# add name of columns 
df.columns = ['index', 'density', 'beta', 'r', 'c']

reward = np.zeros((14,21))
cost = np.zeros((14,21))

betas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
for i in range(1,15):
    reward[i-1] = df[(df['density'] == i)]['r'].values
    cost[i-1] = df[(df['density'] == i)]['c'].values

plt.figure()
plt.imshow(reward, aspect='auto', origin='lower')
plt.savefig('results_pull_estimation_30/reward_pull_estimation_30.png')
tikzplotlib.save('results_pull_estimation_30/reward_pull_estimation_30.tex')
plt.colorbar()

plt.figure()
plt.imshow(cost, aspect='auto', origin='lower')
plt.savefig('results_pull_estimation_30/cost_pull_estimation_30.png')
tikzplotlib.save('results_pull_estimation_30/cost_pull_estimation_30.tex')
plt.colorbar()
