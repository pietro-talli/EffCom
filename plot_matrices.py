import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib

df = pd.read_csv('results/results.csv')

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
plt.imshow(reward_always, aspect='auto')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.figure()
plt.imshow(reward_never, aspect='auto')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.figure()
plt.imshow(cost_always, aspect='auto')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.figure()
plt.imshow(cost_never, aspect='auto')
plt.colorbar()
plt.xlabel('Beta')
plt.ylabel('Density')

plt.show()
