import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
result_path = 'results_estimation_edoardo'

plt.figure()
data = pd.read_csv(result_path+'/results.csv')
data.columns = ['index', 'density', 'beta', 'r_a', 'r_n', 'c_a', 'c_n']

plt.plot(data['c_a'], data['r_a'], 'x', label='always')
plt.plot(data['c_n'], data['r_n'], label='never')
plt.xlabel('Cost')
plt.ylabel('Reward')

tikzplotlib.save(result_path+'/plot_all.tex')
plt.legend()
plt.savefig(result_path+'/plot_all.png')
plt.show()