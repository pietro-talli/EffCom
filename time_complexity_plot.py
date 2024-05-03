import numpy as np

# plot the results
import matplotlib.pyplot as plt
import csv

# load data from csv
time_pull = []
time_push = []
with open("pull_time.csv", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        time_pull.append([float(x) for x in row])
with open("push_time.csv", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        time_push.append([float(x) for x in row])

time_pull = np.array(time_pull)
time_push = np.array(time_push)

time_pull = np.log10(time_pull)
time_push = np.log10(time_push)

mean_pull = np.mean(time_pull,axis=1)
mean_push = np.mean(time_push,axis=1)
std_pull = np.std(time_pull,axis=1)
std_push = np.std(time_push,axis=1)

max_n = 10

plt.plot(range(1,max_n+1), mean_pull, label="Pull")
plt.fill_between(range(1,max_n+1), mean_pull-std_pull, mean_pull+std_pull, alpha=0.2)
plt.plot(range(1,max_n+1), mean_push, label="Push")
plt.fill_between(range(1,max_n+1), mean_push-std_push, mean_push+std_push, alpha=0.2)

import tikzplotlib
tikzplotlib.save("figs/time_complexity.tex")

plt.show()

