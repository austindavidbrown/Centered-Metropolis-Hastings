"""
ssh brow5079@compute.cla.umn.edu
#qsub -I -q gpu
qsub -I -l nodes=1:ppn=10
module load python/conda/3.7
source activate env
ipython
"""

from math import sqrt, pi, exp
import time

import torch

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

linewidth = 4
alpha = .8

plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

iterations = torch.arange(0, 1000, 1)

gammas = [.5, 1, 1.5, 2, 2.5]
colors = sns.color_palette("tab10")

for i in range(0, len(gammas)):
  gamma = gammas[i]
  color = colors[i]
  y = (1 - exp(-(1 + gamma**(1/2))**(2)))**(iterations)
  plt.plot(iterations, y, 
               label = r"$\gamma$ = {}".format(gamma),
               alpha = alpha,
               color = color,
               linewidth = linewidth)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Iterations", fontsize = 25, color="black")
plt.ylabel(r"Decrease in Wasserstein distance", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("decrease_plot.png", pad_inches=0, bbox_inches='tight',)

