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
linestyles = [(0, (1, 1)), # densely dotted,
              (0, (3, 1, 1, 1)), # densely dashdotted
              (0, (3, 1, 1, 1, 1, 1)), # densely dashdotteddotted
              (0, (5, 1)), # densely dashed 
              "solid"
              ]

for i in range(0, len(gammas)):
  gamma = gammas[i]
  color = colors[i]
  linestyle = linestyles[i]
  y = (1 - exp(-(1 + gamma**(1/2))**(2)))**(iterations)
  plt.plot(iterations, y, 
               label = r"$\gamma$ = {}".format(gamma),
               alpha = alpha,
               color = color,
               linestyle = linestyle,
               linewidth = linewidth)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Iterations", fontsize = 25, color="black")
plt.ylabel(r"Decrease in Wasserstein distance", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("decrease_plot.eps", 
            format='eps', 
            pad_inches=0, 
            bbox_inches='tight')

'''
from PIL import Image
Image.open('decrease_plot2.png').convert('L').save('decrease_plot2.png')
'''