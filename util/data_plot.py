# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
ma_dir = os.path.join(base_dir, 'multi-atlas')
plot_dir = os.path.join(ma_dir, 'plot')

##-- fig. 1 relation between atlas rank and its dice accuracy
## load data
#roi_name = 'rasts'
#data_file = os.path.join(plot_dir, 'atlas_rank', roi_name + '_atlas_rank.csv')
#data = np.loadtxt(data_file, delimiter=',')
#fig, ax = plt.subplots()
#ax.plot(np.arange(1, 201, 1), data, 'o', markersize=5)
#ax.set_xlim(0, 210)
#ax.set_title('right aSTS')
#ax.set_xlabel('Rank of atlas')
#ax.set_ylabel('Dice overlap')
#plt.show()

#-- fig. 2 effect of atlas selection and number of selected atlas
# load data
roi_name = 'lofa'
data_file = os.path.join(plot_dir, 'atlas_selection', roi_name + '_comp.csv')
data = np.loadtxt(data_file, delimiter=',')
# plot
fig, ax=plt.subplots()
plt.plot(data[0, :], data[1, :], '-o', ms=6, lw=2, alpha=0.7, mfc='blue')
plt.plot(data[0, :], data[2, :], '-o', ms=6, lw=2, alpha=0.7, mfc='green')
ax.set_xlim(0, 210)
#ax.set_ylim(0.4, 0.75)
ax.set_title('left OFA')
ax.set_xlabel('Number fused')
ax.set_ylabel('Dice overlap')
plt.show()

##-- fig. 2 effect of forest parameter
## load data
#data_dir = os.path.join(plot_dir, 'tree_parameter')
#rofa_file = os.path.join(data_dir, 'rofa_para.csv')
#rffa_file = os.path.join(data_dir, 'rffa_para.csv')
#rofa = np.loadtxt(rofa_file, delimiter=',')
#rffa = np.loadtxt(rffa_file, delimiter=',')
#data = (rofa + rffa)/2
## plot 3D fig
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#depth_idx, ntree_idx = np.mgrid[10:31:5, 10:51:5]
#print depth_idx.shape
#print ntree_idx.shape
#surf = ax.plot_surface(ntree_idx, depth_idx, data,
#                       cmap=cm.coolwarm, rstride=1, cstride=1)
#ax.set_xlabel('depth')
#ax.set_ylabel('# of trees')
#ax.set_zlabel('Dice value')
#ax.set_zlim3d(0.78, 0.83)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

##-- fig. 3 comparison between random and similarity selection
## load data
#data_file = os.path.join(plot_dir, 'random_50.csv')
#d = np.loadtxt(data_file, delimiter=',')
## plot
#sns.set(style='whitegrid')
#f, ax = plt.subplots()
#sns.violinplot(d, names=['rOFA', 'rFFA'])
#sns.despine(left=True)
#ax.set_ylim(0.7, 0.87)
#ax.set_ylabel('Dice value')
#ax.plot(1, 0.75, color='cornflowerblue', marker='o', markersize=10)
#ax.plot(2, 0.85, color='cornflowerblue', marker='o', markersize=10)
#plt.show()
