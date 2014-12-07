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

#-- fig. 1 relation between atlas rank and its dice accuracy
# load data
roi_name = 'rasts'
data_file = os.path.join(plot_dir, 'atlas_rank', roi_name + '_atlas_rank.csv')
data = np.loadtxt(data_file, delimiter=',')
fig, ax = plt.subplots()
ax.plot(np.arange(1, 201, 1), data, 'o', markersize=5)
ax.set_xlim(0, 210)
ax.set_title('right aSTS')
ax.set_xlabel('Rank of atlas')
ax.set_ylabel('Dice overlap')
plt.show()

##-- fig. 1 effect of atlas selection and number of selected atlas
## load data
#rand_rofa_file = os.path.join(plot_dir, 'rand_rofa.csv')
#rand_rffa_file = os.path.join(plot_dir, 'rand_rffa.csv')
#rand_rofa = np.loadtxt(rand_rofa_file, delimiter=',')
#rand_rffa = np.loadtxt(rand_rffa_file, delimiter=',')
#r_data = np.zeros((rand_rofa.shape[0], rand_rofa.shape[1], 4))
#r_data[..., 0] = rand_rofa
#r_data[..., 1] = rand_rffa
#
#ma_rofa_file = os.path.join(plot_dir, 'ma_rofa.csv')
#ma_rffa_file = os.path.join(plot_dir, 'ma_rffa.csv')
#ma_rofa = np.loadtxt(ma_rofa_file, delimiter=',')
#ma_rffa = np.loadtxt(ma_rffa_file, delimiter=',')
#r_data[..., 2] = np.tile(ma_rofa, (rand_rofa.shape[0], 1))
#r_data[..., 3] = np.tile(ma_rffa, (rand_rofa.shape[0], 1))
#
## plot
#sns.set(style='darkgrid')
#step = pd.Series([1, 5]+range(10, 201, 10), name='number of atlas')
#roi = pd.Series(["random-rOFA", 'random-rFFA', 'AS-rOFA', 'AS-rFFA'], name='condition')
#sns.tsplot(r_data, time=step, condition=roi, value='Dice index')
#plt.show()

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
