# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
ma_dir = os.path.join(base_dir, 'multi-atlas')
plot_dir = os.path.join(ma_dir, 'plot')

#-- figure 1 effect of atlas selection and number of selected atlas
# load data
rand_rofa_file = os.path.join(plot_dir, 'rand_rofa.csv')
rand_rffa_file = os.path.join(plot_dir, 'rand_rffa.csv')
rand_rofa = np.loadtxt(rand_rofa_file, delimiter=',')
rand_rffa = np.loadtxt(rand_rffa_file, delimiter=',')
r_data = np.zeros((rand_rofa.shape[0], rand_rofa.shape[1], 4))
r_data[..., 0] = rand_rofa
r_data[..., 1] = rand_rffa

ma_rofa_file = os.path.join(plot_dir, 'ma_rofa.csv')
ma_rffa_file = os.path.join(plot_dir, 'ma_rffa.csv')
ma_rofa = np.loadtxt(ma_rofa_file, delimiter=',')
ma_rffa = np.loadtxt(ma_rffa_file, delimiter=',')
r_data[..., 2] = np.tile(ma_rofa, (rand_rofa.shape[0], 1))
r_data[..., 3] = np.tile(ma_rffa, (rand_rofa.shape[0], 1))

# plot
sns.set(style='darkgrid')
step = pd.Series([1, 5]+range(10, 201, 10), name='number of atlas')
roi = pd.Series(["random-rOFA", 'random-rFFA', 'AS-rOFA', 'AS-rFFA'], name='ROI')
sns.tsplot(r_data, time=step, condition=roi, value='Dice similarity index')
plt.show()

