# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from nipytools import math as mymath
from nipytools import base as mybase

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
#data_dir = os.path.join(base_dir, 'ma_202', 'data')
data_dir = os.path.join(base_dir, 'ma_202', 'r_fc', 'posterior_maps')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# load label data
label_file = os.path.join(data_dir, 'merged_label.nii.gz')
label_data = nib.load(label_file).get_data()

roi_index = 3
sim_mtx = []

for i in range(len(sessid)):
    tmp_dice = []
    labeli = label_data[..., i].copy()
    for j in range(len(sessid)):
        labelj = label_data[..., j].copy()
        # compute Dice
        P = labeli == roi_index
        T = labelj == roi_index
        dice_val = mymath.dice_coef(T, P)
        tmp_dice.append(dice_val)
    sim_mtx.append(tmp_dice)

out_file = 'dice_sim_output.txt'
f = open(out_file, 'w')
for line in sim_mtx:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')


