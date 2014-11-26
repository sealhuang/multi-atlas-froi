# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import nibabel as nib
from sklearn.metrics import normalized_mutual_info_score

from nipytools import math as mymath
from nipytools import base as mybase

import lib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# load zstat and label file
zstat_file = os.path.join(data_dir, 'merged_zstat.nii.gz')
label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
mask_file = os.path.join(base_dir, 'multi-atlas', 'r_ofa_ffa', 'mask.nii.gz')
zstat_data = nib.load(zstat_file).get_data()
label_data = nib.load(label_file).get_data()
mask_data = nib.load(mask_file).get_data()
mask_shape = mask_data.shape
voxel_num = mask_shape[0] * mask_shape[1] * mask_shape[2]
mask_vtr = mask_data.reshape(voxel_num)
mask_vtr = mask_vtr > 0

selected_num = [1, 5] + range(10, 201, 10)
cls_list = [0, 1, 3]

ffa_dice = []
ofa_dice = []

# predict individual label
for i in range(len(sessid)):
    print 'Test subject %s'%(sessid[i])
    test_data = zstat_data[..., i].copy()
    test_label = label_data[..., i].copy()
    test_label = test_label * mask_data
    # calculate similarity
    test_vtr = test_data.copy()
    test_vtr[test_vtr<0] = 0
    test_vtr = test_vtr.reshape(voxel_num)
    test_data[test_data<2.3] = 0
    test_data[test_data>0] = 1
    similarity = []
    atlas_idx = []
    for j in range(len(sessid)):
        if i == j:
            continue
        atlas_idx.append(j)
        temp_data = zstat_data[..., j].copy()
        temp_data[temp_data<0] = 0
        temp_vtr = temp_data.reshape(voxel_num)
        #s = normalized_mutual_info_score(test_vtr[mask_vtr],
        #                                 temp_vtr[mask_vtr])
        s = np.corrcoef(test_vtr[mask_vtr], temp_vtr[mask_vtr])[0, 1]
        similarity.append(s)

    # sort the similarity and get n atlases
    sorted_sim_idx = np.argsort(similarity)[::-1]

    cls_list = [0, 1 ,3]
    temp_ffa_dice = []
    temp_ofa_dice = []
    for num in selected_num:
        print 'selected atlas number %s'%(num)
        selected_atlas = sorted_sim_idx[0:num]
        pred_prob = np.zeros((91, 109, 91, 3))
        for idx in selected_atlas:
            temp_label = label_data[..., atlas_idx[idx]]
            for j in range(3):
                temp = temp_label.copy()
                temp[temp!=cls_list[j]] = 100
                temp[temp==cls_list[j]] = 1
                temp[temp==100] = 0
                pred_prob[..., j] += temp
                #pred_prob[..., j] += temp * similarity[idx]
        pred_label = np.argmax(pred_prob, axis=3)
        pred_label[pred_label==2] = 3
        pred_label = pred_label * mask_data * test_data

        # compute Dice
        for label_idx in [1, 3]:
            P = pred_label == label_idx
            T = test_label == label_idx
            dice_val = mymath.dice_coef(T, P)
            print 'Dice for label %s: %f'%(label_idx, dice_val)
            if label_idx == 3:
                temp_ffa_dice.append(dice_val)
            else:
                temp_ofa_dice.append(dice_val)
    ffa_dice.append(temp_ffa_dice)
    ofa_dice.append(temp_ofa_dice)

out_file = 'ffa_output.txt'
f = open(out_file, 'w')
str_line = [str(item) for item in selected_num]
str_line = ','.join(str_line)
f.write(str_line + '\n')
for line in ffa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

out_file = 'ofa_output.txt'
f = open(out_file, 'w')
str_line = [str(item) for item in selected_num]
str_line = ','.join(str_line)
f.write(str_line + '\n')
for line in ofa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

