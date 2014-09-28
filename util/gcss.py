# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from mypy import math as mymath

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
gcss_dir = os.path.join(base_dir, 'gcss')

## read all subjects' SID
#sessid_file = os.path.join(doc_dir, 'sessid')
#sessid = open(sessid_file).readlines()
#sessid = [line.strip() for line in sessid]

# load files
act_file = os.path.join(gcss_dir, 'bin_zstat.nii.gz')
label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
parcel_file = os.path.join(gcss_dir, 'parcel_thr_01.nii.gz')
prob_file = os.path.join(gcss_dir, 'sm_prob_thr_01.nii.gz')
act_data = nib.load(act_file).get_data()
label_data = nib.load(label_file).get_data()
parcel_data = nib.load(parcel_file).get_data()
prob_data = nib.load(prob_file).get_data()

prob_thr = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

## activation prob across subjects in parcel
#print 'threshold, roi, size, ratio'
#for thr in prob_thr:
#    prob = prob_data.copy()
#    prob[prob<thr] = 0
#    prob[prob>0] = 1
#    thr_parcel = prob * parcel_data
#    for label in [1, 2, 3, 4]:
#        roi_mask = thr_parcel.copy()
#        roi_mask[roi_mask!=label] = 0
#        roi_mask[roi_mask==label] = 1
#        roi_size = roi_mask.sum()
#        count = 0
#        for i in range(act_data.shape[3]):
#            temp = act_data[..., i] * roi_mask
#            if temp.sum():
#                count += 1
#        ratio = float(count) / act_data.shape[3]
#        if label == 1:
#            print '%s, %s, %s, %s'%(thr, 'rOFA', roi_size, ratio)
#        elif label == 3:
#            print '%s, %s, %s, %s'%(thr, 'rFFA', roi_size, ratio)
#        if label == 2:
#            print '%s, %s, %s, %s'%(thr, 'lOFA', roi_size, ratio)
#        elif label == 4:
#            print '%s, %s, %s, %s'%(thr, 'lFFA', roi_size, ratio)

# compute Dice in each probability threshold
rffa_dice = []
rofa_dice = []
lffa_dice = []
lofa_dice = []
for thr in prob_thr:
    print 'Threshold %s'%(thr)
    tmp_rffa_dice = []
    tmp_rofa_dice = []
    tmp_lffa_dice = []
    tmp_lofa_dice = []
    prob = prob_data.copy()
    prob[prob<thr] = 0
    prob[prob>0] = 1
    thr_parcel = prob * parcel_data
    for l in [1, 2, 3, 4]:
        roi_mask = thr_parcel.copy()
        roi_mask[roi_mask!=l] = 0
        roi_mask[roi_mask==l] = 1
        label = label_data.copy()
        label[label!=l] = 0
        label[label==l] = 1
        for i in range(act_data.shape[3]):
            pred_roi = act_data[..., i] * roi_mask
            dice = mymath.dice_coef(pred_roi, label[..., i])
            if l == 1:
                tmp_rofa_dice.append(dice)
            elif l == 2:
                tmp_lofa_dice.append(dice)
            elif l == 3:
                tmp_rffa_dice.append(dice)
            else:
                tmp_lffa_dice.append(dice)
    print 'rFFA Dice %s'%(np.array(tmp_rffa_dice).mean())
    print 'rOFA Dice %s'%(np.array(tmp_rofa_dice).mean())
    print 'lFFA Dice %s'%(np.array(tmp_lffa_dice).mean())
    print 'lOFA Dice %s'%(np.array(tmp_lofa_dice).mean())
    rffa_dice.append(tmp_rffa_dice) 
    rofa_dice.append(tmp_rofa_dice) 
    lffa_dice.append(tmp_lffa_dice) 
    lofa_dice.append(tmp_lofa_dice) 

out_file = os.path.join(gcss_dir, 'rffa_output.txt')
f = open(out_file, 'w')
for line in rffa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

out_file = os.path.join(gcss_dir, 'rofa_output.txt')
f = open(out_file, 'w')
for line in rofa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

out_file = os.path.join(gcss_dir, 'lffa_output.txt')
f = open(out_file, 'w')
for line in lffa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

out_file = os.path.join(gcss_dir, 'lofa_output.txt')
f = open(out_file, 'w')
for line in lofa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

