# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from nipytools import math as mymath

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
parcel_file = os.path.join(gcss_dir, 'face_parcel.nii.gz')
act_data = nib.load(act_file).get_data()
label_data = nib.load(label_file).get_data()
parcel_data = nib.load(parcel_file).get_data()

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
roi_dict = {'rofa': 1,
            'lofa': 2,
            'rffa': 3,
            'lffa': 4,
            'rpcsts': 7,
            'lpcsts': 8,
            'rpsts': 9,
            'lpsts': 10,
            'rasts': 11,
            'lasts': 12}

for roi in roi_dict:
    roi_idx = roi_dict[roi]
    roi_dice = []

    roi_mask = parcel_data.copy()
    roi_mask[roi_mask!=roi_idx] = 0
    roi_mask[roi_mask==roi_idx] = 1
    label = label_data.copy()
    label[label!=roi_idx] = 0
    label[label==roi_idx] = 1

    for i in range(act_data.shape[3]):
        pred_roi = act_data[..., i] * roi_mask
        dice = mymath.dice_coef(pred_roi, label[..., i])
        roi_dice.append(dice)
    print 'Mean Dice - %s: %s'%(roi, np.array(roi_dice).mean())

    out_file = os.path.join(roi + '_dice_output.txt')
    f = open(out_file, 'w')
    for line in roi_dice:
        f.write(str(line) + '\n')

