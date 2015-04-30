# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from nipytools import math as mymath

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
pred_dir = os.path.join(base_dir, 'multi-atlas', 'predicted_files')
#pred_dir = os.path.join(base_dir, 'gcss')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# load data
zstat_file = os.path.join(data_dir, 'merged_zstat.nii.gz')
label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
pred_file = os.path.join(pred_dir, 'merged_sts_pred.nii.gz')
zstat_data = nib.load(zstat_file).get_data()
label_data = nib.load(label_file).get_data()
pred_data = nib.load(pred_file).get_data()
pred_data = np.around(pred_data)

roi_label = [7, 8, 9, 10, 11, 12]
#roi_label = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]

for roi in roi_label:
    out_file = os.path.join(pred_dir, 'perf_' + str(roi) + '.txt')
    f = open(out_file, 'w')
    f.write('SID, peak, dice\n')

    for i in range(len(sessid)):
        tmp_pred = pred_data[..., i].copy()
        tmp_pred[tmp_pred!=roi] = 0
        tmp_pred[tmp_pred==roi] = 1
        tmp_label = label_data[..., i].copy()
        tmp_label[tmp_label!=roi] = 0
        tmp_label[tmp_label==roi] = 1
        pred_zstat = tmp_pred * zstat_data[..., i]
        true_zstat = tmp_label * zstat_data[..., i]
        if tmp_pred.sum() and tmp_label.sum():
            if np.argmax(pred_zstat) == np.argmax(true_zstat):
                c = 1
            else:
                c = 0
        elif tmp_pred.sum() or tmp_label.sum():
            c = 0
        else:
            c = 1
        dice = mymath.dice_coef(tmp_pred, tmp_label)
        f.write(','.join([sessid[i], str(c), str(dice)]) + '\n')

