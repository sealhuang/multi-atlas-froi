# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from mypy import base as mybase
from froi.algorithm import imtool

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
ma_dir = os.path.join(base_dir, 'multi-atlas', 'predicted_files')
gcss_dir = os.path.join(base_dir, 'gcss')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# load label and zstat file
label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
label_data = nib.load(label_file).get_data()
header = nib.load(label_file).get_header()

zstat_file = os.path.join(data_dir, 'merged_zstat.nii.gz')
zstat_data = nib.load(zstat_file).get_data()

roi_list = [1, 2, 3, 4]

out_dir = os.path.join(data_dir, 'peak_mask')

for i in range(len(sessid)):
    ind_label = label_data[..., i].copy()
    ind_label = np.around(ind_label)
    ind_zstat = zstat_data[..., i].copy()
    peak_label = np.zeros(ind_label.shape)
    for roi in roi_list:
        tmp = ind_label.copy()
        tmp[tmp!=roi] = 0
        tmp[tmp==roi] = 1
        if not tmp.sum():
            continue
        tmp_zstat = tmp * ind_zstat
        peak_coord = np.unravel_index(tmp_zstat.argmax(), tmp_zstat.shape)
        tmp_peak = imtool.cube_roi(peak_label, peak_coord[0],
                                   peak_coord[1], peak_coord[2],
                                   3, roi)
        peak_label += tmp_peak * tmp
    out_file = os.path.join(out_dir, sessid[i] + '_atlas.nii.gz')
    mybase.save2nifti(peak_label, header, out_file)

