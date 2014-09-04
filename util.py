# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# load label data
label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
label_data = nib.load(label_file).get_data()

out_file = os.path.join(data_dir, 'subject_label_stats.txt')
f = open(out_file, 'w')

f.write('SID,rOFA,lOFA,rFFA,lFFA\n')
for i in range(len(sessid)):
    buf = [sessid[i]]
    data = label_data[..., i]
    for j in range(1, 5):
        temp = data.copy()
        temp[temp!=j] = 0
        if temp.sum():
            buf.append('1')
        else:
            buf.append('0')
    f.write(','.join(buf) + '\n')

