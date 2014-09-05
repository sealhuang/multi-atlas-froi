# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
ma_dir = os.path.join(base_dir, 'multi-atlas')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

##-- ROI stats
## load label data
#label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
#label_data = nib.load(label_file).get_data()
#
#out_file = os.path.join(data_dir, 'subject_label_stats.txt')
#f = open(out_file, 'w')
#
#f.write('SID,rOFA,lOFA,rFFA,lFFA\n')
#for i in range(len(sessid)):
#    buf = [sessid[i]]
#    data = label_data[..., i]
#    for j in range(1, 5):
#        temp = data.copy()
#        temp[temp!=j] = 0
#        if temp.sum():
#            buf.append('1')
#        else:
#            buf.append('0')
#    f.write(','.join(buf) + '\n')

##-- merge predicted labels (left and right hemisphere)
#l_dir = os.path.join(ma_dir, 'l_ffa_ofa')
#r_dir = os.path.join(ma_dir, 'r_ffa_ofa')
#targ_dir = os.path.join(ma_dir, 'predicted_files')
#
#for subj in sessid:
#    l_file = os.path.join(l_dir, '50_atlas_pred', subj + '_pred.nii.gz')
#    r_file = os.path.join(r_dir, '50_atlas_pred', subj + '_pred.nii.gz')
#    targ = os.path.join(targ_dir, subj + '_pred.nii.gz')
#    os.system(' '.join(['fslmaths', l_file, '-add', r_file, targ]))
#
#merged_file = os.path.join(targ_dir, 'merged_pred.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid:
#    temp = os.path.join(targ_dir, subj + '_pred.nii.gz')
#    cmd_str.append(temp)
#os.system(' '.join(cmd_str))

