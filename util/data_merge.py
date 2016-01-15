# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
gss_dir = os.path.join(base_dir, 'ma_202', 'l_sts', '40_atlas_pred')
doc_dir = os.path.join(base_dir, 'doc')
#l_dir = os.path.join(gss_dir, 'l_fc')
#r_dir = os.path.join(gss_dir, 'r_fc')
#b_dir = os.path.join(gss_dir, 'fc')

sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#for subj in sessid:
#    l_file = os.path.join(l_dir, subj + '_40.nii.gz')
#    r_file = os.path.join(r_dir, subj + '_40.nii.gz')
#    out_file = os.path.join(b_dir, subj + '_40.nii.gz')
#    cmdstr = ['fslmaths', l_file, '-add', r_file, out_file]
#    os.system(' '.join(cmdstr))

merged_file = os.path.join(gss_dir, 'merged_data_40.nii.gz')
str_cmd = ['fslmerge', '-a', merged_file]
for subj in sessid:
    temp = os.path.join(gss_dir, subj + '_pred.nii.gz')
    str_cmd.append(temp)
os.system(' '.join(str_cmd))

