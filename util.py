# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from mypy import base as mybase

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
ma_dir = os.path.join(base_dir, 'multi-atlas')
gcss_dir = os.path.join(base_dir, 'gcss')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

##-- ROI stats
## load label data
##label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
##label_file = os.path.join(gcss_dir, 'merged_pred.nii.gz')
#label_file = os.path.join(ma_dir, 'predicted_files', 'merged_pred.nii.gz')
#label_data = nib.load(label_file).get_data()
#
##out_file = os.path.join(data_dir, 'subject_label_stats.txt')
##out_file = os.path.join(gcss_dir, 'subject_label_stats.txt')
#out_file = os.path.join(ma_dir, 'predicted_files', 'subject_label_stats.txt')
#f = open(out_file, 'w')
#
#f.write('SID,rOFA,lOFA,rFFA,lFFA\n')
#for i in range(len(sessid)):
#    buf = [sessid[i]]
#    data = label_data[..., i]
#    for j in range(1, 5):
#        temp = data.copy()
#        temp = np.around(temp)
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

##-- split an individual atlas into several roi mask
## load label data
##label_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
##label_file = os.path.join(gcss_dir, 'merged_pred.nii.gz')
#label_file = os.path.join(ma_dir, 'predicted_files', 'merged_pred.nii.gz')
#label_data = nib.load(label_file).get_data()
#header = nib.load(label_file).get_header()
#
##out_dir = os.path.join(data_dir, 'roi_mask')
##out_dir = os.path.join(gcss_dir, 'roi_mask')
#out_dir = os.path.join(ma_dir, 'predicted_files', 'roi_mask')
#
#for i in range(len(sessid)):
#    ind_atlas = label_data[..., i].copy()
#    ind_atlas = np.around(ind_atlas)
#    ind_atlas[ind_atlas>4] = 0
#    out_file = os.path.join(out_dir, sessid[i] + '_atlas.nii.gz')
#    mybase.save2nifti(ind_atlas, header, out_file)

#-- compute rsfc
atlas_dir = os.path.join(data_dir, 'peak_mask_1')
#atlas_dir = os.path.join(gcss_dir, 'peak_mask')
#atlas_dir = os.path.join(ma_dir, 'predicted_files', 'peak_mask')
rsfc_dir = os.path.join(atlas_dir, 'rsfc')
sessid_file = os.path.join(rsfc_dir, 'sessid')

## extract time courses
#for i in range(len(sessid)):
#    f = open(sessid_file, 'w')
#    f.write(sessid[i])
#    f.close()
#    atlas_file = os.path.join(atlas_dir, sessid[i] + '_atlas.nii.gz')
#    cmd_str = ['extract-roi-tc', '-method', 'roi', '-mask', atlas_file,
#               '-sf', sessid_file, '-outDir', rsfc_dir]
#    os.system(' '.join(cmd_str))

# compute rsfc
roi = ['rOFA', 'rFFA']

# load roi stats file
roi_stats_file = os.path.join(atlas_dir, 'roi_stat.csv')
roi_stats = open(roi_stats_file).readlines()
roi_stats = [line.strip().split(',') for line in roi_stats]
header = roi_stats.pop(0)
roi_idx = {}
for i in range(1, 5):
    roi_idx[header[i]] = i

for i in range(len(roi_stats)):
    flag = roi_stats[i]
    subj = flag[0]
    roi_flag_1 = int(flag[roi_idx[roi[0]]])
    roi_flag_2 = int(flag[roi_idx[roi[1]]])
    if roi_flag_1 and roi_flag_2:
        flag = flag[1:5]
        flag = [int(item) for item in flag]
        new_idx = []
        for j in range(len(flag)):
            new_idx.append(sum(flag[0:(j+1)]))
        # load ts data
        ts_file = os.path.join(rsfc_dir, subj, 'seed_ts',
                               subj + '_atlas_ts.txt')
        ts_data = np.loadtxt(ts_file)
        r = np.corrcoef(ts_data[..., new_idx[roi_idx[roi[0]]-1]],
                        ts_data[..., new_idx[roi_idx[roi[1]]-1]])[0, 1]
        print subj, r

