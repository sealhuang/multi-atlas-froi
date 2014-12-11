# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from nipytools import base as mybase

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')
ma_dir = os.path.join(base_dir, 'multi-atlas')
gcss_dir = os.path.join(base_dir, 'gcss')
group08_dir = os.path.join(ma_dir, 'group08')

## read all subjects' SID from group 06
#sessid_file = os.path.join(doc_dir, 'sessid')
#sessid = open(sessid_file).readlines()
#sessid = [line.strip() for line in sessid]

# read all subjects' SID from group 08
sessid_file = os.path.join(group08_dir, 'sessid')
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

##-- examine overlap between predicted fROIs
#fc_dir = os.path.join(ma_dir, 'l_ofa_ffa', '40_atlas_pred')
#sts_dir = os.path.join(ma_dir, 'l_sts', '40_atlas_pred')
#
#for subj in sessid:
#    fc_file = os.path.join(fc_dir, subj + '_pred.nii.gz')
#    sts_file = os.path.join(sts_dir, subj + '_pred.nii.gz')
#    fc_data = nib.load(fc_file).get_data()
#    sts_data = nib.load(sts_file).get_data()
#    fc_data[fc_data>0] = 1
#    sts_data[sts_data>0] = 1
#    sum_data = fc_data + sts_data
#    union_data = sum_data>0
#    dif = sum_data - union_data
#    if dif.sum():
#        print subj,
#        print dif.sum()

##-- merge predicted labels (left and right hemisphere)
#l_dir = os.path.join(ma_dir, 'l_sts', '40_atlas_pred')
#r_dir = os.path.join(ma_dir, 'r_sts', '40_atlas_pred')
#targ_dir = os.path.join(ma_dir, 'predicted_files')
#
#for subj in sessid:
#    l_file = os.path.join(l_dir, subj + '_pred.nii.gz')
#    r_file = os.path.join(r_dir, subj + '_pred.nii.gz')
#    targ = os.path.join(targ_dir, subj + '_pred.nii.gz')
#    os.system(' '.join(['fslmaths', l_file, '-add', r_file, targ]))
#
#merged_file = os.path.join(targ_dir, 'merged_sts_pred.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid:
#    temp = os.path.join(targ_dir, subj + '_pred.nii.gz')
#    cmd_str.append(temp)
#os.system(' '.join(cmd_str))

##-- merge zstat file from 08 group
#src_dir = os.path.join(group08_dir, 'localizer')
#merged_file = os.path.join(group08_dir, 'merged_face_obj_zstat.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid:
#    temp = os.path.join(src_dir, subj + '_face_obj_zstat.nii.gz')
#    cmd_str.append(temp)
#os.system(' '.join(cmd_str))

##-- merge cope file from 08 group
#src_dir = os.path.join(group08_dir, 'exp')
#merged_file = os.path.join(group08_dir, 'merged_face_obj_cope.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid:
#    temp = os.path.join(src_dir, subj + '_face_obj_cope.nii.gz')
#    cmd_str.append(temp)
#os.system(' '.join(cmd_str))

##-- merge beta file
#src_dir = r'/nfs/t2/fmricenter/volume'
#merged_file = os.path.join(data_dir, 'merged_face_obj_cope.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid:
#    temp = os.path.join(src_dir, subj, 'obj.gfeat', 'cope1.feat', 'stats',
#                        'cope1.nii.gz')
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

##-- compute rsfc
#atlas_dir = os.path.join(data_dir, 'peak_mask_1')
##atlas_dir = os.path.join(gcss_dir, 'peak_mask')
##atlas_dir = os.path.join(ma_dir, 'predicted_files', 'peak_mask')
#rsfc_dir = os.path.join(atlas_dir, 'rsfc')
#sessid_file = os.path.join(rsfc_dir, 'sessid')
#
### extract time courses
##for i in range(len(sessid)):
##    f = open(sessid_file, 'w')
##    f.write(sessid[i])
##    f.close()
##    atlas_file = os.path.join(atlas_dir, sessid[i] + '_atlas.nii.gz')
##    cmd_str = ['extract-roi-tc', '-method', 'roi', '-mask', atlas_file,
##               '-sf', sessid_file, '-outDir', rsfc_dir]
##    os.system(' '.join(cmd_str))
#
## compute rsfc
#roi = ['rOFA', 'rFFA']
#
## load roi stats file
#roi_stats_file = os.path.join(atlas_dir, 'roi_stat.csv')
#roi_stats = open(roi_stats_file).readlines()
#roi_stats = [line.strip().split(',') for line in roi_stats]
#header = roi_stats.pop(0)
#roi_idx = {}
#for i in range(1, 5):
#    roi_idx[header[i]] = i
#
#for i in range(len(roi_stats)):
#    flag = roi_stats[i]
#    subj = flag[0]
#    roi_flag_1 = int(flag[roi_idx[roi[0]]])
#    roi_flag_2 = int(flag[roi_idx[roi[1]]])
#    if roi_flag_1 and roi_flag_2:
#        flag = flag[1:5]
#        flag = [int(item) for item in flag]
#        new_idx = []
#        for j in range(len(flag)):
#            new_idx.append(sum(flag[0:(j+1)]))
#        # load ts data
#        ts_file = os.path.join(rsfc_dir, subj, 'seed_ts',
#                               subj + '_atlas_ts.txt')
#        ts_data = np.loadtxt(ts_file)
#        r = np.corrcoef(ts_data[..., new_idx[roi_idx[roi[0]]-1]],
#                        ts_data[..., new_idx[roi_idx[roi[1]]-1]])[0, 1]
#        print subj, r

#-- extract beta value for each subject
roi_label = [8, 10, 12]
#roi_label = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
roi_name = ['lpcSTS', 'lpSTS', 'laSTS']
#roi_name = ['rOFA', 'lOFA', 'rFFA', 'lFFA', 'rpcSTS', 'lpcSTS', 'rpSTS',
#            'lpSTS', 'raSTS', 'laSTS']
#merged_pred = os.path.join(group08_dir, 'merged_gss_pred.nii.gz')
pred_dir = os.path.join(group08_dir, 'predicted_files', 'l_sts')
merged_cope = os.path.join(group08_dir, 'merged_face_cope.nii.gz')
#merged_cope = os.path.join(data_dir, 'merged_zstat.nii.gz')
out_file = r'face_cope_ma.log'

# load data
#pred_data = np.around(nib.load(merged_pred).get_data())
cope_data = nib.load(merged_cope).get_data()

out_data = []

for i in range(len(sessid)):
    pred_file = os.path.join(pred_dir, sessid[i] + '_pred.nii.gz')
    pred_data = np.around(nib.load(pred_file).get_data())
    temp_data = []
    for roi in roi_label:
        #mask = pred_data[..., i].copy()
        mask = pred_data.copy()
        mask[mask!=roi] = 0
        mask[mask==roi] = 1
        if mask.sum():
            masked_cope = mask * cope_data[..., i]
            m = masked_cope.sum() / mask.sum()
            temp_data.append(m)
        else:
            temp_data.append(0)
    out_data.append(temp_data)

f = open(out_file, 'w')
f.write(','.join(roi_name)+'\n')
for line in out_data:
    temp = [str(item) for item in line]
    f.write(','.join(temp)+'\n')

##-- copy data from 08 group
#src_dir = r'/nfs/h1/workingshop/huanglijie/fmri/face_feat_08'
#vol_dir = os.path.join(src_dir, 'volume')
#targ_dir = os.path.join(ma_dir, 'group08')
#
#sessid_08_file = os.path.join(targ_dir, 'sessid')
#sessid_08 = open(sessid_08_file).readlines()
#sessid_08 = [line.strip() for line in sessid_08]
#
#for subj in sessid_08:
#    rlf_file = os.path.join(vol_dir, subj, 'obj', 'obj.rlf')
#    rlf = open(rlf_file).readlines()
#    rlf = [line.strip() for line in rlf]
#    #src_zstat = os.path.join(vol_dir, subj, 'obj.gfeat', 'cope1.feat',
#    #                         'stats', 'zstat1.nii.gz')
#    src_cope = os.path.join(vol_dir, subj, 'obj', rlf[2], 'func.feat',
#                            'reg_standard', 'stats', 'cope8.nii.gz')
#    #targ_zstat = os.path.join(targ_dir, 'localizer',
#    #                          subj + '_face_obj_zstat.nii.gz')
#    targ_cope = os.path.join(targ_dir, 'scramble_cope', subj + '_cope.nii.gz')
#    os.system('cp ' + src_cope + ' ' + targ_cope)
#    #os.system('cp ' + src_zstat + ' ' + targ_zstat)
#
#merged_file = os.path.join(targ_dir, 'merged_scramble_cope.nii.gz')
#cmd_str = ['fslmerge', '-a', merged_file]
#for subj in sessid_08:
#    temp = os.path.join(targ_dir, 'scramble_cope', subj + '_cope.nii.gz')
#    cmd_str.append(temp)
#os.system(' '.join(cmd_str))

##-- compute mean dice for random selection approach
#roi_list = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
#
#base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi/multi-atlas'
#data_dir = os.path.join(base_dir, 'plot', 'random_select_40')
#f_list = os.listdir(data_dir)
#f_list = [line.split('.') for line in f_list]
#
#for idx in roi_list:
#    out_file = os.path.join(base_dir, 'mean_' + str(idx) + '.csv')
#    f = open(out_file, 'wb')
#    file_header = 'label_' + str(idx)
#    for item in f_list:
#        if file_header in item:
#            data = np.loadtxt(os.path.join(data_dir, '.'.join(item)),
#                              delimiter=',')
#            m = data.mean(axis=0)
#            #f.write(','.join([str(num) for num in m])+'\n')
#            f.write(str(m)+'\n')
#        else:
#            continue

