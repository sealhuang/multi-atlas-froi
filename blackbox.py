# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import re

import autoroilib as arlib
import macro

def get_label_file(subject_dir):
    """
    Get a subject-specific label file.

    """
    f_list = os.listdir(subject_dir)
    for f in f_list:
        if re.search('_ff.nii.gz', f):
            return os.path.join(subject_dir, f)
        else:
            return None

def get_zstat_list(sid_list, db_dir):
    """
    Get a zstat file list based on sessid list and database directory.

    """
    zstat_list = []
    for subj in sid_list:
        subject_dir = os.path.join(db_dir, sid, 'face-object')
        if not os.path.exists(subject_dir):
            print 'Subject %s does not exist in database.'%(subj)
            return
        zstat = os.path.join(subject_dir, 'zstat1.nii.gz')
        zstat_list.append(zstat)
    return zstat_list

def get_label_list(sid_list, db_dir):
    """
    Get a zstat file list based on sessid list and database directory.

    """
    label_list = []
    for subj in sid_list:
        subject_dir = os.path.join(db_dir, sid, 'face-object')
        if not os.path.exists(subject_dir):
            print 'Subject %s does not exist in database.'%(subj)
            return
        label = get_label_file(subject_dir)
        label_list.append(label)
    return label_list

def ext_subj_sample(sid_list, zstat_file_list, mask_coord, class_label,
                    output_dir, label_file_list=None):
    """
    Warper of ext_sample and save_sample for each subject.

    """
    subj_num = len(sid_list)
    zstat_num = len(zstat_file_list)
    if not subj_num == zstat_num:
        print 'Inconsistency of input subject number! (zstat file list)'
        return
    if label_file_list:
        label_num = len(label_list_num)
        if not subj_num == label_num:
            print 'Inconsistency of input subject number! (label file list)'
            return

    for i in range(subj_num):
        subj = sid_list[i]
        if label_file_list:
            feature_name, sample_data = arlib.ext_sample(zstat_file_list[i],
                                                         mask_coord,
                                                         class_label,
                                                         label_file_list[i])
        else:
            feature_name, sample_data = arlib.ext_sample(zstat_file_list[i],
                                                         mask_coord,
                                                         class_label)
        out_file = os.path.join(output_dir, subj + '_data.csv')
        arlib.save_sample(feature_name, sample_data, out_file)

def get_subj_sample(sid_list, data_dir):
    """
    Get data for a list of subjects.
    #sid_list# could be either a subject's SID or a SID list.

    """
    if isinstance(sid_list, str):
        sid_list = [sid_list]
    if not isinstance(sid_list, list):
        print 'Invalid input!'
        return
    samples = np.array([])
    for subj in sid_list:
        f = os.path.join(data_dir, subj + '_data.csv')
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        if not samples.size:
            samples = data
        else:
            samples = np.vstack((samples, data))
    return samples

    


db_dir = r'/nfs/t2/atlas/database'
base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'multi-atlas', 'l_ffa_ofa')

class_label = [2, 4]
#atlas_num = [1, 5] + range(10, 201, 10)
#atlas_num = [1, 5]
atlas_num = [50]
#atlas_num = range(1, 201)

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

## extract samples
#macro.extract_sample(sessid, class_label, data_dir)

## model training and testing
#forest_list, classes_list, spatial_ptn = macro.train_model(sessid, data_dir)
#dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
#                                forest_list, classes_list, spatial_ptn)
#
## save dice to a file
#macro.save_dice(dice, data_dir)

##-- random atlas selection
#for i in range(50):
#    #start_time = time.time()
#    # model training and testing
#    forest_list, classes_list, spatial_ptn = macro.train_model(sessid,
#                                                               data_dir)
#    dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
#                                    forest_list, classes_list, spatial_ptn,
#                                    sorted=False)
#    #print 'Cost %s'%(time.time()-start_time)
#    ## save dice to a file
#    #macro.save_dice(dice, data_dir)

##-- effect of forest parameter
#tree_num = range(10, 51, 5)
#tree_depth = range(10, 31, 5)
#
#for n in tree_num:
#    for d in tree_depth:
#        print 'number - %s, depth - %s'%(n, d)
#        forest_list, classes_list, spatial_ptn = macro.train_model(sessid,
#                                                                   data_dir,
#                                                                   n_tree=n,
#                                                                   d_tree=d)
#
#        dice = macro.leave_one_out_test(sessid, atlas_num, data_dir,
#                                        class_label, forest_list,
#                                        classes_list, spatial_ptn)
#
#        # save dice to a file
#        macro.save_dice(dice, data_dir)

##-- relation between atlas rank and Dice
## model training and testing
#forest_list, classes_list, spatial_ptn = macro.train_model(sessid, data_dir)
#dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
#                                forest_list, classes_list, spatial_ptn,
#                                single_atlas=True)
#
## save dice to a file
#macro.save_dice(dice, data_dir)

#-- test model in an independent dataset
mask_data = arlib.make_prior(sessid, class_label, data_dir)
mask_coords = arlib.get_mask_coord(mask_data, data_dir)
forest_list, classes_list, spatial_ptn = macro.train_model(sessid, data_dir)

test_dir = os.path.join(base_dir, 'multi-atlas', 'group08', 'localizer')
pred_dir = os.path.join(base_dir, 'multi-atlas', 'group08', 'l_predicted_files')
test_sessid_file = os.path.join(base_dir, 'multi-atlas', 'group08', 'sessid')
test_sessid = open(test_sessid_file).readlines()
test_sessid = [line.strip() for line in test_sessid]

for subj in test_sessid:
    zstat_file = os.path.join(test_dir, subj + '_face_obj_zstat.nii.gz')
    sample_label, sample_data = arlib.ext_sample(zstat_file, mask_coords,
                                                 class_label)
    macro.predict(sample_data, atlas_num, pred_dir, subj + '_pred.nii.gz',
                  class_label, forest_list, classes_list, spatial_ptn)

