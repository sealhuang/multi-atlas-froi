# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import re
import nibabel as nib

import lib
import model

def get_label_file(subject_dir):
    """
    Get a subject-specific label file.

    """
    f_list = os.listdir(subject_dir)
    for f in f_list:
        if re.search('_ff.nii.gz', f):
            return os.path.join(subject_dir, f)

def get_zstat_list(sid_list, db_dir):
    """
    Get a zstat file list based on sessid list and database directory.

    """
    zstat_list = []
    for subj in sid_list:
        subject_dir = os.path.join(db_dir, subj, 'obj', 'face-object')
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
        subject_dir = os.path.join(db_dir, subj, 'obj', 'face-object')
        if not os.path.exists(subject_dir):
            print 'Subject %s does not exist in database.'%(subj)
            return
        label = get_label_file(subject_dir)
        label_list.append(label)
    return label_list

def extract_mean_overlap():
    """
    Calculate mean overlap across subjects for a ROI.

    """
    #-- directory config
    db_dir = r'/nfs/t2/BAA/SSR'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    roi_index = 9
    print 'ROI index: %s'%(roi_index)

    label_file_list = get_label_list(sessid, db_dir)

    new_data = np.zeros((91, 109, 91))
    for item in label_file_list:
        data = nib.load(item).get_data()
        data[data!=roi_index] = 0
        data[data==roi_index] = 1
        new_data += data
    mask = new_data.copy()
    mask[mask>0] = 1
    new_data = new_data / len(sessid)
    print 'Max: %s'%(new_data.max())
    mean_prob = new_data.sum() / mask.sum()
    print 'Mean: %s'%(mean_prob)

def model_training_with_LOOCV_testing():
    """
    Training model and test it with leave-one-out cross-validation.

    """
    print 'Traing model and test it with leave-one-out cross-validation ...'
    #-- directory config
    db_dir = r'/nfs/t2/BAA/SSR'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'ma_202', 'r_fc')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [1, 3]
    atlas_num = [40]
    #atlas_num = [1, 5] + range(10, 201, 10)
    #atlas_num = range(1, 10)
    #atlas_num = range(1, 201)

    #-- preparation for model training
    # get zstat and label file for training dataset
    zstat_file_list = get_zstat_list(sessid, db_dir)
    label_file_list = get_label_list(sessid, db_dir)
    model.prepare(sessid, zstat_file_list, label_file_list,
                  class_label, data_dir)

    #-- model training
    forest_list, classes_list, spatial_ptn = model.train(sessid, data_dir)
    dice = model.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
                                    forest_list, classes_list, spatial_ptn)
                                    #save_nifti=True)

    #-- save dice to a file
    model.save_dice(dice, data_dir)

def model_testing_with_LOOCV_random():
    """
    Training a model with random atlas selection, and test it using
    leave-one-out cross-validation.

    """
    print 'Traing model with random atlas selection and test it with ' + \
          'leave-one-out cross-validation ...'
    #-- directory config
    db_dir = r'/nfs/t2/atlas/database'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'code_test')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [1, 3]
    #atlas_num = [50]
    atlas_num = [1, 5] + range(10, 201, 10)
    #atlas_num = range(1, 201)
    iter_num = 50

    for i in range(iter_num):
        print 'Iter - %s'%(i)
        start_time = time.time()
        # model training and testing
        forest_list, classes_list, spatial_ptn = model.train(sessid, data_dir)
        dice = model.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
                                        forest_list, classes_list, spatial_ptn,
                                        sorted=False)
        print 'Cost %s'%(time.time()-start_time)
        # save dice to a file
        model.save_dice(dice, data_dir)

def forest_parameter_selection():
    """
    Assessment of impact of forest parameters.

    """
    print 'Assess the imapct of forest parameters with leave-one-out ' + \
          'cross-validation ...'
    #-- directory config
    db_dir = r'/nfs/t2/atlas/database'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'ma_202', 'l_sts')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [8, 10, 12]
    atlas_num = [40]
    #atlas_num = [1, 5] + range(10, 201, 10)
    #atlas_num = range(1, 201)

    tree_num = range(10, 41, 5)
    tree_depth = range(10, 41, 5)

    for n in tree_num:
        for d in tree_depth:
            print 'number - %s, depth - %s'%(n, d)
            forest_list, classes_list, spatial_ptn = model.train(sessid,
                                                                 data_dir,
                                                                 n_tree=n,
                                                                 d_tree=d)

            dice = model.leave_one_out_test(sessid, atlas_num, data_dir,
                                            class_label, forest_list,
                                            classes_list, spatial_ptn)

            # save dice to a file
            model.save_dice(dice, data_dir)

def model_testing_with_LOOCV_single_atlas():
    """
    Training a model with one atlas selected.

    """
    print 'Traing model with one atlas selected and test it with ' + \
          'leave-one-out cross-validation ...'
    #-- directory config
    db_dir = r'/nfs/t2/atlas/database'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'ma_202', 'r_fc')

    #-- load session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [1, 3]
    #atlas_num = [50]
    #atlas_num = [1, 5] + range(10, 201, 10)
    atlas_num = range(1, 202)
    #iter_num = 50

    #-- model training and testing
    forest_list, classes_list, spatial_ptn = model.train(sessid, data_dir)
    dice = model.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
                                    forest_list, classes_list, spatial_ptn,
                                    single_atlas=True)

    #-- save dice to a file
    model.save_dice(dice, data_dir)

def model_testing_independent():
    """
    Training model and test it with an independent dataset.

    """
    print 'Traing model and test it with an independent dataset.'
    #-- directory config
    db_dir = r'/nfs/t2/atlas/database'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'multi-atlas', 'l_sts')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [8, 10, 12]
    atlas_num = [40]
    #atlas_num = [1, 5] + range(10, 201, 10)
    #atlas_num = range(1, 201)

    #-- model training
    forest_list, classes_list, spatial_ptn = model.train(sessid, data_dir)

    #-- load mask coordinate derived from training dataset
    mask_coords = lib.load_mask_coord(data_dir)

    #-- load testing dataset
    test_dir = r'/nfs/h1/workingshop/huanglijie/autoroi/multi-atlas/group08'
    loc_dir = os.path.join(test_dir, 'localizer')
    pred_dir = os.path.join(test_dir, 'predicted_files', 'l_sts')
    test_sessid_file = os.path.join(test_dir, 'sessid')
    test_sessid = open(test_sessid_file).readlines()
    test_sessid = [line.strip() for line in test_sessid]
    
    for subj in test_sessid:
        zstat_file = os.path.join(loc_dir, subj + '_face_obj_zstat.nii.gz')
        feature_name, sample_data = lib.ext_sample(zstat_file, mask_coords,
                                                   class_label)
        model.predict(sample_data, atlas_num, pred_dir, subj + '_pred.nii.gz',
                      class_label, forest_list, classes_list, spatial_ptn)

def get_af_posterior():
    """
    Get posterior estimate of each AF.

    """
    print 'Training model and generate posterior for each atlas ...'
    #-- directory config
    db_dir = r'/nfs/t2/atlas/database'
    base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'ma_202', 'r_fc')

    #-- laod session ID list for training
    sessid_file = os.path.join(doc_dir, 'sessid')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    #-- parameter config
    class_label = [1, 3]

    #-- model training and testing
    forest_list, classes_list, spatial_ptn = model.train(sessid, data_dir)
    model.get_posterior_map(sessid, data_dir, class_label, forest_list,
                            classes_list, spatial_ptn, save_nifti=True,
                            probabilistic=False)

if __name__ == '__main__':
    model_training_with_LOOCV_testing()
    #model_testing_independent()
    #model_testing_with_LOOCV_single_atlas()
    #forest_parameter_selection()
    #get_af_posterior()
    #extract_mean_overlap()

