# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import time

# modules for model specification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import normalized_mutual_info_score

import lib as arlib
from mypy import math as mymath
from mypy import base as mybase

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
        label_num = len(label_file_list)
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

def prepare(sid_list, zstat_file_list, label_file_list,
            class_label, data_dir):
    """
    Preparation for model training, including mask generation, sample
    extraction for each subject from training dataset.

    """
    # generate mask
    mask_data = arlib.make_mask(label_file_list, class_label, data_dir)
    mask_coord = arlib.get_mask_coord(mask_data, data_dir)
    # extract samples from each subject
    ext_subj_sample(sid_list, zstat_file_list, mask_coord, class_label,
                    data_dir, label_file_list)

def train(sid_list, data_dir, n_tree=30, d_tree=20):
    """
    Traing each atlas forest with one subject's data.

    """
    forest_list = []
    classes_list = []
    spatial_ptn = None
    print 'Model training ...'
    start_time = time.time()
    for subj in sid_list:
        train_data = get_subj_sample(subj, data_dir)
        z_vtr = train_data[..., 0].copy()
        z_vtr[z_vtr<2.3] = 0
        z_vtr[z_vtr>0] = 1
        # mask out voxels which are not activated significantly
        smp_mask = z_vtr > 0
        train_x = train_data[smp_mask, 0:4]
        train_y = train_data[smp_mask, -1]
        # save activation pattern
        if not isinstance(spatial_ptn, np.ndarray):
            spatial_ptn = np.zeros((train_data.shape[0], len(sid_list)))
            count = 0
        spatial_ptn[..., count] = z_vtr
        count += 1
        clf = RandomForestClassifier(n_estimators=n_tree,
                                     max_depth=d_tree,
                                     criterion='gini', n_jobs=20)
        clf.fit(train_x, train_y)
        forest_list.append(clf)
        classes_list.append(clf.classes_)
    return forest_list, classes_list, spatial_ptn

def leave_one_out_test(sid_list, atlas_num, data_dir, class_label,
                       forest_list, classes_list, spatial_ptn,
                       save_nifti=False, sorted=True, single_atlas=False):
    """
    Evaluate classifier performance with leave-one-out scheme.

    """
    # initial output dice value
    dice = {}
    for idx in class_label:
        dice[idx] = []

    for i in range(len(sid_list)):
        print 'Test subject %s'%(sid_list[i])
        test_data = get_subj_sample(sid_list[i], data_dir)
        # mask out voxels which are not activated significantly
        smp_mask = test_data[..., 0] >= 2.3
        test_x = test_data[smp_mask, 0:4]
        test_y = test_data[smp_mask, -1]
        
        # define similarity index
        similarity = []
        atlas_idx = []
        for j in range(len(sid_list)):
            if i == j:
                continue
            atlas_idx.append(j)
            r = normalized_mutual_info_score(spatial_ptn[..., i],
                                             spatial_ptn[..., j])
            similarity.append(r)

        # sort the similarity
        sorted_sim_idx = np.argsort(similarity)[::-1]

        # label the activation voxels with atlas forests (AFs)
        tmp_dice = {}
        for idx in class_label:
            tmp_dice[idx] = []

        for num in atlas_num:
            print 'atlas number %s'%(num)
            if sorted:
                if not single_atlas:
                    selected_atlas = sorted_sim_idx[0:num]
                else:
                    selected_atlas = sorted_sim_idx[(num-1):num]
            else:
                selected_atlas = np.random.choice(len(sorted_sim_idx),
                                                  num, replace=False)
            pred_prob = None
            for idx in selected_atlas:
                clf = forest_list[atlas_idx[idx]]
                prob = clf.predict_proba(test_x)
                std_prob = np.zeros((prob.shape[0], len(class_label)+1))
                # TODO: construct a std prob table
                for cls_idx in range(prob.shape[1]):
                    if classes_list[atlas_idx[idx]][cls_idx] == 0:
                        std_prob[..., 0] = prob[..., cls_idx]
                    else:
                        tmp_idx = class_label.index(
                                classes_list[atlas_idx[idx]][cls_idx])
                        std_prob[..., tmp_idx+1] = prob[..., cls_idx]
                new_prob = np.zeros(std_prob.shape)
                new_prob[range(new_prob.shape[0]),
                        np.argmax(std_prob, axis=1)] = 1
                # optional: modulate prediction probability with similarity
                #new_prob = new_prob * similarity[clf_idx]
                if not isinstance(pred_prob, np.ndarray):
                    pred_prob = new_prob
                else:
                    pred_prob += new_prob
            #print pred_prob.shape
            tmp_pred_y = np.argmax(pred_prob, axis=1)
            pred_y = np.zeros(tmp_pred_y.shape)
            for k in range(1, pred_prob.shape[1]):
                pred_y[tmp_pred_y==k] = class_label[k-1]

            for label_idx in class_label:
                P = pred_y == label_idx
                T = test_y == label_idx
                dice_val = mymath.dice_coef(T, P)
                print 'Dice for label %s: %f'%(label_idx, dice_val)
                tmp_dice[label_idx].append(dice_val)

            # save predicted label as a nifti file
            if save_nifti:
                # predicted nifti directory
                pred_dir = os.path.join(data_dir, str(num) + '_atlas_pred')
                if not os.path.exists(pred_dir):
                    os.system('mkdir ' + pred_dir)
                fsl_dir = os.getenv('FSL_DIR')
                img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
                                            'MNI152_T1_2mm_brain.nii.gz'))
                # save predicted label
                header = img.get_header()
                coords = test_x[..., 1:4]
                pred_data = arlib.write2array(coords, pred_y)
                pred_data = np.around(pred_data)
                out_file = os.path.join(pred_dir, sid_list[i]+'_pred.nii.gz')
                mybase.save2nifti(pred_data, header, out_file)

        for idx in class_label:
            dice[idx].append(tmp_dice[idx])

    for idx in class_label:
        print 'Mean Dice for label %s: %s'%(idx, np.mean(dice[idx], axis=0)[0])
    return dice

def predict(x_mtx, atlas_num, out_dir, out_name, class_label,
            forest_list, classes_list, spatial_ptn,
            save_nifti=True, sorted=True, single_atlas=False):
    """
    Predict ROI label for unseen query image.

    """
    # mask out voxels which are not activated significantly
    x_mtx = np.array(x_mtx)
    z_vtr = x_mtx[..., 0].copy()
    z_vtr[z_vtr<2.3] = 0
    z_vtr[z_vtr>0] = 1
    smp_mask = z_vtr > 0
    test_x = x_mtx[smp_mask, 0:4]
        
    # define similarity index
    similarity = []
    for i in range(len(forest_list)):
        r = normalized_mutual_info_score(spatial_ptn[..., i], z_vtr)
        similarity.append(r)

    # sort the similarity
    sorted_sim_idx = np.argsort(similarity)[::-1]

    # label the activation voxels with atlas forests (AFs)
    for num in atlas_num:
        print 'atlas number %s'%(num)
        if sorted:
            if not single_atlas:
                selected_atlas = sorted_sim_idx[0:num]
            else:
                selected_atlas = sorted_sim_idx[(num-1):num]
        else:
            selected_atlas = np.random.choice(len(sorted_sim_idx),
                                              num, replace=False)
        pred_prob = None
        for idx in selected_atlas:
            clf = forest_list[idx]
            prob = clf.predict_proba(test_x)
            std_prob = np.zeros((prob.shape[0], len(class_label)+1))
            # TODO: construct a std prob table
            for cls_idx in range(prob.shape[1]):
                if classes_list[idx][cls_idx] == 0:
                    std_prob[..., 0] = prob[..., cls_idx]
                else:
                    tmp_idx = class_label.index(classes_list[idx][cls_idx])
                    std_prob[..., tmp_idx+1] = prob[..., cls_idx]
            new_prob = np.zeros(std_prob.shape)
            new_prob[range(new_prob.shape[0]),
                     np.argmax(std_prob, axis=1)] = 1
            # optional: modulate prediction probability with similarity
            #new_prob = new_prob * similarity[clf_idx]
            if not isinstance(pred_prob, np.ndarray):
                pred_prob = new_prob
            else:
                pred_prob += new_prob
        #print pred_prob.shape
        tmp_pred_y = np.argmax(pred_prob, axis=1)
        pred_y = np.zeros(tmp_pred_y.shape)
        for k in range(1, pred_prob.shape[1]):
            pred_y[tmp_pred_y==k] = class_label[k-1]

        # save predicted label as a nifti file
        if save_nifti:
            # predicted nifti directory
            if not os.path.exists(out_dir):
                os.system('mkdir ' + out_dir)
            fsl_dir = os.getenv('FSL_DIR')
            img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
                                        'MNI152_T1_2mm_brain.nii.gz'))
            # save predicted label
            header = img.get_header()
            coords = test_x[..., 1:4]
            pred_data = arlib.write2array(coords, pred_y)
            pred_data = np.around(pred_data)
            out_file = os.path.join(out_dir, out_name)
            mybase.save2nifti(np.around(pred_data), header, out_file)

def save_dice(dice_dict, out_dir):
    """
    Save Dice dict to a file.

    """
    for label in dice_dict:
        out_file_name = 'label_' + str(label) + '.txt'
        if os.path.exists(os.path.join(out_dir, out_file_name)):
            out_file_name += '.' + str(time.time())
        #print out_file_name
        f = open(os.path.join(out_dir, out_file_name), 'w')
        data = dice_dict[label]
        for line in data:
            tmp_line = [str(item) for item in line]
            tmp_line = ','.join(tmp_line)
            f.write(tmp_line + '\n')

