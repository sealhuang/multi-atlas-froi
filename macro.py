# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

# modules used for sample extraction
import multiprocessing as mps
import functools

# modules for model specification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import normalized_mutual_info_score

import autoroilib as arlib
from mypy import math as mymath
from mypy import base as mybase

def extract_sample(sessid, class_label, data_dir):
    """
    Extract samples from all subjects.

    """
    # generate mask
    mask_data = arlib.make_prior(sessid, class_label, data_dir)
    mask_coords = arlib.get_mask_coord(mask_data, data_dir)
    # extract features from each subject
    pool = mps.Pool(20)
    result = pool.map(functools.partial(arlib.ext_subj_feature,
                                        mask_coord=mask_coords,
                                        class_label=class_label,
                                        out_dir=data_dir),
                      sessid)
    pool.terminate()

def train_model(sessid, data_dir, n_tree=30, d_tree=20):
    """
    Traing each atlas forest with one subject's data.

    """
    forest_list = []
    classes_list = []
    spatial_ptn = None
    print 'Model training ...'
    for subj in sessid:
        train_data = arlib.get_subj_data(subj, data_dir)
        z_vtr = train_data[..., 0].copy()
        z_vtr[z_vtr<2.3] = 0
        z_vtr[z_vtr>0] = 1
        # mask out voxels which are not activated significantly
        smp_mask = z_vtr > 0
        train_x = train_data[smp_mask, 1:4]
        train_y = train_data[smp_mask, -1]
        # save activation pattern
        if not isinstance(spatial_ptn, np.ndarray):
            spatial_ptn = np.zeros((train_data.shape[0], len(sessid)))
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

def leave_one_out_test(sessid, atlas_num, data_dir, class_list,
                       forest_list, classes_list, spatial_ptn,
                       save_nifti=False, sorted=True):
    """
    Test classifier performance with leave-one-out scheme.

    """
    # initial output dice value
    dice = {}
    for class_label in class_list:
        dice[class_label] = []

    for i in range(len(sessid)):
        print 'Test subject %s'%(sessid[i])
        test_data = arlib.get_subj_data(sessid[i], data_dir)
        # mask out voxels which are not activated significantly
        smp_mask = test_data[..., 0] >= 2.3
        test_x = test_data[smp_mask, 1:4]
        test_y = test_data[smp_mask, -1]
        
        # define similarity index
        similarity = []
        atlas_idx = []
        for j in range(len(sessid)):
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
        for class_label in class_list:
            tmp[class_label] = []

        for num in atlas_num:
            print 'atlas number %s'%(num)
            if sorted:
                selected_atlas = sorted_sim_idx[0:num]
            else:
                selected_atlas = np.random.choice(len(sorted_sim_idx),
                                                  num, replace=False)
            pred_prob = None
            for idx in selected_atlas:
                clf = forest_list[atlas_idx[idx]]
                prob = clf.predict_proba(test_x)
                std_prob = np.zeros((prob.shape[0], len(class_list)))
                # TODO: construct a std prob table
                for cls_idx in range(prob.shape[1]):
                    if classes_list[atlas_idx[idx]][cls_idx] == 0:
                        std_prob[..., 0] = prob[..., cls_idx]
                    elif classes_list[atlas_idx[idx]][cls_idx] == 1:
                        std_prob[..., 1] = prob[..., cls_idx]
                    elif classes_list[atlas_idx[idx]][cls_idx] == 3:
                        std_prob[..., 2] = prob[..., cls_idx]
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
            pred_y = np.argmax(pred_prob, axis=1)
            pred_y[pred_y==2] = 3

            for label_idx in [1, 3]:
                P = pred_y == label_idx
                T = test_y == label_idx
                dice_val = mymath.dice_coef(T, P)
                print 'Dice for label %s: %f'%(label_idx, dice_val)
                if label_idx == 3:
                    temp_ffa_dice.append(dice_val)
                else:
                    temp_ofa_dice.append(dice_val)

        ##-- save predicted label as nifti files
        ## predicted nifti directory
        #pred_dir = os.path.join(data_dir, 'predicted_files')
        #if not os.path.exists(pred_dir):
        #    os.system('mkdir ' + pred_dir)
        #fsl_dir = os.getenv('FSL_DIR')
        #img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
        #                            'MNI152_T1_2mm_brain.nii.gz'))
        ## save predicted label
        #header = img.get_header()
        #coords = test_x[test_x, 1:4]
        #pred_data = arlib.write2array(coords, pred_y)
        #out_file = os.path.join(pred_dir, subj + '_pred.nii.gz')
        #mybase.save2nifti(pred_data, header, out_file)

        ffa_dice.append(temp_ffa_dice)
        ofa_dice.append(temp_ofa_dice)

    print 'Mean Dice - FFA:'
    print np.array(ffa_dice).mean(axis=0)
    print 'Mean Dice - OFA:'
    print np.array(ofa_dice).mean(axis=0)

out_file = 'ffa_output.txt'
f = open(out_file, 'w')
str_line = [str(item) for item in selected_num]
str_line = ','.join(str_line)
f.write(str_line + '\n')
for line in ffa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

out_file = 'ofa_output.txt'
f = open(out_file, 'w')
str_line = [str(item) for item in selected_num]
str_line = ','.join(str_line)
f.write(str_line + '\n')
for line in ofa_dice:
    tmp_line = [str(item) for item in line]
    tmp_line = ','.join(tmp_line)
    f.write(tmp_line + '\n')

