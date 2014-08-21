# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import time

# modules for data preparation
import multiprocessing as mps
import functools

# modules for modle training and testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nibabel as nib
from mypy import base as mybase

import autoroilib as arlib
from mypy import math as mymath

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'multi-atlas', 'data', 'all')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# generate mask and probability map
prob_data, mask_data = arlib.make_prior(sessid, data_dir)
mask_coords = arlib.get_mask_coord(mask_data, data_dir)

# extract features from each subject
pool = mps.Pool(20)
result = pool.map(functools.partial(arlib.ext_subj_feature,
                                    mask_coord=mask_coords,
                                    data_dir=cv_dir, mask_out=False),
                  sessid)

##-- Cross-validation to evaluate performance of model
#ofa_dice = np.zeros((cv_num))
#ffa_dice = np.zeros((cv_num))
#
### get feature name
##feature_name_file = os.path.join(data_dir, 'feature_name.txt')
##feature_name = arlib.get_label(feature_name_file)
##feature_name = [feature_name[i] for i in feature_idx]
#
## predicted nifti directory
#pred_dir = os.path.join(data_dir, 'predicted_files')
#os.system('mkdir ' + pred_dir)
#
## split all subjects into 5 folds
#subj_group = arlib.split_subject(sessid, cv_num)
#for i in range(cv_num):
#    print 'CV iter - ' + str(i)
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    
#    # split data into training and test group
#    test_sessid = subj_group[i]
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#
#    # Forest training with single subject's data
#    forest_list = []
#    spatial_ptn = None
#    for subj in train_sessid:
#        train_data = arlib.get_list_data([subj], cv_dir)
#        train_x = train_data[..., :-1]
#        train_y = train_data[..., -1]
#        if not isinstance(spatial_ptn, np.array):
#            spatial_ptn = np.zeros((train_data.shape[0], len(train_sessid)))
#            count = 0
#        spatial_ptn[..., count] = train_x[..., 0]
#        count += 1
#        clf = RandomForestClassifier(n_estimators=50, max_depth=30,
#                                     criterion='gini', n_jobs=20)
#        clf.fit(train_x, train_y)
#        forest_list.append(clf)
#
#    # Model testing
#    ffa_dice_tmp = []
#    ofa_dice_tmp = []
#
#    for subj in test_sessid:
#        print 'Test data - subject %s'%(subj)
#        test_data = arlib.get_list_data([subj], cv_dir)
#        test_x = test_data[..., :-1]
#        test_y = test_data[..., -1]
#        # TODO: define similarity index
#        similarity = []
#        for j in range(len(train_sessid)):
#            similarity.append(
#                    np.corrcoef(test_x[..., 0], spatial_ptn[..., j])[0, 1])
#        pred_prob = None
#        count = 0
#        for clf_idx in range(len(train_sessid)):
#            clf = forest_list[clf_idx]
#            prob = clf.predict_proba(test_x) * similarity[clf_idx]
#            if not isinstance(pred_prob, np.array):
#                pred_prob = prob
#            else:
#                pred_prob += prob
#            count += 1
#        pred_prob = pred_prob / count
#        pred_y = np.argmax(pred_prob, axis=1)
#        pred_y[pred_y==2] = 3
#
#        for label_idx in [1, 3]:
#            P = pred_y == label_idx
#            T = test_y == label_idx
#            dice_val = mymath.dice_coef(T, P)
#            print 'Dice for label %s: %f'%(label_idx, dice_val)
#            if label_idx == 3:
#                ffa_dice_tmp.append(dice_val)
#            else:
#                ofa_dice_tmp.append(dice_val)
#        print '-----------------------'
#
#        #-- save predicted label as nifti files
#        fsl_dir = os.getenv('FSL_DIR')
#        img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
#                                    'MNI152_T1_2mm_brain.nii.gz'))
#        # save predicted label
#        header = img.get_header()
#        coords = test_x[testy_x, 1:4]
#        pred_data = arlib.write2array(coords, pred_y)
#        out_file = os.path.join(pred_dir, subj + '_pred.nii.gz')
#        mybase.save2nifti(pred_data, header, out_file)
#
#    ffa_dice[i] = np.mean(ffa_dice_tmp)
#    ofa_dice[i] = np.mean(ofa_dice_tmp)
#
#print 'Mean FFA Dice is %s'%(ffa_dice.mean())
#print 'Mean OFA Dice is %s'%(ofa_dice.mean())

