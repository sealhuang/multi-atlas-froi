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
data_dir = os.path.join(base_dir, 'multi-atlas', 'data', 'cv')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#-- 5-folds cross-validation
cv_num = 5

# split all subjects into 5 folds
subj_group = arlib.split_subject(sessid, cv_num)
arlib.save_subject_group(subj_group, data_dir)

## data preparation for cross-validation
#for i in range(cv_num):
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    os.system('mkdir ' + cv_dir)
#    test_sessid = subj_group[i]
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#
#    # generate mask and probability map
#    prob_data, mask_data = arlib.make_prior(train_sessid, cv_dir)
#    mask_coords = arlib.get_mask_coord(mask_data, cv_dir)
#
#    # extract features from each subject
#    pool = mps.Pool(20)
#    result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                        mask_coord=mask_coords,
#                                        out_dir=cv_dir, mask_out=False),
#                      sessid)
#    pool.terminate()

##-- Cross-validation to select parameters
## parameters of random forest
#n_tree = range(10, 70, 10)
#depth = range(10, 70, 10)
#
## output matrix
#oob_score = np.zeros((len(n_tree), len(depth), cv_num))
#cv_score = np.zeros((len(n_tree), len(depth), cv_num))
#ofa_dice = np.zeros((len(n_tree), len(depth), cv_num))
#ffa_dice = np.zeros((len(n_tree), len(depth), cv_num))
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
#    # load test and train data
#    print 'Load data ...'
#    test_data = arlib.get_list_data(test_sessid, cv_dir)
#    train_data = arlib.get_list_data(train_sessid, cv_dir)
#    
#    ## sample stats
#    #print 'Samples stats for train-dataset:'
#    #arlib.samples_stat(train_data)
#    #print 'Samples stats for test-dataset:'
#    #arlib.samples_stat(test_data)
#
#    # split label and feature
#    train_x = train_data[..., feature_idx]
#    train_y = train_data[..., -1]
#    test_x = test_data[..., feature_idx]
#    test_y = test_data[..., -1]
#
#    for t_idx in range(len(n_tree)):
#        for d_idx in range(len(depth)):
#            p = [n_tree[t_idx], depth[d_idx]]
#            print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
#            #-- compare OOB eror rate and the CV error rate
#            # OOB error rate
#            clf = RandomForestClassifier(n_estimators=p[0],
#                                         criterion='gini',
#                                         max_depth=p[1],
#                                         n_jobs=20,
#                                         oob_score=True)
#            clf.fit(train_x, train_y)
#            oob_score[t_idx, d_idx, i] = clf.oob_score_
#            print 'OOB score is %s'%(str(clf.oob_score_))
#            # cross-validation
#            cv_score[t_idx, d_idx, i] = clf.score(test_x, test_y)
#            print 'Prediction score is %s'%(clf.score(test_x, test_y))
#            print 'Dice coefficient:'
#            pred_y = clf.predict(test_x)
#            for label_idx in [1, 3]:
#                P = pred_y == label_idx
#                T = test_y == label_idx
#                dice_val = mymath.dice_coef(T, P)
#                print 'Dice for label %s: %f'%(label_idx, dice_val)
#                if label_idx == 3:
#                    ffa_dice[t_idx, d_idx, i] = dice_val
#                else:
#                    ofa_dice[t_idx, d_idx, i] = dice_val
#            print '-----------------------'
#
#out_data_file = os.path.join(data_dir, 'parameter_cv_data.npz')
#np.savez(out_data_file, cv_score=cv_score, oob_score=oob_score,
#         ffa_dice=ffa_dice, ofa_dice=ofa_dice)

#-- Cross-validation to evaluate performance of model
selected_num = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
ofa_dice = {}
ffa_dice = {}
for num in selected_num:
    ofa_dice[num] = np.zeros((cv_num))
    ffa_dice[num] = np.zeros((cv_num))

## get feature name
#feature_name_file = os.path.join(data_dir, 'feature_name.txt')
#feature_name = arlib.get_label(feature_name_file)
#feature_name = [feature_name[i] for i in feature_idx]

## predicted nifti directory
#pred_dir = os.path.join(data_dir, 'predicted_files')
#os.system('mkdir ' + pred_dir)

# split all subjects into 5 folds
subj_group = arlib.split_subject(sessid, cv_num)
for i in range(cv_num):
    print 'CV iter - ' + str(i)
    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
    
    # split data into training and test group
    test_sessid = subj_group[i]
    train_sessid = []
    for j in range(cv_num):
        if not j == i:
            train_sessid += subj_group[j]

    # Forest training with single subject's data
    forest_list = []
    classes_list = []
    spatial_ptn = None
    for subj in train_sessid:
        #print 'train subject %s'%(subj)
        train_data = arlib.get_list_data([subj], cv_dir)
        z_vtr = train_data[..., 0].copy()
        z_vtr[z_vtr<2.3] = 0
        smp_mask = z_vtr > 0
        train_x = train_data[smp_mask, 0:4]
        train_y = train_data[smp_mask, -1]
        if not isinstance(spatial_ptn, np.ndarray):
            spatial_ptn = np.zeros((train_data.shape[0], len(train_sessid)))
            count = 0
        spatial_ptn[..., count] = z_vtr
        count += 1
        clf = RandomForestClassifier(n_estimators=50, max_depth=20,
                                     criterion='gini', n_jobs=20)
        clf.fit(train_x, train_y)
        forest_list.append(clf)
        #print clf.classes_
        classes_list.append(clf.classes_)

    #print 'Selected classifier number: %s'%(selected_num)
    # Model testing
    ffa_dice_tmp = {}
    ofa_dice_tmp = {}
    for num in selected_num:
        ffa_dice_tmp[num] = []
        ofa_dice_tmp[num] = []

    for subj in test_sessid:
        #print 'Test data - subject %s'%(subj)
        test_data = arlib.get_list_data([subj], cv_dir)
        z_vtr = test_data[..., 0].copy()
        z_vtr[z_vtr < 2.3] = 0
        smp_mask = z_vtr > 0
        test_x = test_data[smp_mask, 0:4]
        test_y = test_data[smp_mask, -1]
        # TODO: define similarity index
        similarity = []
        for j in range(len(train_sessid)):
            #r = np.corrcoef(z_vtr, spatial_ptn[..., j])[0, 1]
            r = arlib.normalized_mutual_info(z_vtr.copy(),
                                             spatial_ptn[..., j].copy(),
                                             2.3)
            if not np.isnan(r):
                similarity.append(r)
            else:
                similarity.append(0)

        # sort the similarity and get first n RFs as classifier
        sorted_similarity_idx = np.argsort(similarity)[::-1]
        for num in selected_num:
            selected_clf = sorted_similarity_idx[0:num]

            pred_prob = None
            #print 'similarity: '
            for clf_idx in selected_clf:
                #print train_sessid[clf_idx],
                #print similarity[clf_idx],
                #print classes_list[clf_idx]
                clf = forest_list[clf_idx]
                prob = clf.predict_proba(test_x)
                std_prob = np.zeros((prob.shape[0], 3))
                for cls_idx in range(prob.shape[1]):
                    if classes_list[clf_idx][cls_idx] == 0:
                        std_prob[..., 0] = prob[..., cls_idx]
                    elif classes_list[clf_idx][cls_idx] == 1:
                        std_prob[..., 1] = prob[..., cls_idx]
                    if classes_list[clf_idx][cls_idx] == 3:
                        std_prob[..., 2] = prob[..., cls_idx]
                new_prob = np.zeros(std_prob.shape)
                new_prob[range(new_prob.shape[0]),
                         np.argmax(std_prob, axis=1)] = 1
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
                #print 'Dice for label %s: %f'%(label_idx, dice_val)
                if label_idx == 3:
                    ffa_dice_tmp[num].append(dice_val)
                else:
                    ofa_dice_tmp[num].append(dice_val)

        ##-- save predicted label as nifti files
        #fsl_dir = os.getenv('FSL_DIR')
        #img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
        #                            'MNI152_T1_2mm_brain.nii.gz'))
        ## save predicted label
        #header = img.get_header()
        #coords = test_x[testy_x, 1:4]
        #pred_data = arlib.write2array(coords, pred_y)
        #out_file = os.path.join(pred_dir, subj + '_pred.nii.gz')
        #mybase.save2nifti(pred_data, header, out_file)
    for num in selected_num:
        ffa_dice[num][i] = np.mean(ffa_dice_tmp[num])
        ofa_dice[num][i] = np.mean(ofa_dice_tmp[num])

for num in selected_num:
    print 'Slected clf num, %d'%(num)
    print 'Mean FFA Dice, %s'%(ffa_dice[num].mean())
    print 'Mean OFA Dice, %s'%(ofa_dice[num].mean())

##-- effect of sample numbers (subjects number)
#out_file = os.path.join(data_dir, 'subj_size_effect.txt')
#f = open(out_file, 'wb')
#f.write('subj_size,samples,label_1,label_3,score,ofa_dice,ffa_dice\n')
## subjects size range
#subj_num = range(20, 160, 10)
## repeat number
#repeat_num = 10
#
## feature type
#features = {}
#features['coord'] = [0, 1, 2]
#features['z_type3'] = range(117, 894, 4)
#feature_idx = features['coord'] + features['z_type3']
#
## split subjets into two groups: training and test dataset
#subj_group = arlib.split_subject(sessid, 10)
#for test_idx in range(10):
#    test_sessid = subj_group[test_idx]
#    train_sessid = []
#    for grp_idx in range(10):
#        if not grp_idx == test_idx:
#            train_sessid += subj_group[grp_idx]
#    # load test data
#    sample_dir = os.path.join(data_dir, 'samples')
#    test_data = arlib.get_list_data(test_sessid, sample_dir)
#    print 'Sample stats for testing data'
#    arlib.samples_stat(test_data)
#    test_x = test_data[..., :-1]
#    test_y = test_data[..., -1]
#    for subj_size in subj_num:
#        print 'Subjects number: %s'%(subj_size)
#        for p in range(repeat_num):
#            print 'repeat %s'%(p)
#            out_str = [str(subj_size)]
#            subj_idx = random.sample(range(len(train_sessid)), subj_size)
#            sel_sessid = [train_sessid[i] for i in subj_idx]
#            sample_data = arlib.get_list_data(sel_sessid, sample_dir)
#            # samples stats
#            print 'Sample stats of dataset:'
#            arlib.samples_stat(sample_data)
#            x = sample_data[..., :-1]
#            y = sample_data[..., -1]
#            label_all = y.shape[0]
#            label_1 = np.sum(y == 1)
#            label_3 = np.sum(y == 3)
#            out_str.append(str(label_all))
#            out_str.append(str(label_1))
#            out_str.append(str(label_3))
#            # model training
#            clf = RandomForestClassifier(n_estimators=60, max_depth=30,
#                                         criterion='gini', n_jobs=20)
#            clf.fit(x, y)
#            pred_y = clf.predict(test_x)
#            score = accuracy_score(test_y, pred_y)
#            print 'Predict score is %s'%(score)

