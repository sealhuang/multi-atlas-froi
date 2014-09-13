# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np

import macro

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'multi-atlas', 'l_ffa_ofa')

class_label = [2, 4]
#atlas_num = [1, 5] + range(10, 201, 10)
#atlas_num = [1, 5]
#atlas_num = [50]
atlas_num = range(1, 201)

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
#for i in range(10):
#    # model training and testing
#    forest_list, classes_list, spatial_ptn = macro.train_model(sessid,
#                                                               data_dir)
#    dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
#                                    forest_list, classes_list, spatial_ptn,
#                                    sorted=False)
#
#    # save dice to a file
#    macro.save_dice(dice, data_dir)

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

#-- relation between atlas rank and Dice
# model training and testing
forest_list, classes_list, spatial_ptn = macro.train_model(sessid, data_dir)
dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
                                forest_list, classes_list, spatial_ptn,
                                single_atlas=True)

# save dice to a file
macro.save_dice(dice, data_dir)

