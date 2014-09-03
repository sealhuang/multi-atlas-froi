# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np

import macro

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'multi-atlas', 'data', 'l_ffa_ofa')

class_label = [2, 4]
atlas_num = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
#atlas_num = [1, 5]

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# extract samples
macro.extract_sample(sessid, class_label, data_dir)

# model training and testing
forest_list, classes_list, spatial_ptn = macro.train_model(sessid, data_dir)
dice = macro.leave_one_out_test(sessid, atlas_num, data_dir, class_label,
                                forest_list, classes_list, spatial_ptn)

# save dice to a file
macro.save_dice(dice, data_dir)

