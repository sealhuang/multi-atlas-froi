# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import time

# modules for data preparation
import multiprocessing as mps
import functools

import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'multi-atlas', 'data', 'all')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

## generate mask and probability map
#prob_data, mask_data = arlib.make_prior(sessid, data_dir)
#mask_coords = arlib.get_mask_coord(mask_data, data_dir)
#
## extract features from each subject
#pool = mps.Pool(20)
#result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                    mask_coord=mask_coords,
#                                    out_dir=data_dir, mask_out=False),
#                  sessid)

spatial_ptn = None
for subj in sessid:
    temp_data = arlib.get_list_data([subj], data_dir)
    if not isinstance(spatial_ptn, np.ndarray):
        spatial_ptn = np.zeros((temp_data.shape[0], len(sessid)))
        count = 0
    spatial_ptn[..., count] = temp_data[..., 0]
    count += 1

print spatial_ptn.shape
np.savetxt('raw.txt', spatial_ptn, delimiter=',')

#r_mtx = np.corrcoef(spatial_ptn.T)
#print r_mtx.shape
#np.savetxt('r.txt', r_mtx, delimiter=',')

