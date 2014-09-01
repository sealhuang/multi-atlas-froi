# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nibabel as nib
import numpy as np
import re
from scipy.spatial.distance import euclidean
import time
import scipy.ndimage as ndimage

from mypy import base as mybase

def ext_feature(sid, mask_coord, mask_out=True):
    """
    Feature extraction.

    Input: Subject ID and a mask data (voxel coordinates in the mask)
    Source data: subject-specific zstat map and subject-specific label file
    Output: samples per subject.

    """
    #-- data preparation
    # zstat nad label file
    db_dir = r'/nfs/t2/atlas/database'
    subject_dir = os.path.join(db_dir, sid, 'face-object')
    if not os.path.exists(subject_dir):
        print 'Subject ' + sid + 'does not exist in database.'
        return
    zstat_file = os.path.join(subject_dir, 'zstat1.nii.gz')
    label_file = get_label_file(subject_dir)

    # load data
    label_data = nib.load(label_file).get_data()
    zstat_data = nib.load(zstat_file).get_data()

    #-- extract features for each voxel in the mask
    sample_data = []
    sample_label = [] 
    sample_num = len(mask_coord)
    
    for idx in range(sample_num):
        feature_buff = []

        # voxel coordinates
        coord = mask_coord[idx]

        # generate flag
        flag = mask_out and zstat_data[tuple(coord)] < 2.3

        if idx and flag:
            continue

        if not idx:
            sample_label.append('z_val')
        feature_buff.append(zstat_data[tuple(coord)])

        if not idx:
            sample_label.append('coord_x')
        feature_buff.append(coord[0])
        if not idx:
            sample_label.append('coord_y')
        feature_buff.append(coord[1])
        if not idx:
            sample_label.append('coord_z')
        feature_buff.append(coord[2])
        
        # get voxel label
        label = label_data[tuple(coord)]
        if not idx:
            sample_label.append('label')
        if label == 1 or label == 3:
            feature_buff.append(label)
        else:
            feature_buff.append(0)
        
        # store feature vector in the data matrix
        if not flag:
            sample_data.append(feature_buff)

    return sample_label, sample_data

def get_label_file(subject_dir):
    """
    Return subject-specific label file.

    """
    f_list = os.listdir(subject_dir)
    for f in f_list:
        if re.search('_ff.nii.gz', f):
            return os.path.join(subject_dir, f)

def get_neighbor_offset(radius):
    """
    Get neighbor offset for generating cubiods.

    """
    offsets = []
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                offsets.append([x, y, z])
    return np.array(offsets)

def get_mean(data, coords):
    """
    Get mean value of the input voxels.

    """
    val = 0
    for coord in coords:
        val += data[tuple(coord)]
    return val / len(coords)

def make_prior(subj_list, output_dir):
    """
    Create a label mask derived from a group of subjects.

    """
    print 'Create a whole-fROI mask ...'
    db_dir = r'/nfs/t2/atlas/database'
    subj_num = len(subj_list)
    for i in range(subj_num):
        sid = subj_list[i]
        subj_dir = os.path.join(db_dir, sid, 'face-object')
        label_file = get_label_file(subj_dir)
        label_data = nib.load(label_file).get_data()
        img_header = nib.load(label_file).get_header()
        if not i:
            mask_data = np.zeros(label_data.shape)
        temp = label_data.copy()
        temp[temp==1] = 100
        temp[temp==3] = 100
        temp[temp!=100] = 0
        temp[temp>0] = 1
        mask_data += temp
    mask_data[mask_data!=0] = 1
    # save to file
    mask_file = os.path.join(output_dir, 'mask.nii.gz')
    mybase.save2nifti(mask_data, img_header, mask_file)
    # return data
    return mask_data

def get_mask_coord(mask_data, output_dir):
    """
    Get coordinates of the mask.

    """
    coords = mask_data.nonzero()
    vxl_num = coords[0].shape[0]
    c = [[coords[0][i], coords[1][i], coords[2][i]] for i in range(vxl_num)]
    output_file = os.path.join(output_dir, 'mask_coords.csv')
    f = open(output_file, 'wb')
    f.write('x,y,z\n')
    for line in c:
        strline = [str(item) for item in line]
        f.write(','.join(strline) + '\n')
    return c

def split_subject(subject_list, group_number):
    """
    Split all subjects into n groups.

    """
    subj_num = len(subject_list)
    x = np.arange(subj_num)
    y = np.array_split(x, group_number)
    subject_group = []
    for l in y:
        sid = [subject_list[item] for item in l]
        subject_group.append(sid)
    return subject_group

def save_subject_group(subject_group, output_dir):
    """
    Save subject group into file.

    """
    for idx in range(len(subject_group)):
        l = subject_group[idx]
        out_file = os.path.join(output_dir, 'sessid_' + str(idx))
        f = open(out_file, 'wb')
        for item in l:
            f.write(item + '\n')

def save_sample(sample_label, sample_data, out_file):
    """
    Save sample data into a file.

    """
    f = open(out_file, 'wb')
    f.write(','.join(sample_label) + '\n')
    for line in sample_data:
        strline = [str(item) for item in line]
        f.write(','.join(strline) + '\n')
    f.close()

def ext_subj_feature(sid, mask_coord=None, out_dir=None, mask_out=True):
    """
    Warper of ext_feature nad save_sample.

    """
    sample_label, subj_data = ext_feature(sid, mask_coord, mask_out)
    out_file = os.path.join(out_dir, sid + '_data.csv')
    save_sample(sample_label, subj_data, out_file)

def get_subj_data(subj_file):
    """
    Get samples from individual subject.

    """
    return np.loadtxt(subj_file, skiprows=1, delimiter=',')

def get_list_data(subj_list, data_dir):
    """
    Get data for a list of subjects.

    """
    samples = np.array([])
    for subj in subj_list:
        f = os.path.join(data_dir, subj + '_data.csv')
        data = get_subj_data(f)
        if not samples.size:
            samples = data
        else:
            samples = np.vstack((samples, data))
    return samples

def samples_stat(samples):
    """
    A brief stats of categories of the samples.

    """
    labels = samples[..., -1]
    uniq_label = np.unique(labels)
    for val in uniq_label:
        print str(val) + '; ',
        print np.sum(labels == val)

def get_label(label_file):
    """
    Get feature of label of samples.

    """
    label = open(label_file).readlines()
    label = [line.strip() for line in label]
    label = label[0].split(',')
    label.pop(-1)
    return label

def write2array(coords, voxel_val):
    """
    Write the voxel_val into a nifti file based on the coordinates.

    """
    data = np.zeros((91, 109, 91))
    x_coord = tuple(coords[:, 0])
    y_coord = tuple(coords[:, 1])
    z_coord = tuple(coords[:, 2])
    data[x_coord, y_coord, z_coord] = voxel_val
    return data

def get_subj_sample_num(stats_file):
    """
    Get sample size for each subject.

    """
    info = open(stats_file).readlines()
    info = [int(line.strip().split()[1]) for line in info]
    return np.array(info)

def offset_dist(offset_1_idx, offset_2_idx):
    """
    Get Euler distance between two different offset-seed voxel.

    """
    # seed_offset_vtr
    offset_len = 4
    seed_offset_vtr_x = [[i, 0, 0] for i in
                         range(-offset_len, offset_len+1) if i]
    seed_offset_vtr_y = [[0, i, 0] for i in 
                         range(-offset_len, offset_len+1) if i]
    seed_offset_vtr_z = [[0, 0, i] for i in 
                         range(-offset_len, offset_len+1) if i]
    seed_offset_vtr = np.array(seed_offset_vtr_x + \
                               seed_offset_vtr_y + \
                               seed_offset_vtr_z)
    offset_1 = seed_offset_vtr[offset_1_idx]
    offset_2 = seed_offset_vtr[offset_2_idx]
    return euclidean(offset_1, offset_2)

def smooth_data(data, sigma):
    """
    Smooth each 3D volume in input data.

    """
    dim = len(data.shape)
    if dim == 4:
        '4D data input ...'
        vol_num = data.shape[3]
        for i in range(vol_num):
            data[..., i] = ndimage.gaussian_filter(data[..., i], sigma)
    else:
        data = ndimage.gaussian_filter(data, sigma)
    return data

