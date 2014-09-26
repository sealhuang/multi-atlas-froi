# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nibabel as nib
import numpy as np
import re
import scipy.ndimage as ndimage

from mypy import base as mybase

def ext_sample(zstat_file, mask_coord, class_label, label_file=None):
    """
    Sample extraction.

    Input: zstat and label image for a subject, a mask table (voxel
           coordinates in the mask), and a class label list. 
    Output: samples from each subject.

    """
    #-- load data
    if label_file:
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
        if label_file:
            label = label_data[tuple(coord)]
            if not idx:
                sample_label.append('label')
            if label in class_label:
                feature_buff.append(label)
            else:
                feature_buff.append(0)
        
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

def make_prior(subj_list, class_label, output_dir):
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
        uniq_val = np.unique(temp)
        for val in uniq_val:
            if not val in class_label:
                temp[temp==val] = 0
        temp[temp>0] = 1
        mask_data += temp
    mask_data[mask_data>0] = 1
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

def ext_subj_feature(sid, mask_coord, class_label, out_dir):
    """
    Warper of ext_sample and save_sample.

    """
    # zstat and label file
    db_dir = r'/nfs/t2/atlas/database'
    subject_dir = os.path.join(db_dir, sid, 'face-object')
    if not os.path.exists(subject_dir):
        print 'Subject ' + sid + 'does not exist in database.'
        return
    zstat_file = os.path.join(subject_dir, 'zstat1.nii.gz')
    label_file = get_label_file(subject_dir)

    sample_label, subj_data = ext_sample(zstat_file,
                                         mask_coord, class_label,
                                         label_file=label_file)
    out_file = os.path.join(out_dir, sid + '_data.csv')
    save_sample(sample_label, subj_data, out_file)

def get_subj_data(subj_list, data_dir):
    """
    Get data for a list of subjects.
    #subj_list# could be either a subject's SID or a SID list.

    """
    if isinstance(subj_list, str):
        subj_list = [subj_list]
    if not isinstance(subj_list, list):
        print 'Invalid input!'
        return
    samples = np.array([])
    for subj in subj_list:
        f = os.path.join(data_dir, subj + '_data.csv')
        data = np.loadtxt(f, skiprows=1, delimiter=',')
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

