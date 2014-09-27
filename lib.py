# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nibabel as nib
import numpy as np

from mypy import base as mybase

def ext_sample(zstat_file, mask_coord, class_label, label_file=None):
    """
    Sample extraction.

    Input: zstat image from a subject, a mask table (voxel coordinates in
           the mask), and a label list are required.
           For training dataset, a corresponding label image is also needed.
    Output: samples from each subject.

    """
    #-- load data
    if label_file:
        label_data = nib.load(label_file).get_data()
    zstat_data = nib.load(zstat_file).get_data()

    #-- extract features for each voxel in the mask
    sample_data = []
    feature_name = [] 
    sample_num = len(mask_coord)
    
    for idx in range(sample_num):
        feature_buff = []

        # voxel coordinates
        coord = mask_coord[idx]

        if not idx:
            feature_name.append('z_val')
        feature_buff.append(zstat_data[tuple(coord)])
        if not idx:
            feature_name.append('coord_x')
        feature_buff.append(coord[0])
        if not idx:
            feature_name.append('coord_y')
        feature_buff.append(coord[1])
        if not idx:
            feature_name.append('coord_z')
        feature_buff.append(coord[2])
        
        # get voxel label
        if label_file:
            label = label_data[tuple(coord)]
            if not idx:
                feature_name.append('label')
            if label in class_label:
                feature_buff.append(label)
            else:
                feature_buff.append(0)
        
        sample_data.append(feature_buff)

    return feature_name, sample_data

def make_mask(label_file_list, class_label, output_dir):
    """
    Create a label mask derived from a group of label files.

    """
    print 'Create a whole-fROI mask ...'
    num = len(label_file_list)
    for i in range(num):
        label_data = nib.load(label_file_list[i]).get_data()
        if not i:
            mask_data = np.zeros(label_data.shape)
            header = nib.load(label_file_list[i]).get_header()
        uniq_val = np.unique(label_data)
        for val in uniq_val:
            if not val in class_label:
                label_data[label_data==val] = 0
        label_data[label_data>0] = 1
        mask_data += label_data
    mask_data[mask_data>0] = 1
    # save to file
    mask_file = os.path.join(output_dir, 'mask.nii.gz')
    mybase.save2nifti(mask_data, header, mask_file)
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

def save_sample(feature_name, sample_data, out_file):
    """
    Save sample data into a file.

    """
    f = open(out_file, 'wb')
    f.write(','.join(feature_name) + '\n')
    for line in sample_data:
        strline = [str(item) for item in line]
        f.write(','.join(strline) + '\n')
    f.close()

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

