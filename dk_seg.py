# -*- coding: utf-8 -*-
"""
"""


from __future__ import division

import numpy as np
import SimpleITK as sitk
#import os
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import neurite as ne




def shapes_match(shapes_list):
    '''Checks if shapes in the provided list are all the same'''
    
    match= True
    
    n_shapes= len(shapes_list)
    
    for i in range(n_shapes):
        for j in range(1,n_shapes):
            
            shape_1= shapes_list[i]
            shape_2= shapes_list[j]
            
            if not shape_1==0 and not shape_2==0:
                if not shape_1==shape_2:
                    match= False
                    
    return match



def shapes_are_close(shapes_list, voxel_tolerance= 2):
    
    '''Checks if shapes in the provided list are all close'''
    
    match= True
    
    n_shapes= len(shapes_list)
    
    for i in range(n_shapes):
        for j in range(1,n_shapes):
            
            shape_1= shapes_list[i]
            shape_2= shapes_list[j]
            
            if not shape_1==0 and not shape_2==0:
                
                for i in range(len(shape_1)):
                    
                    if shape_1[i]<shape_2[i]-voxel_tolerance or \
                       shape_1[i]>shape_2[i]+voxel_tolerance:
                        match= False
                    
    return match



def dice(x1, x2, eps=1e-3):
    
    if np.mean(x1==0)+ np.mean(x1==1)<1 or np.mean(x2==0)+ np.mean(x2==1)<1:
        print('The arrays should include ones and zeros only')
        val= None
    
    else:
        dice_num = 2 * np.sum(( x1 == 1) * (x2 == 1)) 
        dice_den = np.sum(x1 == 1) + np.sum(x2 == 1)
        #den_zero= dice_den==0
        val= ( dice_num + eps ) / ( dice_den + eps )
    
    return val







def dice_continuous(x1, x2, eps=1e-3):
    
    dice_num = 2 * np.sum(( x1 ) * (x2 )) 
    dice_den = np.sum(x1 * x2 )
    val= ( dice_num + eps ) / ( dice_den + eps )
    
    return val





def seg_2_boundary_3d(x):
    
    a, b, c= x.shape
    
    y= np.zeros(x.shape)
    z= np.nonzero(x)
    
    if len(z[0])>1:
        x_sum= np.zeros(x.shape)
        for shift_x in range(-1, 2):
            for shift_y in range(-1, 2):
                for shift_z in range(-1, 2):
                    x_sum[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
        y= np.logical_and( x==1 , np.logical_and( x_sum>0, x_sum<27 ) )
        
    return y


def hausdorff_3d(y_true, y_pred):
    
    y_true_b = seg_2_boundary_3d(y_true)
    y_pred_b = seg_2_boundary_3d(y_pred)
    
    z = np.nonzero(y_true_b)
    zx, zy, zz = z[0], z[1], z[2]
    contour_true = np.vstack([zx, zy, zz]).T
    z = np.nonzero(y_pred_b)
    zx, zy, zz = z[0], z[1], z[2]
    contour_pred = np.vstack([zx, zy, zz]).T
    
    err = max(directed_hausdorff(contour_true, contour_pred)[0],
              directed_hausdorff(contour_pred, contour_true)[0])
    
    return err


def asd_3d(y_true, y_pred, n= 100):
    
    y_true_b = seg_2_boundary_3d(y_true)
    y_pred_b = seg_2_boundary_3d(y_pred)
    
    z = np.nonzero(y_true_b)
    zx, zy, zz = z[0], z[1], z[2]
    contour_true = np.vstack([zx, zy, zz]).T
    z = np.nonzero(y_pred_b)
    zx, zy, zz = z[0], z[1], z[2]
    contour_pred = np.vstack([zx, zy, zz]).T
    
    n_true= contour_true.shape[0]
    n_pred= contour_pred.shape[0]
    
    d_true= np.zeros(n_true)
    d_pred= np.zeros(n_pred)
    
    for i in range(n_true//n):
        d= cdist(contour_true[i*n:(i+1)*n,:], contour_pred)
        d_true[i*n:(i+1)*n]= np.min(d, axis=1)
    d= cdist(contour_true[(i+1)*n:,:], contour_pred)
    d_true[(i+1)*n:]= np.min(d, axis=1)
    
    for i in range(n_pred//n):
        d= cdist(contour_pred[i*n:(i+1)*n,:], contour_true)
        d_pred[i*n:(i+1)*n]= np.min(d, axis=1)
    d= cdist(contour_pred[(i+1)*n:,:], contour_true)
    d_pred[(i+1)*n:]= np.min(d, axis=1)
    
    err= ( d_true.sum() + d_pred.sum() )/ (n_true+n_pred)
    
    return err



def resample3d(x, original_spacing, new_spacing, resampling_mode):
    '''Resamples the given array into new spacing'''
    
    original_size = x.GetSize()
    I_size = [int(spacing/new_s*size) 
                            for spacing, size, new_s in zip(original_spacing, original_size, new_spacing)] 
    
    I = sitk.Image(I_size, x.GetPixelIDValue())
    I.SetSpacing(new_spacing)
    I.SetOrigin(x.GetOrigin())
    I.SetDirection(x.GetDirection())
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(I)
    resample.SetInterpolator(resampling_mode)
    resample.SetTransform(sitk.Transform())
    I = resample.Execute(x)
    
    return I






def resample3d_exactshape(x, original_spacing, new_spacing, ref_size, resampling_mode, delta=0.999, verbose=True):
    '''Resamples the given array into new spacing'''
    
    original_spacing= np.array(original_spacing)
    
    original_spacing_copy= original_spacing.copy()
    
    resample_size= (-1, -1, -1)
    
    resample_trials= 0
    
    while not resample_size==ref_size:
        
        resample_trials+= 1
        
        original_size = x.GetSize()
        I_size = [int(spacing/new_s*size) 
                                for spacing, size, new_s in zip(original_spacing, original_size, new_spacing)] 
        
        I = sitk.Image(I_size, x.GetPixelIDValue())
        I.SetSpacing(new_spacing)
        I.SetOrigin(x.GetOrigin())
        I.SetDirection(x.GetDirection())
        
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(I)
        resample.SetInterpolator(resampling_mode)
        resample.SetTransform(sitk.Transform())
        I = resample.Execute(x)
        
        resample_size= I.GetSize()
        
        for i in range(3):
            if resample_size[i]<ref_size[i]:
                original_spacing[i]/= delta
            elif resample_size[i]>ref_size[i]:
                original_spacing[i]*= delta
    
    if resample_trials>1 and verbose:
        print('Resampled more than once: ', resample_trials)
        print('Original and new spacing: ', original_spacing_copy, ' ', original_spacing)
    
    return I













def normalize_with_mask(img, mask=None):
    '''normalizes image to have zero mean and unit variance
    if ninary mask is given, only values inside the mask are used and the rest 
    of the image is set to zero'''
    
    temp_img = img[mask >= 0.5]
    temp_mean = temp_img.mean()
    temp_std = temp_img.std()
    img[mask >= 0.5] = img[mask >= 0.5] - temp_mean
    img[mask >= 0.5] = img[mask >= 0.5] / temp_std
    img[mask < 0.5] = 0
    
    return img, temp_mean, temp_std





def number_n_size_of_bboxes(seg):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    seg_b_marked= np.zeros( seg.shape )
    n = seg.max()
    s = np.zeros(n)
    box= np.zeros((n,6))
    
    for i in range(1, n + 1):
        
        s[i-1]= np.sum((seg == i))
        z= np.where(seg == i)
        box[i-1,:]= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        seg_b_marked[ z[0].min() : z[0].max()+1 , z[1].min() : z[1].max()+1 , z[2].min() : z[2].max()+1 ] = i
        
    return n, s, box, seg_b_marked




def number_n_size_of_components(seg, return_seg=False):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n = seg.max()
    s = np.zeros(n)
    
    for i in range(1, n + 1):
        
        s[i-1]= np.sum((seg == i))
    
    if return_seg:
        
        return n, s, seg
    
    else:
        
        return n, s


def seg_2_components(seg):

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)

    return seg


def seg_statistics(seg_true, seg_pred):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg_true= sitk.GetImageFromArray(seg_true)
    seg_true = c_filter.Execute(seg_true)
    seg_true= sitk.GetArrayFromImage(seg_true)

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg_pred = sitk.GetImageFromArray(seg_pred)
    seg_pred = c_filter.Execute(seg_pred)
    seg_pred= sitk.GetArrayFromImage(seg_pred)
    
    n_true= seg_true.max()
    n_pred= seg_pred.max()
    
    TP= FP= FN= 0
    
    for i in range(1, n_true+1):
        
        if np.sum(( seg_true == i) * (seg_pred > 0 )) > 0:
            TP+= 1
        else:
            FN+= 1
            
    for i in range(1, n_pred+1):
        
        if np.sum(( seg_pred == i) * (seg_true > 0 )) == 0:
            FP+= 1
    
    return TP, FP, FN



def seg_diff(seg_true, seg_pred):
    
    seg_true_copy = seg_true.copy()
    seg_pred_copy = seg_pred.copy()
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg_true = sitk.GetImageFromArray(seg_true)
    seg_true = c_filter.Execute(seg_true)
    seg_true = sitk.GetArrayFromImage(seg_true)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg_pred = sitk.GetImageFromArray(seg_pred)
    seg_pred = c_filter.Execute(seg_pred)
    seg_pred = sitk.GetArrayFromImage(seg_pred)
    
    n_true = seg_true.max()
    n_pred = seg_pred.max()
    
    for i in range(1, n_true + 1):
        
        if np.sum((seg_true == i) * (seg_pred > 0)) > 0:
            seg_true_copy[seg_true == i] = 0
            
    for i in range(1, n_pred + 1):
        
        if np.sum((seg_pred == i) * (seg_true > 0)) > 0:
            seg_pred_copy[seg_pred == i] = 0
            
    return seg_true_copy, seg_pred_copy



def tumor_statistics(seg, t1, t2, fl):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_tumor = seg.max()
    
    mean= np.zeros( (n_tumor, 3) )
    std = np.zeros((n_tumor, 3))
    
    for i in range(1, n_tumor + 1):
        
        temp = t1[ seg==i ]
        mean[i-1,0] = temp.mean()
        std[i-1,0] = temp.std()
        
        temp = t2[seg == i]
        mean[i-1, 1] = temp.mean()
        std[i-1, 1] = temp.std()
        
        temp = fl[seg == i]
        mean[i-1, 2] = temp.mean()
        std[i-1, 2] = temp.std()
        
    return mean, std




def tumor_statistics_kars(seg, t1, t2, fl):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg_kars = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg_kars)
    
    n_tumor = seg.max()
    
    tumor_stats= np.zeros( (n_tumor, 7) )
    
    for i in range(1, n_tumor + 1):
        
        ind= np.where(seg==i)
        tumor_stats[i-1,0]= len(ind[0])
        
        temp = t1[ seg==i ]
        tumor_stats[i-1,1] = temp.mean()
        tumor_stats[i-1,2] = temp.std()
        
        temp = t2[seg == i]
        tumor_stats[i-1, 3] = temp.mean()
        tumor_stats[i-1, 4] = temp.std()
        
        temp = fl[seg == i]
        tumor_stats[i-1, 5] = temp.mean()
        tumor_stats[i-1, 6] = temp.std()
        
    return seg_kars, tumor_stats



def tumor_statistics_uncertainty(seg, t1, t2, fl, uncertainty):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_tumor = seg.max()
    
    tumor_stats= np.zeros( (n_tumor, 9) )
    
    for i in range(1, n_tumor + 1):
        
        temp = t1[ seg==i ]
        tumor_stats[i-1,0] = temp.mean()
        tumor_stats[i-1,4] = temp.std()
        
        temp = t2[seg == i]
        tumor_stats[i-1, 1] = temp.mean()
        tumor_stats[i-1, 5] = temp.std()
        
        temp = fl[seg == i]
        tumor_stats[i-1, 2] = temp.mean()
        tumor_stats[i-1, 6] = temp.std()
        
        temp = uncertainty[seg == i]
        tumor_stats[i-1, 3] = temp.mean()
        tumor_stats[i-1, 7] = temp.std()
        
        ind= np.where(seg==i)
        tumor_stats[i-1,8]= len(ind[0])
         
    return tumor_stats



def tumor_statistics_uncertainty_distance(seg, t1, t2, fl, uncertainty, seg_true):

    seg_true_org = seg_true.copy()

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)

    n_tumor = seg.max()

    tumor_stats = np.zeros((n_tumor, 10))

    for i in range(1, n_tumor + 1):

        temp = t1[seg == i]
        tumor_stats[i - 1, 0] = temp.mean()
        tumor_stats[i - 1, 4] = temp.std()

        temp = t2[seg == i]
        tumor_stats[i - 1, 1] = temp.mean()
        tumor_stats[i - 1, 5] = temp.std()

        temp = fl[seg == i]
        tumor_stats[i - 1, 2] = temp.mean()
        tumor_stats[i - 1, 6] = temp.std()

        temp = uncertainty[seg == i]
        tumor_stats[i - 1, 3] = temp.mean()
        tumor_stats[i - 1, 7] = temp.std()

        ind = np.where(seg == i)
        tumor_stats[i - 1, 8] = len(ind[0])

    for i in range(1, n_tumor + 1):

        seg_true_temp = seg_true_org.copy()
        seg_true_temp[seg == i] = 0
        b_true_temp = np.int8(seg_2_boundary_3d(seg_true_temp.copy()))
        b_image = sitk.GetImageFromArray(b_true_temp)
        d_image = sitk.SignedMaurerDistanceMap(b_image, insideIsPositive=True, useImageSpacing=True,
                                               squaredDistance=False)
        d_true_temp = sitk.GetArrayFromImage(d_image)
        d_cur = -d_true_temp * (seg == i)
        d_cur = d_cur[seg == i]
        tumor_stats[i - 1, -1] = d_cur.min()

    return tumor_stats




def clean_fp(seg, t1, t2, fl, uncertainty, seg_true, clf, CLEAN_PROB_THRESH):

    fp_clean= seg.copy()

    seg_true_org = seg_true.copy()

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)

    n_tumor = seg.max()

    print('Total number of FPs: ', n_tumor)

    for i in range(1, n_tumor + 1):

        f_vector = np.zeros((1, 6))

        temp = t1[seg == i]
        f_vector[0, 0] = temp.mean()

        temp = t2[seg == i]
        f_vector[0, 1] = temp.mean()

        temp = fl[seg == i]
        f_vector[0, 2] = temp.mean()

        temp = uncertainty[seg == i]
        f_vector[0, 3] = temp.mean()

        ind = np.where(seg == i)
        f_vector[0, 4] = len(ind[0])

        seg_true_temp = seg_true_org.copy()
        # seg_true_temp[seg == i] = 0
        b_true_temp = np.int8(seg_2_boundary_3d(seg_true_temp.copy()))
        b_image = sitk.GetImageFromArray(b_true_temp)
        d_image = sitk.SignedMaurerDistanceMap(b_image, insideIsPositive=True, useImageSpacing=True,
                                               squaredDistance=False)
        d_true_temp = sitk.GetArrayFromImage(d_image)
        d_cur = -d_true_temp * (seg == i)
        d_cur = d_cur[seg == i]
        f_vector[0, 5] = d_cur.min()

        seg_prob= clf.predict_proba(f_vector)[0][1]

        if seg_prob<CLEAN_PROB_THRESH:

            fp_clean[seg == i]= 0

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg = sitk.GetImageFromArray(fp_clean)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)

    n_tumor = seg.max()

    print('Number of FPs kept: ', n_tumor)

    return fp_clean





def return_tp_fn_fp_images(seg_true, seg_pred):

    tp_img= np.zeros(seg_true.shape)
    fp_img= np.zeros(seg_true.shape)
    fn_img= np.zeros(seg_true.shape)
     
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg_true= sitk.GetImageFromArray(seg_true)
    seg_true = c_filter.Execute(seg_true)
    seg_true = sitk.GetArrayFromImage(seg_true)

    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()

    seg_pred= sitk.GetImageFromArray(seg_pred)
    seg_pred = c_filter.Execute(seg_pred)
    seg_pred= sitk.GetArrayFromImage(seg_pred)
     
    n_true= seg_true.max()
    n_pred= seg_pred.max()
     
    for i in range(1, n_true+1):
         
        if np.sum(( seg_true == i) * (seg_pred > 0.5 )) > 0:
            tp_img[seg_true == i]= 1
        else:
            fn_img[seg_true == i]= 1
             
    for i in range(1, n_pred+1):
         
        if np.sum(( seg_pred == i) * (seg_true > 0.5 )) == 0:
            fp_img[seg_pred == i]= 1
     
    return tp_img, fn_img, fp_img




def remove_small_lesions(seg, min_size=10, verbose= False):
    
    seg= np.uint(seg>0)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_lesion = seg.max()
    
    for i in range(1, n_lesion + 1):
        
        if np.sum(seg == i) < min_size:
            if verbose:
                print('Found one, ' , np.sum(seg == i))
            seg[seg == i] = 0

    return (seg>0).astype(np.int8)



def small_lesion_sampling(seg, max_lesion_size= 500):
    
    sampleing_coord= np.zeros((1000000, 3))
    count= 0
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_lesion = seg.max()
    
    for i in range(1, n_lesion + 1):
        
        temp = np.where(seg == i)
        
        count_c= len(temp[0])
        
        if count_c<max_lesion_size:
            
            n_c= int( np.max([1, 1000/count_c]))
            
            for j in range(n_c):
                
                sampleing_coord[count:count + count_c, 0] = temp[0]
                sampleing_coord[count:count + count_c, 1] = temp[1]
                sampleing_coord[count:count + count_c, 2] = temp[2]
                
                count += count_c
                
    return sampleing_coord[:count+1,:].astype(np.int)







def keep_largest_connected_component(seg):
    
    seg= np.uint(seg>0)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_cc = seg.max()
    
    if n_cc<2:
        
        return seg
    
    else:
        
        labels= np.arange(1,n_cc+1)
        cc_sizes= [np.sum(seg==label) for label in labels]
        
        cc_ind= np.argmax(cc_sizes)
        label= labels[cc_ind]
        
        seg_z= np.zeros(seg.shape, np.int)
        seg_z[seg==label]= 1
        
        return seg_z
    
    







def keep_largest_connected_component_multilabel(seg_orig):
    
    mask= np.uint(seg_orig>0)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    mask = sitk.GetImageFromArray(mask)
    mask = c_filter.Execute(mask)
    mask = sitk.GetArrayFromImage(mask)
    
    n_cc = mask.max()
    
    if n_cc<2:
        
        return seg_orig
    
    else:
        
        cc_sizes= [np.sum(mask==i+1) for i in range(n_cc)]
        
        cc_ind= np.argmax(cc_sizes)+1
        
        mask_new= np.zeros(mask.shape, np.int)
        mask_new[mask==cc_ind]= 1
        
        return seg_orig*mask_new
    
    





def remove_small_lesions_multilabel(seg_orig, min_size):
    
    mask= np.uint(seg_orig>0)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    mask = sitk.GetImageFromArray(mask)
    mask = c_filter.Execute(mask)
    mask = sitk.GetArrayFromImage(mask)
    
    n_cc = mask.max()
    
    for i in range(1, n_cc + 1):
        
        if np.sum(mask == i) < min_size:
            
            seg_orig[mask == i] = 0
            
    return seg_orig
    
    
    




def augment_batch(batch_x, batch_y, epoch_i, APPLY_DEFORMATION, EPOCH_BEGIN_DEFORMATION, alpha,
                  APPLY_SHIFT, EPOCH_BEGIN_SHIFT, shift_x, shift_y, shift_z,
                  ADD_NOISE, EPOCH_BEGIN_NOISE, noise_sigma,
                  ADD_FLIP, EPOCH_BEGIN_FLIP):
    
    batch_x_mean= np.mean(batch_x).astype(np.float64)
    
    if APPLY_DEFORMATION and epoch_i > EPOCH_BEGIN_DEFORMATION:
        
        x0 = batch_x[0, :, :, :, 0].copy()
        y = batch_y[0, :, :, :, :].copy()
        y = np.argmax(y, axis=-1)
        
        x0 = np.transpose(x0, [2, 1, 0])
        x0 = sitk.GetImageFromArray(x0)
        y = np.transpose(y, [2, 1, 0])
        y = sitk.GetImageFromArray(y)
        
        grid_physical_spacing = [25.0, 25.0, 25.0]
        image_physical_size = [size * spacing for size, spacing in zip(x0.GetSize(), x0.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5) \
                     for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        
        tx = sitk.BSplineTransformInitializer(x0, mesh_size)
        
        direction_size = (mesh_size[0] + 3) * (mesh_size[1] + 3) * (mesh_size[2] + 3) * 3
        
        direction = alpha * np.random.randn(direction_size)
        tx.SetParameters(direction)
        
        xx0 = sitk.Resample(x0, x0, tx, sitk.sitkBSpline, batch_x_mean, x0.GetPixelIDValue())
        yy = sitk.Resample(y, y, tx, sitk.sitkNearestNeighbor, 0.0, y.GetPixelIDValue())
        
        x0 = sitk.GetArrayFromImage(xx0)
        x0 = np.transpose(x0, [2, 1, 0])
        
        y = sitk.GetArrayFromImage(yy)
        y = np.transpose(y, [2, 1, 0])
        
        batch_x[0, :, :, :, 0] = x0.copy()
        
        batch_y = np.zeros(batch_y.shape)
        n_class= batch_y.shape[-1]
        for i_class in range(n_class):
            ind_matrix = (y == i_class)
            batch_y[0, ind_matrix, i_class] = 1
            
        x0 = y = xx = yy = 0
        
    if APPLY_SHIFT and epoch_i > EPOCH_BEGIN_SHIFT:
        
        x = batch_x[0, :, :, :, :].copy()
        y = batch_y[0, :, :, :, :].copy()
        
        sx, sy, sz, n_channel= x.shape
        _ , _ , _ , n_class=   y.shape
        
        #xx = np.zeros((sx + shift_x, sy + shift_y, sz + shift_z, n_channel), np.float32)
        xx = batch_x_mean* np.ones((sx + shift_x, sy + shift_y, sz + shift_z, n_channel), np.float32)
        xx[shift_x // 2:shift_x // 2 + sx, shift_y // 2:shift_y // 2 + sy, shift_z // 2:shift_z // 2 + sz,:] = x.copy()
        yy = np.zeros((sx + shift_x, sy + shift_y, sz + shift_z, n_class), np.float32)
        yy[:,:,:,0]= 1.0
        yy[shift_x // 2:shift_x // 2 + sx, shift_y // 2:shift_y // 2 + sy, shift_z // 2:shift_z // 2 + sz,:] = y.copy()
        
        shift_xx = np.random.randint(shift_x)
        shift_yy = np.random.randint(shift_y)
        shift_zz = np.random.randint(shift_z)
        
        batch_x[0, :, :, :, :] = xx[shift_xx:shift_xx + sx, shift_yy:shift_yy + sy, shift_zz:shift_zz + sz,:].copy()
        batch_y[0, :, :, :, :] = yy[shift_xx:shift_xx + sx, shift_yy:shift_yy + sy, shift_zz:shift_zz + sz,:].copy()
        
        x = y = xx = yy = 0
        
    if ADD_NOISE and epoch_i > EPOCH_BEGIN_NOISE:
        
        batch_size, sx, sy, sz, n_channel= batch_x.shape
        
        batch_x += np.random.randn( batch_size, sx, sy, sz, n_channel ) * noise_sigma
    
    if ADD_FLIP and epoch_i > EPOCH_BEGIN_FLIP:
        
        p_temp= np.random.rand()
        if p_temp>0.5:
            batch_x = batch_x[:, ::-1, :, :, :]
            batch_y = batch_y[:, ::-1, :, :, :]
        
        p_temp= np.random.rand()
        if p_temp>0.5:
            batch_x = batch_x[:, : , ::-1, :, :]
            batch_y = batch_y[:, : , ::-1, :, :]
        
        p_temp= np.random.rand()
        if p_temp>0.5:
            batch_x = batch_x[:, : , :, ::-1, :]
            batch_y = batch_y[:, : , :, ::-1, :]
    
    return batch_x, batch_y


#def crop_vol_n_seg(vol, seg, SX, SY, SZ):
#
#    sx, sy, sz = vol.shape
#
#    z = np.where(seg > 0)
#    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
#
#    if sx > SX:
#        x_0 = max(0, (x_min + x_max) // 2 - SX // 2)
#        x_0 = min(x_0, sx - SX - 1)
#        vol = vol[x_0:x_0 + SX, :, :]
#        seg = seg[x_0:x_0 + SX, :, :]
#
#    if sy > SY:
#        y_0 = max(0, (y_min + y_max) // 2 - SY // 2)
#        y_0 = min(y_0, sy - SY - 1)
#        vol = vol[:, y_0:y_0 + SY, :]
#        seg = seg[:, y_0:y_0 + SY, :]
#
#    if sz > SZ:
#        z_0 = max(0, (z_min + z_max) // 2 - SZ // 2)
#        z_0 = min(z_0, sz - SZ - 1)
#        vol = vol[:, :, z_0:z_0 + SZ]
#        seg = seg[:, :, z_0:z_0 + SZ]
#
#    vol_n = np.zeros((SX, SY, SZ), np.float32)
#    seg_n = np.zeros((SX, SY, SZ), np.int8)
#
#    vol_n[:sx, :sy, :sz] = vol.copy()
#    seg_n[:sx, :sy, :sz] = seg.copy()
#
#    return vol_n, seg_n


def crop_vol_n_seg(vol, seg, SX, SY, SZ):

    sx, sy, sz = vol.shape

    z = np.where(seg > 0)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()

    if sx > SX:
        x_0 = max(0, (x_min + x_max) // 2 - SX // 2)
        x_0 = min(x_0, sx - SX)
        vol = vol[x_0:x_0 + SX, :, :]
        seg = seg[x_0:x_0 + SX, :, :]

    if sy > SY:
        y_0 = max(0, (y_min + y_max) // 2 - SY // 2)
        y_0 = min(y_0, sy - SY)
        vol = vol[:, y_0:y_0 + SY, :]
        seg = seg[:, y_0:y_0 + SY, :]

    if sz > SZ:
        z_0 = max(0, (z_min + z_max) // 2 - SZ // 2)
        z_0 = min(z_0, sz - SZ)
        vol = vol[:, :, z_0:z_0 + SZ]
        seg = seg[:, :, z_0:z_0 + SZ]
        
    z = np.where(seg > 0)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
    
    vol_n, seg_n= vol, seg
    sx, sy, sz = vol.shape
    
    if sx < SX:
        
        vol_n = np.zeros((SX, sy, sz), np.float32)
        seg_n = np.zeros((SX, sy, sz), np.int8)
        
        x_0= (SX-sx)//2
        vol_n[x_0:x_0+sx, :, :] = vol.copy()
        seg_n[x_0:x_0+sx, :, :] = seg.copy()
        
        vol, seg= vol_n, seg_n
    
    if sy < SY:
        
        vol_n = np.zeros((SX, SY, sz), np.float32)
        seg_n = np.zeros((SX, SY, sz), np.int8)
        
        y_0= (SY-sy)//2
        vol_n[ :, y_0:y_0+sy, :] = vol.copy()
        seg_n[ :, y_0:y_0+sy, :] = seg.copy()
        
        vol, seg= vol_n, seg_n
    
    if sz < SZ:
        
        vol_n = np.zeros((SX, SY, SZ), np.float32)
        seg_n = np.zeros((SX, SY, SZ), np.int8)
        
        z_0= (SZ-sz)//2
        vol_n[ :, :, z_0:z_0+sz] = vol.copy()
        seg_n[ :, :, z_0:z_0+sz] = seg.copy()
        
        vol, seg= vol_n, seg_n

    return vol_n, seg_n








def crop_vol_n_seg_centered(vol, seg, SX, SY, SZ):

    sx, sy, sz = vol.shape

    z = np.where(seg > 0)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
    
    x_cntr= (x_min + x_max) // 2
    x_beg=  max(0,  x_cntr- SX//2)
    x_end=  min(sx, x_cntr+ SX//2)
    if x_beg>0:
        x_beg2= 0
    else:
        x_beg2= SX//2 -  x_cntr
        
    y_cntr= (y_min + y_max) // 2
    y_beg=  max(0,  y_cntr- SY//2)
    y_end=  min(sy, y_cntr+ SY//2)
    if y_beg>0:
        y_beg2= 0
    else:
        y_beg2= SY//2 -  y_cntr
        
    z_cntr= (z_min + z_max) // 2
    z_beg=  max(0,  z_cntr- SZ//2)
    z_end=  min(sz, z_cntr+ SZ//2)
    if z_beg>0:
        z_beg2= 0
    else:
        z_beg2= SZ//2 -  z_cntr
    
    vol_n = np.zeros((SX, SY, SZ), np.float32)
    seg_n = np.zeros((SX, SY, SZ), np.int8)
    
    vol_n[x_beg2:x_beg2+x_end-x_beg, y_beg2:y_beg2+y_end-y_beg, z_beg2:z_beg2+z_end-z_beg]= vol[x_beg:x_end, y_beg:y_end, z_beg:z_end].copy()
    seg_n[x_beg2:x_beg2+x_end-x_beg, y_beg2:y_beg2+y_end-y_beg, z_beg2:z_beg2+z_end-z_beg]= seg[x_beg:x_end, y_beg:y_end, z_beg:z_end].copy()
    
    print(x_beg, y_beg, z_beg, x_beg2, y_beg2, z_beg2)
    
    return vol_n, seg_n

















def augment_batch_nonrandom(batch_x, batch_y, APPLY_DEFORMATION, alpha,
                  APPLY_SHIFT, shift_x, shift_y, shift_z,
                  ADD_NOISE, noise_sigma,
                  ADD_FLIP, FLIP_x, FLIP_y, FLIP_z):
    
    batch_x_mean= np.mean(batch_x).astype(np.float64)
    
    if APPLY_DEFORMATION:
        
        x0 = batch_x[0, :, :, :, 0].copy()
        y = batch_y[0, :, :, :, :].copy()
        y = np.argmax(y, axis=-1)
        
        x0 = np.transpose(x0, [2, 1, 0])
        x0 = sitk.GetImageFromArray(x0)
        y = np.transpose(y, [2, 1, 0])
        y = sitk.GetImageFromArray(y)
        
        grid_physical_spacing = [25.0, 25.0, 25.0]
        image_physical_size = [size * spacing for size, spacing in zip(x0.GetSize(), x0.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5) \
                     for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        
        tx = sitk.BSplineTransformInitializer(x0, mesh_size)
        
        direction_size = (mesh_size[0] + 3) * (mesh_size[1] + 3) * (mesh_size[2] + 3) * 3
        
        direction = alpha * np.random.randn(direction_size)
        tx.SetParameters(direction)
        
        xx0 = sitk.Resample(x0, x0, tx, sitk.sitkBSpline, batch_x_mean, x0.GetPixelIDValue())
        yy = sitk.Resample(y, y, tx, sitk.sitkNearestNeighbor, 0.0, y.GetPixelIDValue())
        
        x0 = sitk.GetArrayFromImage(xx0)
        x0 = np.transpose(x0, [2, 1, 0])
        
        y = sitk.GetArrayFromImage(yy)
        y = np.transpose(y, [2, 1, 0])
        
        batch_x[0, :, :, :, 0] = x0.copy()
        
        batch_y = np.zeros(batch_y.shape)
        n_class= batch_y.shape[-1]
        for i_class in range(n_class):
            ind_matrix = (y == i_class)
            batch_y[0, ind_matrix, i_class] = 1
            
        x0 = y = xx = yy = 0
        
    if APPLY_SHIFT:
        
        x = batch_x[0, :, :, :, :].copy()
        y = batch_y[0, :, :, :, :].copy()
        
        sx, sy, sz, n_channel= x.shape
        _ , _ , _ , n_class=   y.shape
        
        #xx = np.zeros((sx + shift_x, sy + shift_y, sz + shift_z, n_channel), np.float32)
        xx = batch_x_mean* np.ones((sx + shift_x, sy + shift_y, sz + shift_z, n_channel), np.float32)
        xx[shift_x // 2:shift_x // 2 + sx, shift_y // 2:shift_y // 2 + sy, shift_z // 2:shift_z // 2 + sz,:] = x.copy()
        yy = np.zeros((sx + shift_x, sy + shift_y, sz + shift_z, n_class), np.float32)
        yy[:,:,:,0]= 1.0
        yy[shift_x // 2:shift_x // 2 + sx, shift_y // 2:shift_y // 2 + sy, shift_z // 2:shift_z // 2 + sz,:] = y.copy()
        
        batch_x[0, :, :, :, :] = xx[:sx, :sy, :sz,:].copy()
        batch_y[0, :, :, :, :] = yy[:sx, :sy, :sz,:].copy()
        
        x = y = xx = yy = 0
        
    if ADD_NOISE:
        
        batch_size, sx, sy, sz, n_channel= batch_x.shape
        
        batch_x += np.random.randn( batch_size, sx, sy, sz, n_channel ) * noise_sigma
    
    if ADD_FLIP:
        
        if FLIP_x:
            batch_x = batch_x[:, ::-1, :, :, :]
            batch_y = batch_y[:, ::-1, :, :, :]
        
        if FLIP_y:
            batch_x = batch_x[:, : , ::-1, :, :]
            batch_y = batch_y[:, : , ::-1, :, :]
        
        if FLIP_z:
            batch_x = batch_x[:, : , :, ::-1, :]
            batch_y = batch_y[:, : , :, ::-1, :]
    
    return batch_x, batch_y















#### ***************

def eul2quat(ax, ay, az, atol=1e-8):
    
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     
    
    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx
    
    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx
    
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv



def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    
    return [list(eul2quat(parameter_values[0],parameter_values[1], parameter_values[2])) + 
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]



def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                    output_prefix, output_suffix,
                    interpolator = sitk.sitkLinear, default_intensity_value = 0.0):
    
    all_images = [] # Used only for display purposes in this notebook.
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)        
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)
         
        all_images.append(aug_image)
    return all_images



def augment_batch_deterministic(batch_x, batch_y, theta_x, theta_y, theta_z,
                                shift_x, shift_y, shift_z,
                                scale):
    
    img= batch_x[0,:,:,:,0]
    seg= batch_y[0,:,:,:,1]
    
    img= sitk.GetImageFromArray(img)
    seg= sitk.GetImageFromArray(seg)
    
    aug_transform = sitk.Similarity3DTransform()
    
    reference_image = sitk.Image(img.GetSize(), img.GetPixelIDValue())
    
    reference_origin = np.zeros(3)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(img.GetSpacing())
    reference_image.SetDirection(img.GetDirection())
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    
    centering_transform = sitk.TranslationTransform(3)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    
    aug_transform.SetCenter(reference_center)
    
    transformation_parameters_list = similarity3D_parameter_space_regular_sampling(theta_x, theta_y, theta_z,shift_x, shift_y, shift_z, 1/scale)
    
    generated_images = augment_images_spatial(img, reference_image, centered_transform, 
                                       aug_transform, transformation_parameters_list, 
                                       None, None)
    
    img2= generated_images[0]
    img2np= sitk.GetArrayFromImage(img2)
    
    generated_images = augment_images_spatial(seg, reference_image, centered_transform, 
                                       aug_transform, transformation_parameters_list, 
                                       None, None,sitk.sitkNearestNeighbor)
    
    seg2= generated_images[0]
    seg2np= sitk.GetArrayFromImage(seg2)
    
    batch_xx= np.zeros(batch_x.shape)
    batch_yy= np.zeros(batch_y.shape)
    
    batch_xx[0,:,:,:,0]= img2np
    
    batch_yy[0,:,:,:,0]= 1
    batch_yy[0,:,:,:,0]= 1- seg2np
    batch_yy[0,:,:,:,1]= seg2np
    
    return batch_xx, batch_yy


#### ***************























def level_set_seg(x, y, vol_frac= 2, LS_iters= 400, max_rmse=0.05, curvature_scaling=1.0, propagation_scaling=4.0, vol_min= 5000):
    
    img = sitk.GetImageFromArray(x)
    
    seg_np0= y
    
    vol0= seg_np0.sum()
    rad= rad_best= 0
    
    while rad_best==0:
        rad+= 1
        seg= sitk.GetImageFromArray(seg_np0)
        seg = sitk.BinaryErode(seg, rad)
        seg_np= sitk.GetArrayFromImage(seg)
        voln= seg_np.sum()
        if voln==0:
            rad_best= rad-3
        elif voln<vol0/vol_frac or voln<vol_min:
            rad_best= rad
            
    seg= sitk.GetImageFromArray(seg_np0)
    if rad_best>0:
        seg= sitk.BinaryErode(seg, rad_best)
    
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(img, seg)
    
    factor = 2
    lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
    upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
    
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
    
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower_threshold)
    lsFilter.SetUpperThreshold(upper_threshold)
    lsFilter.SetMaximumRMSError(max_rmse)
    lsFilter.SetNumberOfIterations(LS_iters)
    lsFilter.SetCurvatureScaling(curvature_scaling)
    lsFilter.SetPropagationScaling(propagation_scaling)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(img, sitk.sitkFloat32))
    
    y_rough= sitk.GetArrayFromImage(ls)
    y_rough= y_rough>0
    y_rough= y_rough.astype(np.int8)
    
    return y_rough
    




def level_set_seg_multiparameter(x, y, max_rmse_v, curvature_scaling_v, propagation_scaling_v, vol_frac= 2, LS_iters= 400, vol_min= 5000):
    
    SX, SY, SZ= x.shape
    
    n_level_set= len(max_rmse_v)*len(curvature_scaling_v)*len(propagation_scaling_v)
    
    y_level_set = np.zeros( (SX, SY, SZ, n_level_set) )
    i_level_set= -1
    
    img = sitk.GetImageFromArray(x)
    
    seg_np0= y
    
    vol0= seg_np0.sum()
    rad= rad_best= 0
    
    while rad_best==0:
        rad+= 1
        seg= sitk.GetImageFromArray(seg_np0)
        seg = sitk.BinaryErode(seg, rad)
        seg_np= sitk.GetArrayFromImage(seg)
        voln= seg_np.sum()
        if voln==0:
            rad_best= rad-3
        elif voln<vol0/vol_frac or voln<vol_min:
            rad_best= rad
            
    seg= sitk.GetImageFromArray(seg_np0)
    if rad_best>0:
        seg= sitk.BinaryErode(seg, rad_best)
    
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(img, seg)
    
    factor = 2
    lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
    upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
    
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
    
    for max_rmse in max_rmse_v:
        for curvature_scaling in curvature_scaling_v:
            for propagation_scaling in propagation_scaling_v:
                
                lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
                lsFilter.SetLowerThreshold(lower_threshold)
                lsFilter.SetUpperThreshold(upper_threshold)
                lsFilter.SetMaximumRMSError(max_rmse)
                lsFilter.SetNumberOfIterations(LS_iters)
                lsFilter.SetCurvatureScaling(curvature_scaling)
                lsFilter.SetPropagationScaling(propagation_scaling)
                lsFilter.ReverseExpansionDirectionOn()
                ls = lsFilter.Execute(init_ls, sitk.Cast(img, sitk.sitkFloat32))
                
                y_rough= sitk.GetArrayFromImage(ls)
                y_rough= y_rough>0
                y_rough= y_rough.astype(np.int8)
                
                i_level_set+= 1
                y_level_set[ :,:,:,i_level_set]= y_rough
                
    return y_level_set
    










def fill_multi_class_seg_holes(y_tr_pr_c):
    
    seg_with_bad_background=y_tr_pr_c

    seg_bad= seg_with_bad_background.copy()
    
    seg_good= seg_bad.copy()
    
    n_con, s_con, seg_con= number_n_size_of_components( (seg_bad==0).astype(np.int) , return_seg=True )
    
    good_points= np.where(seg_bad>0)
    n_good_points= len( good_points[0] )
    P_good= np.zeros((n_good_points,3))
    P_good[:,0]= good_points[0]
    P_good[:,1]= good_points[1]
    P_good[:,2]= good_points[2]
    
    for i_con in range(n_con):
        
        if not s_con[i_con]==s_con.max():
            
            bad_points= np.where(seg_con==i_con+1)
            n_bad_points= len( bad_points[0] )
            P_bad= np.zeros((n_bad_points,3))
            P_bad[:,0]= bad_points[0]
            P_bad[:,1]= bad_points[1]
            P_bad[:,2]= bad_points[2]
            
            assert( n_bad_points>0 )
            
            temp= cdist(P_bad, P_good)
            temp= np.argmin(temp, axis=1)
            
            P_bad= P_bad.astype(np.int)
            P_good= P_good.astype(np.int)
            
            for i_bad in range(n_bad_points):
                
                seg_good[ P_bad[i_bad,0] , P_bad[i_bad,1] , P_bad[i_bad,2] ]=\
                    seg_good[ P_good[temp[i_bad],0] , P_good[temp[i_bad],1] , P_good[temp[i_bad],2] ]
           
    return seg_good




















def create_label_smoothing_kernels(smooth_R=4, smooth_r=[1,2,3], smoothness=1):
    
    n_r= len(smooth_r)
    
    W= np.zeros((smooth_R*2+1,smooth_R*2+1,smooth_R*2+1,n_r))
    
    for i_r in range(n_r):
        
        sigx= np.sqrt( -smooth_r[i_r]**2/ np.log(0.5) ) * smoothness
        
        wx = np.exp(- np.abs(np.linspace(- smooth_R, smooth_R , smooth_R*2+1)) ** 2 / sigx ** 2)
        
        wxy= np.matmul( wx[:,np.newaxis], wx[np.newaxis,:] )
        
        W[:,:,:,i_r] = np.matmul(wxy[:,:, np.newaxis], wx[np.newaxis, :])
        
        W[:,:,:,i_r]/= W[:,:,:,i_r].sum()
    
    return W






# def smooth_labels(y_hard_compressed, y_unc, W):
    
#     sx, sy, sz= y_hard_compressed.shape
#     n_class= y_hard_compressed.max()+1
    
#     w0= 2
#     w= W.shape[0]//2
    
#     y_smooth= np.zeros( (sx, sy, sz, n_class) )
    
#     for ix in range(w,sx-w-1):
#         print(ix)
#         for iy in range(w,sy-w-1):
#             for iz in range(w,sz-w-1):
                
#                 unc_block= y_unc[ix-w0:ix+w0+1,iy-w0:iy+w0+1,iz-w0:iz+w0+1]
#                 seg_block= y_hard_compressed[ix-w:ix+w+1,iy-w:iy+w+1,iz-w:iz+w+1]
                
#                 if unc_block.min()>0:
                    
#                     if seg_block.std()>0:
                        
#                         ker= W[:,:,:,int(unc_block.min())-1]
                        
#                         for class_ind in range(n_class):
#                             temp_block= seg_block==class_ind
#                             prob= np.sum( temp_block * ker  )
#                             y_smooth[ix, iy, iz, class_ind]= prob
#                             # print( class_ind, prob )
                
#                 else:
                    
#                     class_ind= y_hard_compressed[ix, iy, iz]
#                     y_smooth[ix, iy, iz, class_ind]= 1
    
#     return y_smooth
    


def smooth_labels(y_hard_compressed, y_unc, W):
    
    sx, sy, sz= y_hard_compressed.shape
    n_class= y_hard_compressed.max()+1
    
    w0= 2
    w= W.shape[0]//2
    
    y_smooth= np.zeros( (sx, sy, sz, n_class) )
    
    for ix in range(w,sx-w-1):
        # print(ix)
        for iy in range(w,sy-w-1):
            for iz in range(w,sz-w-1):
                
                unc_block= y_unc[ix-w0:ix+w0+1,iy-w0:iy+w0+1,iz-w0:iz+w0+1]
                seg_block= y_hard_compressed[ix-w:ix+w+1,iy-w:iy+w+1,iz-w:iz+w+1]
                
                if unc_block.min()>0 and seg_block.std()>0:
                    
                    ker= W[:,:,:,int(unc_block.min())-1]
                    
                    classes_temp= np.unique(seg_block)
                    
                    for class_ind in classes_temp:
                        # temp_block= seg_block==class_ind
                        # prob= np.sum( seg_block==class_ind * ker  )
                        y_smooth[ix, iy, iz, class_ind]= np.sum( (seg_block==class_ind) * ker  )
                        # print( class_ind, prob )
                    
                else:
                    
                    class_ind= y_hard_compressed[ix, iy, iz]
                    y_smooth[ix, iy, iz, class_ind]= 1
    
    y_smooth_sum= np.sum(y_smooth, axis=-1)
    
    y_smooth[y_smooth_sum==0,0]= 1
    
    y_smooth_sum= np.sum(y_smooth, axis=-1)
    
    y_smooth= y_smooth/ np.tile( y_smooth_sum[:,:,:,np.newaxis], [1, 1, 1, n_class])
    
    return y_smooth













def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.
    
    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """
    
    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()
        
    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])
        
    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]
    
    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])
        
    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)






def scaling_and_squaring(vec, nb_steps):
    
    
    vec = vec / (2**nb_steps)
    for _ in range(nb_steps):
        vec += transform(vec, vec)
    disp = vec
    
    
    return disp







def NCC_loss(moved, fixed, w=9, eps=1e-5, signed=False, reduce='mean'):
    
    win = [w] * 3
    
    conv_fn = getattr(tf.nn, 'conv%dd' % 3)
    
    I2 = moved * moved
    J2 = fixed * fixed
    IJ = moved * fixed
    
    in_ch = fixed.get_shape().as_list()[-1]
    sum_filt = tf.ones([w,w, w, in_ch, 1])
    strides = [1] * (3 + 2)

    padding = 'SAME'
    I_sum =  conv_fn(moved, sum_filt, strides, padding)
    J_sum =  conv_fn(fixed, sum_filt, strides, padding)
    I2_sum = conv_fn(I2,    sum_filt, strides, padding)
    J2_sum = conv_fn(J2,    sum_filt, strides, padding)
    IJ_sum = conv_fn(IJ,    sum_filt, strides, padding)
    
    win_size = np.prod(win) * in_ch
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    cross = tf.maximum(cross, eps)
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    I_var = tf.maximum(I_var, eps)
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    J_var = tf.maximum(J_var, eps)
    
    if signed:
        cc = cross / tf.sqrt(I_var * J_var + eps)
    else:
        cc = (cross / I_var) * (cross / J_var)
    
    if reduce == 'mean':
        cc_loss = - tf.reduce_mean( cc)
    elif reduce == 'max':
        cc_loss = - tf.reduce_max( cc)
    elif reduce is not None:
        raise ValueError(f'Unknown NCC reduction type: {reduce}')
    
    return cc_loss, cc



















def elastic_ferom(x0, delta_x, alpha, interpolator= sitk.sitkBSpline):
    
    x0 = np.transpose(x0, [2, 1, 0])
    x0 = sitk.GetImageFromArray(x0)
    
    grid_physical_spacing = [delta_x, delta_x, delta_x]
    image_physical_size = [size * spacing for size, spacing in zip(x0.GetSize(), x0.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    
    tx = sitk.BSplineTransformInitializer(x0, mesh_size)
    
    direction_size = (mesh_size[0] + 3) * (mesh_size[1] + 3) * (mesh_size[2] + 3) * 3
    
    direction = alpha * np.random.randn(direction_size)
    tx.SetParameters(direction)
    
    xx0 = sitk.Resample(x0, x0, tx, interpolator, 0, x0.GetPixelIDValue())
    
    x0 = sitk.GetArrayFromImage(xx0)
    x0 = np.transpose(x0, [2, 1, 0])
        
    return x0






def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.
    
    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')





def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.

    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.

    Returns:
        Dense shift (warp) of shape (*shape, N).
    """
    
    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)
    
    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [f if f.dtype == matrix.dtype else tf.cast(f, matrix.dtype) for f in mesh]
    
    if shift_center:
        mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]
        
    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype=matrix.dtype))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels
    
    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N
    
    # get shifts and return
    return loc - tf.stack(mesh, axis=ndims)
















def crop_DMRI_n_TEN_centered(dmri, ten, LX, LY, LZ):
    
    n_dmri = dmri.shape[-1]
    n_ten  = ten.shape[-1]
    
    z = np.where(dmri[:,:,:,0] > 0)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
    
    dmri= dmri[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1, :]
    ten = ten[ x_min:x_max+1, y_min:y_max+1, z_min:z_max+1, :]
    
    assert(dmri.shape[0]<=LX)
    assert(dmri.shape[1]<=LY)
    assert(dmri.shape[2]<=LZ)
    
    x_beg= ( LX - (x_max - x_min + 1 ) )//2
    y_beg= ( LY - (y_max - y_min + 1 ) )//2
    z_beg= ( LZ - (z_max - z_min + 1 ) )//2
    
    dmri_n = np.zeros((LX, LY, LZ, n_dmri), np.float32)
    ten_n  = np.zeros((LX, LY, LZ, n_ten),  np.float32)
    
    dmri_n[x_beg:x_beg+dmri.shape[0], y_beg:y_beg+dmri.shape[1], z_beg:z_beg+dmri.shape[2],:]= dmri.copy()
    ten_n [x_beg:x_beg+dmri.shape[0], y_beg:y_beg+dmri.shape[1], z_beg:z_beg+dmri.shape[2],:]= ten.copy()
    
    return dmri_n, ten_n


































