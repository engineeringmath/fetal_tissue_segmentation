#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:41:55 2021

@author: davood
"""



from __future__ import division

import numpy as np
import os
import tensorflow as tf
# import tensorlayer as tl
from os import listdir
from os.path import isfile, join
#import scipy.io as sio
#from skimage import io
#import skimage
import SimpleITK as sitk
# import matplotlib.pyplot as plt
import h5py
#import scipy
#from scipy.spatial import ConvexHull
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.spatial import Delaunay
import os.path
#import pandas as pd
# import sys
import pickle
#from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import seg_util
import models
import aux
# from PIL import Image
import pandas as pd
# from scipy.stats import beta
from medpy.metric import hd95
# from medpy.metric import asd
from medpy.metric import assd
# from scipy import ndimage
# from scipy.stats import beta
from shutil import copyfile
# import shutil
# from scipy import stats
import nibabel as nib
# from scipy.spatial.distance import cdist
# from skimage.morphology import skeletonize, skeletonize_3d
# import seaborn as sns








##############################################################################

##   Read images and segmentations

##############################################################################

#   Data directory

base_dir= ' ... '
data_dir= base_dir + 'data/tissue/'
thmb_dir= base_dir + 'data/thmb/tissue/'



#  image size and resolution

SX, SY, SZ= 128, 152, 132
desired_spacing= (0.8, 0.8, 0.8)
data_file_name= 'fetal_tissue_' + str(SX) + '_' + str(SY) + '_' + str(SZ)  +'.h5'


# read subjects ages in weeks

ages_df= pd.read_csv( base_dir + 'data/GAs.csv' , delimiter= ',')


# read label info

labels_df= pd.read_csv( base_dir + 'data/tissue_labels.csv' , delimiter= ',')

label_names= labels_df.loc[:,"7"].to_numpy()

labels= labels_df.loc[:,"0"].to_numpy()




#  read the image and label files

# n_class= len(labels)

# n_all= 312
# n_channel= 1

# image_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and not 'tissue' in f]
# image_files.sort()

# X_all= np.zeros( (n_all, SX, SY, SZ, n_channel), np.float32)
# Y_all= np.zeros( (n_all, SX, SY, SZ, n_class), np.uint8 )
# Y_all[:,:,:,:,0]= 1
# label_counts= np.zeros( (n_all, n_class) )
# info_all= np.zeros( (n_all, 30) )
# age_all= np.zeros( n_all )
# nam_all= list()

# for i_all in range(n_all):
    
#     img_file_name= image_files[i_all]
#     les_file_name= 'tissue-' + img_file_name
#     img= sitk.ReadImage( data_dir + img_file_name )
#     les= sitk.ReadImage( data_dir + les_file_name )
    
#     nam_all.append( img_file_name.split('.nii')[0] )
    
#     age= img_file_name.split('.nii')[0]
#     age= ages_df.loc[ (ages_df['ID']==age ) ]
#     age= age.iloc[0]['Age']
#     age_all[i_all]= age
    
#     assert( img.GetSpacing()==les.GetSpacing() )
#     assert( np.allclose( img.GetSpacing(), desired_spacing ) )
#     assert( img.GetSize()==les.GetSize() )
    
#     img_np = sitk.GetArrayFromImage(img)
#     img_np = np.transpose(img_np, [2, 1, 0]).astype(np.float32)
#     les_np = sitk.GetArrayFromImage(les)
#     les_np = np.transpose(les_np, [2, 1, 0]).astype(np.float32)
    
#     temp= img_np[img_np>10]
#     info_all[i_all,0]= len(temp)
#     info_all[i_all,1]= temp.mean()
#     info_all[i_all,2]= temp.std()
    
#     img_np/= temp.std()
    
#     z= np.where(les_np>0)
#     mk_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
    
#     sl_x= (z[0].min() + z[0].max())//2
#     sl_y= (z[1].min() + z[1].max())//2
#     sl_z= (z[2].min() + z[2].max())//2
    
#     n_rows, n_cols = 2, 3
#     fig, ax = plt.subplots(figsize=(20, 10), nrows=n_rows, ncols=n_cols)
#     plt.subplot(n_rows, n_cols, 1), plt.imshow(img_np[:, :, sl_z], cmap='gray')
#     plt.subplot(n_rows, n_cols, 4), plt.imshow(les_np[:, :, sl_z])
#     plt.subplot(n_rows, n_cols, 2), plt.imshow(img_np[:, sl_y, :], cmap='gray')
#     plt.subplot(n_rows, n_cols, 5), plt.imshow(les_np[:, sl_y, :])
#     plt.subplot(n_rows, n_cols,3), plt.imshow(img_np[sl_x, :, :], cmap='gray')
#     plt.subplot(n_rows, n_cols,6), plt.imshow(les_np[sl_x, :, :])
#     plt.tight_layout()
#     fig.savefig(thmb_dir + 'x_' + str(i_all) + '.png')
#     plt.close(fig)
    
#     info_all[i_all, 3:9]= mk_extent
    
#     for i_label in range(n_class):
#         label_counts[i_all,i_label]= np.sum(les_np==labels[i_label])
    
#     SX_t, SY_t, SZ_t= les_np.shape
#     les_np_new= np.zeros( (SX_t, SY_t, SZ_t, n_class), np.uint8 )
#     for i_class in range(n_class):
#         mask= les_np==labels[i_class]
#         les_np_new[mask==1,i_class]= 1
    
#     x_beg, x_end, y_beg, y_end, z_beg, z_end= mk_extent
#     img_crop=     img_np[x_beg:x_end+1, y_beg:y_end+1, z_beg:z_end+1].copy()
#     les_crop= les_np_new[x_beg:x_end+1, y_beg:y_end+1, z_beg:z_end+1].copy()
#     x0= (SX-(x_end-x_beg+1))//2
#     y0= (SY-(y_end-y_beg+1))//2
#     z0= (SZ-(z_end-z_beg+1))//2
#     X_all[i_all, x0:x0+(x_end-x_beg)+1, y0:y0+(y_end-y_beg)+1, z0:z0+(z_end-z_beg)+1, 0]= img_crop
#     Y_all[i_all, x0:x0+(x_end-x_beg)+1, y0:y0+(y_end-y_beg)+1, z0:z0+(z_end-z_beg)+1, :]= les_crop


# young_index= np.where(label_counts[:,23])[0]
# assert( np.allclose( np.where(label_counts[:,24])[0] , np.where(label_counts[:,23])[0] ) )
# assert( np.allclose( np.where(label_counts[:,25])[0] , np.where(label_counts[:,23])[0] ) )
# assert( np.allclose( np.where(label_counts[:,26])[0] , np.where(label_counts[:,23])[0] ) )

# old_index= np.where(label_counts[:,29])[0]
# assert( np.allclose( np.where(label_counts[:,29])[0] , np.where(label_counts[:,30])[0] ) )

# X_young= X_all[young_index]
# Y_young= Y_all[young_index]
# nam_young= np.array(nam_all)[young_index]
# info_young= info_all[young_index]
# label_counts_young= label_counts[young_index]
# age_young= age_all[young_index]
# label_counts_young= np.delete(label_counts_young, 30, -1)
# label_counts_young= np.delete(label_counts_young, 29, -1)
# Y_young= np.delete(Y_young, 30, -1)
# Y_young= np.delete(Y_young, 29, -1)
# names_young= np.array(image_files)[young_index.astype(int)]
# labels_young= labels.copy()
# labels_young= np.delete(labels_young, 30)
# labels_young= np.delete(labels_young, 29)
# label_names_young= label_names.copy()
# label_names_young= np.delete(label_names_young, 30)
# label_names_young= np.delete(label_names_young, 29)

# X_old= X_all[old_index]
# Y_old= Y_all[old_index]
# nam_old= np.array(nam_all)[old_index]
# info_old= info_all[old_index]
# label_counts_old= label_counts[old_index]
# age_old= age_all[old_index]
# label_counts_old= np.delete(label_counts_old, 26, -1)
# label_counts_old= np.delete(label_counts_old, 25, -1)
# label_counts_old= np.delete(label_counts_old, 24, -1)
# label_counts_old= np.delete(label_counts_old, 23, -1)
# Y_old= np.delete(Y_old, 26, -1)
# Y_old= np.delete(Y_old, 25, -1)
# Y_old= np.delete(Y_old, 24, -1)
# Y_old= np.delete(Y_old, 23, -1)
# names_old= np.array(image_files)[old_index.astype(int)]
# labels_old= labels.copy()
# labels_old= np.delete(labels_old, 26)
# labels_old= np.delete(labels_old, 25)
# labels_old= np.delete(labels_old, 24)
# labels_old= np.delete(labels_old, 23)
# label_names_old= label_names.copy()
# label_names_old= np.delete(label_names_old, 26)
# label_names_old= np.delete(label_names_old, 25)
# label_names_old= np.delete(label_names_old, 24)
# label_names_old= np.delete(label_names_old, 23)

# del X_all, Y_all

# h5f = h5py.File(base_dir + 'data/' + data_file_name,'w')
# h5f['X_young']= X_young
# h5f['Y_young']= Y_young
# h5f['info_young']= info_young
# h5f['age_young']= age_young
# h5f['label_counts_young']= label_counts_young
# # h5f['names_young']= names_young
# # h5f['labels_young']= labels_young
# # h5f['label_names_young']= label_names_young
# h5f['X_old']= X_old
# h5f['Y_old']= Y_old
# h5f['info_old']= info_old
# h5f['age_old']= age_old
# h5f['label_counts_old']= label_counts_old
# # h5f['names_old']= names_old
# # h5f['labels_old']= labels_old
# # h5f['label_names_old']= label_names_old
# h5f.close()

# with open(base_dir + 'data/fetal_tissue_names', 'wb') as fp:
#         pickle.dump(names_young, fp)
#         pickle.dump(names_old, fp)

# with open(base_dir + 'data/fetal_tissue_labels_and_names', 'wb') as fp:
#         pickle.dump(labels_young, fp)
#         pickle.dump(label_names_young, fp)
#         pickle.dump(labels_old, fp)
#         pickle.dump(label_names_old, fp)


################################



##############################################################################

##   Select age groups and partician into test/train

##############################################################################


young_or_old= 'young'

train_on_smooth= True

if young_or_old=='young':
    
    h5f = h5py.File(base_dir + 'data/' + data_file_name,'r')
    X_all= h5f['X_young'][:]
    info_all= h5f['info_young'][:]
    age_all= h5f['age_young'][:]
    label_counts_young= h5f['label_counts_young'][:]
    h5f.close()
    if train_on_smooth:
        h5f = h5py.File(base_dir + 'data/young_train_smoothed.h5','r')
        Y_all= h5f['Y_train_smooth'][:]
        h5f.close()
    else:
        h5f = h5py.File(base_dir + 'data/' + data_file_name,'r')
        Y_all= h5f['Y_young'][:]
        h5f.close()
    
else:
    
    h5f = h5py.File(base_dir + 'data/' + data_file_name,'r')
    X_all= h5f['X_old'][:]
    info_all= h5f['info_old'][:]
    age_all= h5f['age_old'][:]
    label_counts_old= h5f['label_counts_old'][:]
    h5f.close()
    if train_on_smooth:
        h5f = h5py.File(base_dir + 'data/old_train_smoothed.h5','r')
        Y_all= h5f['Y_train_smooth'][:]
        h5f.close()
    else:
        h5f = h5py.File(base_dir + 'data/' + data_file_name,'r')
        Y_all= h5f['Y_old'][:]
        h5f.close()

with open (base_dir + 'data/fetal_tissue_names', 'rb') as fp:
    names_young = pickle.load(fp)
    names_old = pickle.load(fp)

with open (base_dir + 'data/fetal_tissue_labels_and_names', 'rb') as fp:
    labels_young = pickle.load(fp)
    label_names_young = pickle.load(fp)
    labels_old = pickle.load(fp)
    label_names_old = pickle.load(fp)



if young_or_old=='young':
    names_all= names_young
    test_subject_names= pd.read_csv( base_dir + 'data/test_subject_names_young.csv' , delimiter= ',')
    test_subject_names= test_subject_names.loc[:,"name"].to_numpy()
else:
    names_all= names_old
    test_subject_names= pd.read_csv( base_dir + 'data/test_subject_names_old.csv' , delimiter= ',')
    test_subject_names= test_subject_names.loc[:,"name"].to_numpy()



n_all= X_all.shape[0]
n_channel= X_all.shape[-1]
n_class=   Y_all.shape[-1]


# p_test=  np.array([ 0, 1, 7, 15, 18, 73, 77, 25, 26, 27, 30, 36, 40, 72, 75 ])
p_test= list()
for i in range(len(test_subject_names)):
    new_ind= np.where(names_all==test_subject_names[i])[0]
    assert(new_ind.shape[0]>0)
    p_test.append(new_ind)
p_test= np.array(p_test)
p_test= np.squeeze(p_test)
p_test= np.sort(p_test)

p_train= list()
for i in range(n_all):
    if i in p_test:
        pass
    else:
        p_train.append(i)
p_train= np.array(p_train)

X_test=  X_all[p_test]
X_train= X_all[p_train]
Y_test=  Y_all[p_test]
Y_train= Y_all[p_train]

del X_all, Y_all

M_train= np.argmax(Y_train, axis=-1)
M_train= (M_train<1).astype(np.int)

print(p_train)
print(p_test)




n_train = X_train.shape[0]
n_test =  X_test.shape[0]




##   T matrices

# for i in range(T_old.shape[0]):
#     if T_old[i,:].sum()>0:
#         T_old[i,:] /= T_old[i,:].sum()
#     else:
#         print(i)
#         T_old[i,i]= 1

# for i in range(T2_old.shape[0]):
#     if T2_old[i,:].sum()>0:
#         T2_old[i,:] /= T2_old[i,:].sum()
#     else:
#         print(i)
#         T2_old[i,i]= 1

# for i in range(T_young.shape[0]):
#     if T_young[i,:].sum()>0:
#         T_young[i,:] /= T_young[i,:].sum()
#     else:
#         print(i)
#         T_young[i,i]= 1

# for i in range(T2_young.shape[0]):
#     if T2_young[i,:].sum()>0:
#         T2_young[i,:] /= T2_young[i,:].sum()
#     else:
#         print(i)
#         T2_young[i,i]= 1

# # plt.figure(), plt.imshow( np.log(T_old+0.001) )
# plt.figure(), plt.imshow( np.log(T2_old+0.001) )
# plt.axis('off')
# plt.colorbar()
# # plt.figure(), plt.imshow( np.log(T_young+0.001) )
# plt.figure(), plt.imshow( np.log(T2_young+0.001) )
# plt.axis('off')
# plt.colorbar()

if young_or_old=='old':
    h5f = h5py.File(base_dir + 'data/old_train_Ts.h5','r')
    T_matrix= h5f['T'][:]
    # T_matrix= h5f['T2'][:]
    h5f.close()
else:
    h5f = h5py.File(base_dir + 'data/young_train_Ts.h5','r')
    T_matrix= h5f['T'][:]
    # T_matrix= h5f['T2'][:]
    h5f.close()






################################

#  Training label smoothing

'''
if young_or_old=='young':
    uncertainty_df= pd.read_csv( base_dir + 'data/uncertainty_young.csv' , delimiter= ',')
else:
    uncertainty_df= pd.read_csv( base_dir + 'data/uncertainty_old.csv' , delimiter= ',')

uncertainty= uncertainty_df.loc[:,"uncertainty"].to_numpy()
# erosion_rad= uncertainty_df.loc[:,"erode"].to_numpy()

# i_train= 0

# x=      X_train[i_train,:,:,:,0].copy()
# y_hard= Y_train[i_train,:,:,:,:].copy()

# y_hard_compressed= np.argmax(y_hard, axis=-1)

# vol_hard= np.zeros(y_hard.shape[-1]-1)
# for i in range(y_hard.shape[-1]-1):
#     vol_hard[i]= np.sum(y_hard_compressed==i+1)

# y_unc= np.zeros(y_hard_compressed.shape)
# for i in range(1, y_hard.shape[-1]):
#     mask_temp= y_hard_compressed==i
#     y_unc[mask_temp]= uncertainty[i-1]

W= seg_util.create_label_smoothing_kernels(smoothness=1.0)

# y_smooth= seg_util.smooth_labels(y_hard_compressed, y_unc, W)

# y_smooth_compressed= np.argmax(y_smooth, axis=-1)


Y_train_smooth= np.zeros(Y_train.shape, np.float16)

for i_train in range(n_train):
    
    print(i_train)
    
    y_hard= Y_train[i_train,:,:,:,:].copy()
    
    y_hard_compressed= np.argmax(y_hard, axis=-1)
    
    y_unc= np.zeros(y_hard_compressed.shape)
    for i in range(1, y_hard.shape[-1]):
        mask_temp= y_hard_compressed==i
        y_unc[mask_temp]= uncertainty[i-1]
    
    y_smooth= seg_util.smooth_labels(y_hard_compressed, y_unc, W)
    
    Y_train_smooth[i_train,:,:,:,:]= y_smooth


h5f = h5py.File(base_dir + 'data/old_train_smoothed_2.h5','w')
h5f['Y_train_smooth']= Y_train_smooth
h5f.close()
# h5f = h5py.File(base_dir + 'data/old_train_smoothed.h5','r')
# Y_train_smooth= h5f['Y_train_smooth'][:]
# h5f.close()



T= np.zeros((n_class,n_class))

for i_train in range(n_train):
    print(i_train)
    y_hard= Y_train[i_train,:,:,:,:].copy()
    y_hard_compressed= np.argmax(y_hard, axis=-1)
    y_smooth= Y_train_smooth[i_train,:,:,:,:].copy()
    
    for i_class in range(n_class):
        temp= y_hard_compressed==i_class
        for j_class in range(n_class):
            temp2= y_smooth[temp==1,j_class]
            T[i_class,j_class]+= np.float32(temp2).sum()

T2= np.zeros((n_class,n_class))

for i_train in range(n_train):
    print(i_train)
    y_hard= Y_train[i_train,:,:,:,:].copy()
    y_hard_compressed= np.argmax(y_hard, axis=-1)
    y_smooth= Y_train_smooth[i_train,:,:,:,:].copy()
    y_smooth_unambiguous= np.max(y_smooth, axis=-1)
    y_smooth_unambiguous= y_smooth_unambiguous<0.99
    
    for i_class in range(n_class):
        temp= y_hard_compressed==i_class
        temp= np.logical_and(temp==1, y_smooth_unambiguous==1)
        for j_class in range(n_class):
            temp2= y_smooth[temp==1,j_class]
            T2[i_class,j_class]+= np.float32(temp2).sum()

# h5f = h5py.File(base_dir + 'data/young_train_Ts.h5','w')
# h5f['T']= T
# h5f['T2']= T2
# h5f.close()

h5f = h5py.File(base_dir + 'data/old_train_Ts.h5','r')
T= h5f['T'][:]
T2= h5f['T2'][:]
h5f.close()
'''


'''

h5f = h5py.File(base_dir + 'data/old_train_Ts.h5','r')
T_old= h5f['T'][:]
T2_old= h5f['T2'][:]
h5f.close()

h5f = h5py.File(base_dir + 'data/young_train_Ts.h5','r')
T_young= h5f['T'][:]
T2_young= h5f['T2'][:]
h5f.close()

for i in range(T_old.shape[0]):
    if T_old[i,:].sum()>0:
        T_old[i,:] /= T_old[i,:].sum()
    else:
        print(i)
        T_old[i,i]= 1

for i in range(T2_old.shape[0]):
    if T2_old[i,:].sum()>0:
        T2_old[i,:] /= T2_old[i,:].sum()
    else:
        print(i)
        T2_old[i,i]= 1

for i in range(T_young.shape[0]):
    if T_young[i,:].sum()>0:
        T_young[i,:] /= T_young[i,:].sum()
    else:
        print(i)
        T_young[i,i]= 1

for i in range(T2_young.shape[0]):
    if T2_young[i,:].sum()>0:
        T2_young[i,:] /= T2_young[i,:].sum()
    else:
        print(i)
        T2_young[i,i]= 1

# for i in range(T_old.shape[1]):
#     if T_old[:,i].sum()>0:
#         T_old[:,i] /= T_old[:,i].sum()
#     else:
#         print(i)
#         T_old[i,i]= 1

# for i in range(T2_old.shape[1]):
#     if T2_old[:,i].sum()>0:
#         T2_old[:,i] /= T2_old[:,i].sum()
#     else:
#         print(i)
#         T2_old[i,i]= 1

# for i in range(T_young.shape[1]):
#     if T_young[:,i].sum()>0:
#         T_young[:,i] /= T_young[:,i].sum()
#     else:
#         print(i)
#         T_young[i,i]= 1

# for i in range(T2_young.shape[1]):
#     if T2_young[:,i].sum()>0:
#         T2_young[:,i] /= T2_young[:,i].sum()
#     else:
#         print(i)
#         T2_young[i,i]= 1

plt.figure(), plt.imshow( np.log(T_old+0.001) )
plt.figure(), plt.imshow( np.log(T2_old+0.00001) )
plt.figure(), plt.imshow( np.log(T_young+0.001) )
plt.figure(), plt.imshow( np.log(T2_young+0.001) )

T_old_inv= np.linalg.inv(T_old)
T2_old_inv= np.linalg.inv(T2_old)
T_young_inv= np.linalg.inv(T_young)
T2_young_inv= np.linalg.inv(T2_young)

plt.figure(), plt.imshow( np.log(T_old_inv+1.001) )
plt.figure(), plt.imshow( np.log(T2_old_inv+1.001) )
plt.figure(), plt.imshow( np.log(T_young_inv+1.001) )
plt.figure(), plt.imshow( np.log(T2_young_inv+1.001) )

'''





##############################################################################

##   Training

##############################################################################


#  Choose training settings

gpu_ind= 3
L_Rate = 1.0e-5
train_dir = base_dir + 'train/tissue_young/train_smooth_3/'
os.makedirs(train_dir)



model_dir = train_dir + 'model/'
thumbs_dir = train_dir + 'thumbs/'
os.makedirs(model_dir)
os.makedirs(thumbs_dir)

copyfile('/home/davood/Documents/codes/fetal_tissue.py', train_dir + 'fetal_tissue.py'  )



# LX= LY= LZ = 96
LX, LY, LZ = 96, 96, 64

test_shift= LX//4
# if SX-LX>0:
#     lx_list= np.squeeze( np.concatenate((np.arange(0,SX-LX,test_shift)[:,np.newaxis],np.array([SX-LX])[:,np.newaxis])).astype(np.int) )
# else:
#     lx_list= np.zeros(1, dtype=np.int)
# if SY-LY>0:
#     ly_list= np.squeeze( np.concatenate((np.arange(0,SY-LY,test_shift)[:,np.newaxis],np.array([SY-LY])[:,np.newaxis])).astype(np.int) )
# else:
#     ly_list= np.zeros(1, dtype=np.int)
# if SZ-LZ>0:
#     lz_list= np.squeeze( np.concatenate((np.arange(0,SZ-LZ,test_shift)[:,np.newaxis],np.array([SZ-LZ])[:,np.newaxis])).astype(np.int) )
# else:
#     lz_list= np.zeros(1, dtype=np.int)
lx_list= np.squeeze( np.concatenate( (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] ) ) .astype(np.int) )
ly_list= np.squeeze( np.concatenate( (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] ) ) .astype(np.int) )
lz_list= np.squeeze( np.concatenate( (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] ) ) .astype(np.int) )
LXc= (LX-10)//2
LYc= (LY-10)//2
LZc= (LZ-10)//2

n_feat_0 = 20
depth = 5
ks_0 = 3

X = tf.placeholder("float32", [None, LX, LY, LZ, n_channel])
Y = tf.placeholder("float32", [None, LX, LY, LZ, n_class])
M = tf.placeholder("float32", [None, LX, LY, LZ])
T = tf.placeholder("float32", [n_class, n_class])
learning_rate = tf.placeholder("float")
p_keep_conv = tf.placeholder("float")

logit_f, _ = models.my_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001)

predicter = tf.nn.softmax(logit_f)

# cost= models.cost_dice_multi_forground(Y, predicter, n_foreground=n_class-1, loss_type='sorensen', smooth=1e-5)
# cost= models.cost_dice_multi_forground_log(Y, predicter, n_foreground=n_class-1, loss_type='sorensen', smooth=1e-3, log_smooth=1e-5)
# cost= models.cost_x_entropy(Y, logit_f)
cost= models.cost_x_entropy_noisy(Y, logit_f, M, T, LX, LY, LZ, n_class)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)





os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



# MIXUP= False
# beta_a, beta_b= 0.3, 0.3


i_global = 0
i_gross = 0
best_test = 0
i_eval = -1

#ADD_NOISE = True
#EPOCH_BEGIN_NOISE = 10
#noise_sigma = 0.03


batch_size = 1
n_epochs = 10000


test_interval = 5000
# test_interval_uncr= 100
# n_MC=10

#center_jitter= 36
# center_jitter_x= (SX-LX)//2-5
# center_jitter_y= (SY-LY)//2-5
# center_jitter_z= (SZ-LZ)//2-5


keep_train= 0.9
keep_test= 1.0
# keep_uncert= 0.9




p_trained= np.zeros(n_train)



for epoch_i in range(n_epochs):
    
    # if epoch_i%2==0:
    #     X_train= X_train[:,:,::-1,:,:]
    #     Y_train= Y_train[:,:,::-1,:,:]
    # if epoch_i%3==0:
    #     X_train= X_train[:,::-1,:,:,:]
    #     Y_train= Y_train[:,::-1,:,:,:]
    # if epoch_i%5==0:
    #     X_train= X_train[:,:,:,::-1,:]
    #     Y_train= Y_train[:,:,:,::-1,:]
    
    Ext_train= np.zeros((n_train, 6), np.int)
    for i in range(n_train):
        z= np.where(Y_train[i,:,:,:,0]==0)
        Ext_train[i,:]= z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max()
    
    for i in range(n_train):
        
        # if MIXUP:
            
        #     im_ind= np.random.randint(n_train)
            
        #     x_c, y_c, z_c= SX//2, SY//2, SZ//2
            
        #     x_c+= np.random.randint(-center_jitter_x, center_jitter_x+1)
        #     y_c+= np.random.randint(-center_jitter_y, center_jitter_y+1)
        #     z_c+= np.random.randint(-center_jitter_z, center_jitter_z+1)
            
        #     x_i= np.min( [ np.max( [x_c, LX//2 ] ), SX-LX//2-1 ] )- LX//2
        #     y_i= np.min( [ np.max( [y_c, LY//2 ] ), SY-LY//2-1 ] )- LY//2
        #     z_i= np.min( [ np.max( [z_c, LZ//2 ] ), SZ-LZ//2-1 ] )- LZ//2
            
        #     batch_x1 = X_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        #     batch_y1 = Y_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
            
        #     im_ind= np.random.randint(n_train)
            
        #     x_c, y_c, z_c= SX//2, SY//2, SZ//2
            
        #     x_c+= np.random.randint(-center_jitter_x, center_jitter_x+1)
        #     y_c+= np.random.randint(-center_jitter_y, center_jitter_y+1)
        #     z_c+= np.random.randint(-center_jitter_z, center_jitter_z+1)
            
        #     x_i= np.min( [ np.max( [x_c, LX//2 ] ), SX-LX//2-1 ] )- LX//2
        #     y_i= np.min( [ np.max( [y_c, LY//2 ] ), SY-LY//2-1 ] )- LY//2
        #     z_i= np.min( [ np.max( [z_c, LZ//2 ] ), SZ-LZ//2-1 ] )- LZ//2
            
        #     batch_x2 = X_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        #     batch_y2 = Y_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
            
        #     if np.max( batch_y1[:, LXc:LX-LXc, LYc:LY-LYc, LZc:LZ-LZc, 1:] ) > 0 and np.max( batch_y2[:, LXc:LX-LXc, LYc:LY-LYc, LZc:LZ-LZc, 1:] ) > 0: #True:
        #         beta_lam= beta.rvs(beta_a, beta_b)
        #         batch_x= beta_lam*batch_x1 + (1-beta_lam)*batch_x2
        #         batch_y= beta_lam*batch_y1 + (1-beta_lam)*batch_y2
        #         sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_conv: keep_train})
        #         batch_x = batch_y = 0
                
        # else:
        
        im_ind= i
        
        # x_c, y_c, z_c= SX//2, SY//2, SZ//2
        x_c= (Ext_train[im_ind,1] + Ext_train[im_ind,0])//2
        y_c= (Ext_train[im_ind,3] + Ext_train[im_ind,2])//2
        z_c= (Ext_train[im_ind,5] + Ext_train[im_ind,4])//2
        
        center_jitter_x= (Ext_train[im_ind,1] - Ext_train[im_ind,0])//4
        center_jitter_y= (Ext_train[im_ind,3] - Ext_train[im_ind,2])//4
        center_jitter_z= (Ext_train[im_ind,5] - Ext_train[im_ind,4])//4
        
        x_c+= np.random.randint(-center_jitter_x, center_jitter_x+1)
        y_c+= np.random.randint(-center_jitter_y, center_jitter_y+1)
        z_c+= np.random.randint(-center_jitter_z, center_jitter_z+1)
        
        x_i= np.min( [ np.max( [x_c, LX//2 ] ), SX-LX//2 ] )- LX//2
        y_i= np.min( [ np.max( [y_c, LY//2 ] ), SY-LY//2 ] )- LY//2
        z_i= np.min( [ np.max( [z_c, LZ//2 ] ), SZ-LZ//2 ] )- LZ//2
        
        # assert(x_i>=0 and y_i>=0 and z_i>=0 and x_i+LX<=SX and y_i+LY<=SY and z_i+LZ<=SZ)
        
        # if SX-LX>0:
        #     x_i= np.random.randint(0, SX-LX)
        # else:
        #     x_i= 0
        # if SY-LY>0:
        #     y_i= np.random.randint(0, SY-LY)
        # else:
        #     y_i= 0
        # if SZ-LZ>0:
        #     z_i= np.random.randint(0, SZ-LZ)
        # else:
        #     z_i= 0
        
        batch_x = X_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        batch_y = Y_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        batch_m = M_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ].copy()
        
        '''batch_x, batch_y= seg_util.augment_batch(batch_x, batch_y, epoch_i,
                                              APPLY_DEFORMATION=False, EPOCH_BEGIN_DEFORMATION=np.inf, alpha=2,
                                              APPLY_SHIFT=False, EPOCH_BEGIN_SHIFT=np.inf, shift_x=10, shift_y=10, shift_z=10,
                                              ADD_NOISE=False, EPOCH_BEGIN_NOISE=5, noise_sigma=0.05)'''
        
        if np.max( batch_y[:, LXc:LX-LXc, LYc:LY-LYc, LZc:LZ-LZc, 0] ) == 0: #True:
            # np.mean(batch_y[:,:,:,:,0]==1)>inclusion_threshold:
            #            print (i, np.mean(batch_y[:,:,:,:,0]==1) )
            #batch_eps= np.random.randn(batch_size, LX, LY, LZ, n_class, T)
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, M: batch_m, T:T_matrix,\
                                           learning_rate: L_Rate, p_keep_conv: keep_train})
            batch_x = batch_y = 0
            i_global += 1
            p_trained[im_ind]+= 1
            
            
        i_gross += 1
        
        
        if i_gross % test_interval == 0:
            
            
            i_eval += 1
            
            print('\n' , i_eval, epoch_i, i, i_global, i_gross)
            print(p_trained)
            
            
            dice_c = np.zeros((5, n_class))
            
            for i_c in tqdm(range(5), ascii=True):
                
                y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                y_tr_pr_cnt = np.zeros((SX, SY, SZ))
                
                for lx in lx_list:
                    for ly in ly_list:
                        for lz in lz_list:
                            
                            if np.max( Y_train[i_c:i_c + 1, lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc, 0] ) == 0:
                                
                                batch_x = X_train[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                                
                                pred_temp = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: keep_test})
                                y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                                y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                                
                y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                y_tr_pr_c[y_tr_pr_cnt == 0] = 0
                
                #batch_x = X_train[i_c:i_c + 1, :,:,:, :].copy()
                batch_y = Y_train[i_c , :, :, :, :].copy()
                batch_y = np.argmax(batch_y, axis=-1)
                
                for j_c in range(n_class):
                    dice_c[i_c, j_c] = seg_util.dice( batch_y == j_c , y_tr_pr_c == j_c )
            
            print_text= 'train dice  '
            for j_c in range(n_class):
                print_text+= ', %.3f' % dice_c[:, j_c].mean()
            print(print_text)
            
            
            
            
            dice_c = np.zeros((n_test, n_class))
            
            for i_c in tqdm(range(n_test), ascii=True):
                
                y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                y_tr_pr_cnt = np.zeros((SX, SY, SZ))
                
                for lx in lx_list:
                    for ly in ly_list:
                        for lz in lz_list:
                            
                            if np.max(Y_test[i_c:i_c + 1, lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc, lz + LZc:lz + LZ - LZc, 0]) == 0:
                                
                                batch_x = X_test[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                                
                                pred_temp = sess.run(predicter,  feed_dict={X: batch_x, p_keep_conv: keep_test})
                                y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += pred_temp[0, :, :, :, :]
                                y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                                
                y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                y_tr_pr_c[y_tr_pr_cnt == 0] = 0
                
                # batch_x = X_test[i_c:i_c + 1, :, :, :, :].copy()
                batch_y = Y_test[i_c , :, :, :, :].copy()
                batch_y = np.argmax(batch_y, axis=-1)
                
                for j_c in range(n_class):
                    dice_c[i_c, j_c] = seg_util.dice( batch_y == j_c , y_tr_pr_c == j_c )
                    
                # y_t_c = np.argmax( batch_y[0, :, :, :, :], axis=-1)
                
                # aux.save_pred_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, False, i_c, i_eval, thumbs_dir )
                
                '''if i_eval==0:
                    save_pred_mhds(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, False, i_c, i_eval)
                else:
                    save_pred_mhds(None, None, y_tr_pr_c, False, i_c, i_eval)'''
                
                # dice_c[i_c, n_class] = hd95(y_t_c, y_tr_pr_c)
                
                # y_tr_pr_soft= y_tr_pr_sum[:,:,:,1]/(y_tr_pr_cnt+1e-10)
                # aux.save_pred_soft_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, y_tr_pr_soft, False, i_c, i_eval, thumbs_dir)
                
                # error_mask= aux.seg_2_anulus(y_t_c, radius= 2.0)
                
                # plot_save_path= thumbs_dir + 'calibr_' + str(i_eval) + '.png'
                # ECE, MCE, ECE_curve= aux.estimate_ECE_and_MCE(y_t_c, y_tr_pr_soft, plot_save_path=plot_save_path)
                # dice_c[i_c, n_class+1]= ECE
                # dice_c[i_c, n_class+2]= MCE
                
                # plot_save_path= thumbs_dir + 'calibr_' + str(i_eval) + '_masked.png'
                # ECE, MCE, ECE_curve= aux.estimate_ECE_and_MCE_masked(y_t_c, y_tr_pr_soft, error_mask, plot_save_path=plot_save_path)
                # dice_c[i_c, n_class+3]= ECE
                # dice_c[i_c, n_class+4]= MCE
                
                
            print_text= 'test dice  '
            for j_c in range(n_class):
                print_text+= ', %.3f' % dice_c[:, j_c].mean()
            print(print_text)
            
            #print('test cost   %.3f' % cost_c.mean())
            # print('test dice   %.3f' % dice_c[:, 0].mean(), ', %.3f' % dice_c[:, 1].mean(), ', %.3f' % dice_c[:, 2].mean())
            
            np.savetxt(thumbs_dir + 'stats_test_' + str(i_eval) + '.txt', dice_c, fmt='%6.3f', delimiter=',')
            
            
            
            if dice_c[:,1:].mean()>best_test:
                print('Saving new model checkpoint.')
                best_test = dice_c[:,1:].mean()
                temp_path = model_dir + 'model_saved_' + str(i_eval) + '_' + str(int(round(10000.0 * dice_c[:,1:].mean()))) + '.ckpt'
                saver.save(sess, temp_path)
                
            if i_eval == 0:
                dice_old = dice_c[:,1:].mean()
            else:
                if dice_c[:,1:].mean() < dice_old:
                    L_Rate = L_Rate * 0.95
                dice_old = dice_c[:,1:].mean()
                
            print('learning rate and mean test dice:  ', L_Rate, dice_old)














#   test


restore_model_path= '.../model/model_saved_1.ckpt'
saver.restore(sess, restore_model_path)

labels_df= pd.read_csv( base_dir + 'data/tissue_labels.csv' , delimiter= ',')
labels_all= labels_df.loc[:,"0"].to_numpy()

young_or_old= 'young'

if young_or_old=='young':
    n_class= 34
    test_subject_names= pd.read_csv( base_dir + 'data/test_subject_names_young.csv' , delimiter= ',')
    test_subject_names= test_subject_names.loc[:,"name"].to_numpy()
elif young_or_old=='old':
    n_class= 32
    test_subject_names= pd.read_csv( base_dir + 'data/test_subject_names_old.csv' , delimiter= ',')
    test_subject_names= test_subject_names.loc[:,"name"].to_numpy()


test_dir= '/.../'
os.makedirs(test_dir)

for test_subject_name in test_subject_names:
        print(test_subject_name)
        copyfile(data_dir + test_subject_name ,  test_dir + test_subject_name  )
        copyfile(data_dir + 'tissue-' + test_subject_name ,  test_dir + 'tissue-' + test_subject_name  )



test_image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f)) and not 'tissue' in f]
test_image_files.sort()

n_test= len(test_image_files)

dice_c = np.zeros((n_test, n_class))
haus_c = np.zeros((n_test, n_class))
assd_c = np.zeros((n_test, n_class))

W_gauss= aux.gaussian_w_3d(LX, LY, LZ, LX/2, LY/2, LZ/2)

for i_test in range(n_test):
    
    img_file_name= test_image_files[i_test]
    les_file_name= 'tissue-' + img_file_name
    img= sitk.ReadImage( test_dir + img_file_name )
    les= sitk.ReadImage( test_dir + les_file_name )
    
    img_np = sitk.GetArrayFromImage(img)
    img_np = np.transpose(img_np, [2, 1, 0]).astype(np.float32)
    les_np = sitk.GetArrayFromImage(les)
    les_np = np.transpose(les_np, [2, 1, 0]).astype(np.float32)
    
    temp= img_np[img_np>10]
    img_np/= temp.std()
    
    SX, SY, SZ= les_np.shape
    les_np_new= np.zeros( (SX, SY, SZ, 36), np.uint8 )
    for i_class in range(36):
        mask= les_np==labels_all[i_class]
        les_np_new[mask==1,i_class]= 1
    
    if young_or_old=='young':
        les_np_new= np.delete(les_np_new, 30, -1)
        les_np_new= np.delete(les_np_new, 29, -1)
    elif young_or_old=='old':
        les_np_new= np.delete(les_np_new, 26, -1)
        les_np_new= np.delete(les_np_new, 25, -1)
        les_np_new= np.delete(les_np_new, 24, -1)
        les_np_new= np.delete(les_np_new, 23, -1)
    
    test_shift= LX//4
    lx_list= np.squeeze( np.concatenate( (np.arange(0, SX-LX, test_shift)[:,np.newaxis], np.array([SX-LX])[:,np.newaxis] ) ).astype(np.int) )
    ly_list= np.squeeze( np.concatenate( (np.arange(0, SY-LY, test_shift)[:,np.newaxis], np.array([SY-LY])[:,np.newaxis] ) ).astype(np.int) )
    lz_list= np.squeeze( np.concatenate( (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis], np.array([SZ-LZ])[:,np.newaxis] ) ).astype(np.int) )
    LXc= (LX-10)//2
    LYc= (LY-10)//2
    
    y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
    y_tr_pr_cnt = np.zeros((SX, SY, SZ))
    
    for lx in lx_list:
        for ly in ly_list:
            for lz in lz_list:
                
                if np.min( img_np[lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc] ) > 0:
                    
                    batch_x = img_np[lx:lx + LX, ly:ly + LY, lz:lz + LZ].copy()
                    batch_x= batch_x[np.newaxis, :,:,:, np.newaxis]
                    
                    pred_temp = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
                    
                    pred_temp= pred_temp[0, :, :, :, :]
                    for i_class in range(n_class):
                        pred_temp[:,:,:,i_class]*= W_gauss
                    
                    y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[ :, :, :, :]
                    y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
    
    y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
    y_tr_pr_c[y_tr_pr_cnt == 0] = 0
    
    # y_tr_pr_c= seg_util.fill_multi_class_seg_holes(y_tr_pr_c)
    
    batch_y = les_np_new
    batch_y = np.argmax(batch_y, axis=-1)
    
    for j_c in range(n_class):
        dice_c[i_test, j_c] = seg_util.dice( batch_y == j_c , y_tr_pr_c == j_c )
        if np.sum( y_tr_pr_c == j_c )>0:
            haus_c[i_test, j_c]= hd95( batch_y == j_c , y_tr_pr_c == j_c )
            assd_c[i_test, j_c]= assd( batch_y == j_c , y_tr_pr_c == j_c )
    
    seg_2_save= np.zeros((SX, SY, SZ))
    for i_class in range(n_class):
        mask= y_tr_pr_c==i_class
        if young_or_old=='young':
            seg_2_save[mask]= labels_young[i_class]
        elif young_or_old=='old':
            seg_2_save[mask]= labels_old[i_class]
    
    les= nib.load( test_dir + les_file_name )
    affine_temp= les.affine
    seg_2_save=  nib.Nifti1Image(seg_2_save, affine_temp)
    nib.save(seg_2_save, test_dir + les_file_name.split('.nii')[0]+ '_pred.nii.gz' )


np.savetxt(test_dir + 'dice_c.txt', dice_c, fmt='%6.3f', delimiter=',')
np.savetxt(test_dir + 'haus_c.txt', haus_c, fmt='%6.3f', delimiter=',')
np.savetxt(test_dir + 'assd_c.txt', assd_c, fmt='%6.3f', delimiter=',')

































