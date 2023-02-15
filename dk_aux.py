# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:28:09 2019

Helpers for plotting, saving, etc.

@author: davoo
"""


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from scipy.spatial.distance import directed_hausdorff
import dk_seg
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
import nibabel as nib
from scipy.stats import wasserstein_distance


def save_data_thumbs(x_c, y_t_c, y_a_c, image_index, thumbs_dir, n_class=2):

    n_rows, n_cols = 3, 3

    z = np.where(y_t_c > 0)
    SX, SY, SZ = ( z[0].min()+  z[0].max() ), ( z[1].min() + z[1].max() ) , ( z[2].min() + z[2].max())

    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_t_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX // 2, :, :], vmin=0, vmax= n_class-1)

    fig.savefig(thumbs_dir + 'Data_' + str(image_index) + '.png')

    plt.close(fig)


def save_pred_thumbs(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 3, 3
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    '''plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 4)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 7)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)'''
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_t_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)










def save_pred_thumbs_2d(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 1, 3
      
    fig, ax = plt.subplots(figsize=(16, 6), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :], vmin=0, vmax= n_class-1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)







def save_fa_md_thumbs(batch_x, batch_y, batch_y_n, y_1m, y_2m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 4
    
    AD_vmax= np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ = y_1m.shape
    
    fig, ax = plt.subplots(figsize=(22, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, n_cols+1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 0], vmin=0, vmax= 1)
    plt.subplot(n_rows, n_cols, n_cols+2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 0], vmin=0, vmax= 1)
    plt.subplot(n_rows, n_cols, n_cols+3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    plt.imshow(y_1m[:, :, SZ // 2], vmin=0, vmax= 1)
    plt.subplot(n_rows, n_cols, n_cols+4)
    plt.axis('off')
    plt.imshow(y_2m[:, :, SZ // 2], vmin=0, vmax= AD_vmax)
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)











def save_fa_thumbs(batch_x, batch_y, batch_y_n, y_1m, y_2m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 4
    
    # AD_vmax= np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ = y_1m.shape
    
    fig, ax = plt.subplots(figsize=(22, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, n_cols+1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 0], vmin=0, vmax= 1)
    # plt.subplot(n_rows, n_cols, n_cols+2)
    # plt.axis('off')
    # plt.imshow(batch_y[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 0], vmin=0, vmax= 1)
    # plt.subplot(n_rows, n_cols, n_cols+3)
    # plt.axis('off')
    # plt.imshow(batch_y_n[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    plt.imshow(y_1m[:, :, SZ // 2], vmin=0, vmax= 1)
    # plt.subplot(n_rows, n_cols, n_cols+4)
    # plt.axis('off')
    # plt.imshow(y_2m[:, :, SZ // 2], vmin=0, vmax= AD_vmax)
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)









def save_cfa_thumbs(batch_x, batch_y, batch_y_n, y_1m, y_2m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 4
    
    # AD_vmax= np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ , _= y_1m.shape
    
    fig, ax = plt.subplots(figsize=(22, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, n_cols+1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, :]*3, vmin=0, vmax= 1, cmap='gray')
    # plt.subplot(n_rows, n_cols, n_cols+2)
    # plt.axis('off')
    # plt.imshow(batch_y[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, :]*3, vmin=0, vmax= 1, cmap='gray')
    # plt.subplot(n_rows, n_cols, n_cols+3)
    # plt.axis('off')
    # plt.imshow(batch_y_n[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    plt.imshow(y_1m[:, :, SZ // 2, :]*3, vmin=0, vmax= 1, cmap='gray')
    plt.subplot(n_rows, n_cols, n_cols+4)
    plt.axis('off')
    plt.imshow(y_2m[:, :, SZ // 2, :]*3, vmin=0, vmax= 1, cmap='gray')
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)









def save_tensor_thumbs(batch_x, batch_y, batch_y_n, y_1m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 3, 6
    
    # AD_vmax= np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ , _= y_1m.shape
    
    fig, ax = plt.subplots(figsize=(22, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    for i in range(6):
        
        plt.subplot(n_rows, n_cols, n_cols+i+1)
        plt.axis('off')
        plt.imshow(batch_y[:, :, SZ // 2, i], cmap='gray')  #, vmin=0, vmax= 1, cmap='gray')
    
    for i in range(6):
        
        plt.subplot(n_rows, n_cols, 2*n_cols+i+1)
        plt.axis('off')
        plt.imshow(y_1m[:, :, SZ // 2, i], cmap='gray')  #, vmin=0, vmax= 1, cmap='gray')
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)







def save_famd_thumbs(batch_x, batch_y, batch_y_n, y_1m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 2
    
    # AD_vmax= np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ , _= y_1m.shape
    
    fig, ax = plt.subplots(figsize=(16, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    plt.subplot(n_rows, n_cols, n_cols+1)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 0], cmap='gray')  #, vmin=0, vmax= 1, cmap='gray')
    
    plt.subplot(n_rows, n_cols, n_cols+2)
    plt.axis('off')
    plt.imshow(y_1m[:, :, SZ // 2, 0], cmap='gray')  #, vmin=0, vmax= 1, cmap='gray')
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)









def save_ten_thumbs(batch_x, batch_y, batch_y_n, y_m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 2
    
    AD_vmax= 0.002#np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ, _ = y_m.shape
    
    fig, ax = plt.subplots(figsize=(12, 12), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    plt.imshow(y_m[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)





def save_ten_thumbs_2(batch_x, batch_y, batch_y_n, y_m, training_flag, image_index, iteration_count, prediction_dir, perc_hi= 99):
    
    n_rows, n_cols = 2, 4
    
    AD_vmax= 0.002#np.percentile(batch_y[:, :, :, 1], perc_hi)
    
    SX, SY, SZ, _ = y_m.shape
    
    fig, ax = plt.subplots(figsize=(22, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 0], cmap='gray')
    plt.subplot(n_rows, n_cols, n_cols+1)
    plt.axis('off')
    plt.imshow(batch_x[:, :, SZ // 2, 1], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    plt.subplot(n_rows, n_cols, n_cols+2)
    plt.axis('off')
    plt.imshow(batch_y[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    plt.subplot(n_rows, n_cols, n_cols+3)
    plt.axis('off')
    plt.imshow(batch_y_n[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    plt.imshow(y_m[:, :, SZ // 2, 0], vmin=0, vmax= AD_vmax)
    plt.subplot(n_rows, n_cols, n_cols+4)
    plt.axis('off')
    plt.imshow(y_m[:, :, SZ // 2, 1], vmin=0, vmax= AD_vmax)
    
    plt.tight_layout()
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)









def save_pred_thumbs_seg_centr(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    z = np.where(y_t_c > 0)
    
    if len(z[0])<10:
        
        print('Segmentation mask smaller than 10; image index: ', image_index)
        
    else:
        
        x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
        
        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)
        
        n_rows, n_cols = 3, 3
        
        fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
        
        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(x_c[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 3)
        plt.imshow(x_c[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 4)
        plt.imshow(y_t_c[:, :, SZ // 2])
        plt.subplot(n_rows, n_cols, 5)
        plt.imshow(y_t_c[:, SY // 2, :])
        plt.subplot(n_rows, n_cols, 6)
        plt.imshow(y_t_c[SX // 2, :, :])
        plt.subplot(n_rows, n_cols, 7)
        plt.imshow(y_p_c[:, :, SZ // 2])
        plt.subplot(n_rows, n_cols, 8)
        plt.imshow(y_p_c[:, SY // 2, :])
        plt.subplot(n_rows, n_cols, 9)
        plt.imshow(y_p_c[SX // 2, :, :])
        
        if training_flag:
            fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
        else:
            fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
            
        plt.close(fig)



#
# def save_pred_thumbs(x_c, y_t_c, y_a_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
#
#     n_rows, n_cols = 3, 4
#
#     SX, SY, SZ= x_c.shape
#
#     fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
#
#     plt.subplot(n_rows, n_cols, 1)
#     plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
#     plt.subplot(n_rows, n_cols, 5)
#     plt.imshow(x_c[:, SY // 2, :], cmap='gray')
#     plt.subplot(n_rows, n_cols, 9)
#     plt.imshow(x_c[SX // 2, :, :], cmap='gray')
#     plt.subplot(n_rows, n_cols, 2)
#     plt.imshow(y_t_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 6)
#     plt.imshow(y_t_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 10)
#     plt.imshow(y_t_c[SX // 2, :, :])
#     plt.subplot(n_rows, n_cols, 3)
#     plt.imshow(y_a_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 7)
#     plt.imshow(y_a_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 11)
#     plt.imshow(y_a_c[SX // 2, :, :])
#     plt.subplot(n_rows, n_cols, 4)
#     plt.imshow(y_p_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 8)
#     plt.imshow(y_p_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 12)
#     plt.imshow(y_p_c[SX // 2, :, :])
#     if training_flag:
#         fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
#     else:
#         fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
#     plt.close(fig)




def save_cm(CM, training_flag, iteration_count, label_tags, prediction_dir):
    
    n_rows, n_cols = 1,1
    
    fig, ax = plt.subplots(figsize=(10, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    ax.get_xaxis().set_visible(False)
    plt.imshow(CM)
    ax.set_xticklabels( ['']+label_tags )
    ax.set_yticklabels( ['']+label_tags )
    
    if training_flag:
        fig.savefig(prediction_dir + 'CM_train_' + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'CM_test_' + '_' + str(iteration_count) + '.png')
    plt.close(fig)



def predict_with_shifts(sess, X, softmax_linear, p_keep_conv, batch_x, n_class, MX, MY, MZ, xs_v, ys_v, zs_v):
    
    _, a_x, b_x, c_x, n_channel = batch_x.shape
    y_prob= np.zeros((a_x + MX, b_x + MY, c_x + MZ, n_class), np.float32)
    i_prob= 0
    
    for xs in xs_v:
        for ys in ys_v:
            for zs in zs_v:
                
                x = batch_x[0, :, :, :, :].copy()
                
                batch_xx= np.zeros(batch_x.shape, dtype=np.float32)
                
                xx = np.zeros((a_x + MX, b_x + MY, c_x + MZ, n_channel), np.float32)
                xx[MX // 2:MX // 2 + a_x, MY // 2:MY // 2 + b_x, MZ // 2:MZ // 2 + c_x,:] = x.copy()
                
                batch_xx[0, :, :, :, :] = xx[xs:xs+a_x, ys:ys+b_x, zs:zs+c_x, :].copy()
                
                y_prob_c= sess.run(softmax_linear, feed_dict={X: batch_xx, p_keep_conv: 1.0})
                
                y_prob[xs:xs+a_x, ys:ys+b_x, zs:zs+c_x, :]+= y_prob_c[0,:,:,:]
                
                i_prob+= 1
                
                x = xx = 0
                
    y_prob= y_prob[MX // 2:MX // 2 + a_x, MY // 2:MY // 2 + b_x, MZ // 2:MZ // 2 + c_x,:]
    y_prob/= i_prob
    
    return y_prob


def my_softmax(x):
    x_max = np.max(x, axis=-1)
    x_max = x_max[:,:,:, np.newaxis]
    x= np.exp(x - x_max)
    x_sum= np.sum(x, axis=-1)
    x_sum = x_sum[:,:,:, np.newaxis]
    return x / x_sum




def save_pred_uncr_thumbs(x_c, y_t_c, y_a_c, y_unc, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_unc[:, :, SZ // 2], vmin=0, vmax= 0.367)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_unc[:, SY // 2, :], vmin=0, vmax= 0.367)
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_unc[SX // 2, :, :], vmin=0, vmax= 0.367)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_uncr.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_uncr.png')
    plt.close(fig)





def save_pred_uncr_err_thumbs(x_c, y_t_c, y_a_c, y_unc, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 5
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    y_err= np.logical_or(  np.logical_and( y_t_c<0.5, y_a_c>0.5) , np.logical_and( y_t_c>0.5, y_a_c<0.5)  )
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 6)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 11)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 13)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_unc[:, :, SZ // 2], vmin=0, vmax= 0.367)
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_unc[:, SY // 2, :], vmin=0, vmax= 0.367)
    plt.subplot(n_rows, n_cols, 14)
    plt.imshow(y_unc[SX // 2, :, :], vmin=0, vmax= 0.367)
    
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_err[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_err[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 15)
    plt.imshow(y_err[SX // 2, :, :])
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_err.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_err.png')
    plt.close(fig)






def save_img_slice_and_seg_boundary_cntr(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, markersize=1):
    
    n_rows, n_cols = 1, 3
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    
    slc_in= 2
    slc_no= SZ//2
    
    b= dk_seg.seg_2_boundary_3d(y_t_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(y_a_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[1][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 1)
    
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    
    plt.imshow(x_c[:,:,slc_no], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    
    
    slc_in= 1
    slc_no= SY//2
    
    b= dk_seg.seg_2_boundary_3d(y_t_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[2][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(y_a_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[2][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 2)
    
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    
    plt.imshow(x_c[:,slc_no,:], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    
    
    slc_in= 0
    slc_no= SX//2
    
    b= dk_seg.seg_2_boundary_3d(y_t_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel= z[1][slc_sel].astype(np.int)
    y_sel= z[2][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(y_a_c)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel2= z[1][slc_sel].astype(np.int)
    y_sel2= z[2][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 3)
    
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    
    plt.imshow(x_c[slc_no,:,:], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_bd.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_bd.png')
    plt.close(fig)
    
    
    



def save_pred_soft_thumbs(x_c, y_t_c, y_a_c, y_soft, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_soft[:, :, SZ // 2], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_soft[:, SY // 2, :], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_soft[SX // 2, :, :], vmin=0, vmax=1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_soft.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_soft.png')
    plt.close(fig)




def save_uncrt_soft_thumbs(x_c, y_t_c, y_a_c, y_soft, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_soft[:, :, SZ // 2], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_soft[:, SY // 2, :], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_soft[SX // 2, :, :], vmin=0, vmax=1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_uncrt_soft.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_uncrt_soft.png')
    plt.close(fig)
    
    
    

    
    
    


def save_pred_mhds(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    if x_c is not None:
        x= np.transpose(x_c, [2,1,0])
        x= sitk.GetImageFromArray(x)
        if training_flag:
            sitk.WriteImage(x, prediction_dir +  'X_train_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
        else:
            sitk.WriteImage(x, prediction_dir +  'X_test_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    if y_t_c is not None:
        x= np.transpose(y_t_c, [2,1,0])
        x= sitk.GetImageFromArray(x)
        if training_flag:
            sitk.WriteImage(x, prediction_dir +  'X_train_gold' + str(image_index) + '_' + str(iteration_count) + '.mhd')
        else:
            sitk.WriteImage(x, prediction_dir +  'X_test_gold'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_p_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_pred' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_pred'  + str(image_index) + '_' + str(iteration_count) + '.mhd')



def save_pred_uncr_mhds(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    x= np.transpose(x_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_f_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_f_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_t_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_s_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_s_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_p_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_t_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_t_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')




def divide_patint_wise(patient_code, n_fold, i_fold):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    np.random.seed(0)
    np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test = np.zeros(len(patient_code), np.int)
    p_train = np.zeros(len(patient_code), np.int)
    n_test = n_train = -1
    
    for i in range(len(patient_code)):
        if patient_code[i] in patient_test:
            n_test += 1
            p_test[n_test] = i
        else:
            n_train += 1
            p_train[n_train] = i
            
    return p_test[:n_test + 1], p_train[:n_train + 1]






def divide_patint_wise_with_gold_dwi(patient_code, patient_code_gold, n_fold, i_fold, random=True, train_on_noisy=True):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    if random:
        np.random.seed(0)
        np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test = np.zeros(len(patient_code), np.int)
    p_train = np.zeros(len(patient_code), np.int)
    n_test = n_train = -1
    
    for i in range(len(patient_code_gold)):
        if patient_code_gold[i] in patient_test:
            n_test += 1
            p_test[n_test] = i
            
    if train_on_noisy:
        for i in range(len(patient_code)):
            if not patient_code[i] in patient_test:
                n_train += 1
                p_train[n_train] = i
    else:
        for i in range(len(patient_code_gold)):
            if not patient_code_gold[i] in patient_test:
                n_train += 1
                p_train[n_train] = i
            
    return p_test[:n_test + 1], p_train[:n_train + 1]




def divide_patint_wise_with_gold(patient_code, patient_code_gold, n_fold, i_fold, random=True):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    if random:
        np.random.seed(0)
        np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test_clean  = np.zeros(len(patient_code), np.int)
    p_train_clean = np.zeros(len(patient_code), np.int)
    p_test_noisy  = np.zeros(len(patient_code), np.int)
    p_train_noisy = np.zeros(len(patient_code), np.int)
    n_test_clean = n_train_clean = n_test_noisy = n_train_noisy = -1
    
    for i in range(len(patient_code_gold)):
        if patient_code_gold[i] in patient_test:
            n_test_clean += 1
            p_test_clean[n_test_clean] = i
        else:
            n_train_clean += 1
            p_train_clean[n_train_clean] = i
    
    for i in range(len(patient_code)):
        if patient_code[i] in patient_test:
            n_test_noisy += 1
            p_test_noisy[n_test_noisy] = i
        else:
            n_train_noisy += 1
            p_train_noisy[n_train_noisy] = i
    
    return p_test_noisy[:n_test_noisy + 1], p_train_noisy[:n_train_noisy + 1], p_test_clean[:n_test_clean + 1], p_train_clean[:n_train_clean + 1]







def pruning_error(y_true, y_pred, y_uncert=None, uncert_prcnt=None, mode='Dice'):

    if mode == 'Dice':

        dice_num = 2 * np.sum((y_true == 1) * (y_pred == 1)) + 0
        dice_den = np.sum(y_true == 1) + np.sum(y_pred == 1) + 1
        err = - dice_num / dice_den

    elif mode == 'Hausdorff':

        y_true_b = dk_seg.seg_2_boundary_3d(y_true)
        y_pred_b = dk_seg.seg_2_boundary_3d(y_pred)

        z = np.nonzero(y_true_b)
        zx, zy, zz = z[0], z[1], z[2]
        contour_true = np.vstack([zx, zy, zz]).T
        z = np.nonzero(y_pred_b)
        zx, zy, zz = z[0], z[1], z[2]
        contour_pred = np.vstack([zx, zy, zz]).T

        err = max(directed_hausdorff(contour_true, contour_pred)[0],
                  directed_hausdorff(contour_pred, contour_true)[0])

    elif mode == 'Uncertainry':

        y_uncert = y_uncert[y_true == 1]
        err = np.percentile(y_uncert, uncert_prcnt)

    else:

        print('Unrecognized error mode; returning zero.')
        err = 0

    return err



def samples_2_keep(err_tr, prune_perct=95):

    perc = np.percentile(err_tr, prune_perct)

    ind = np.where(err_tr < perc)

    return ind






def gaussian_w_3d(sx, sy, sz, sigx, sigy, sigz):
    
    wx = np.exp(- np.abs(np.linspace(-sx / 2 + 1 / 2, sx / 2 - 1 / 2, sx)) ** 2 / sigx ** 2)
    wy = np.exp(- np.abs(np.linspace(-sy / 2 + 1 / 2, sy / 2 - 1 / 2, sy)) ** 2 / sigy ** 2)
    wz = np.exp(- np.abs(np.linspace(-sz / 2 + 1 / 2, sz / 2 - 1 / 2, sz)) ** 2 / sigz ** 2)
    
    wxy= np.matmul( wx[:,np.newaxis], wy[np.newaxis,:] )
    
    w  = np.matmul(wxy[:,:, np.newaxis], wz[np.newaxis, :])
    
    return w




def seg_2_bounding_box(y, index):
    
    if np.mean(y==0)+ np.mean(y==1)<1:
        
        print('The segmentation mask must include ones and zeros only!    Returning NaN. ', index)
        return np.nan
    
    if y.shape[-1]!=2:
        
        print('Input segmentation mask must have two channels!   Returning NaN. ', index)
        return np.nan
    
    z = np.where(y[:,:,:,1] > 0.5)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
    
    yt= np.zeros( y.shape[:-1] )
    yt[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]= 1
    
    yn= np.zeros( y.shape )
    yn[:, :, :, 1] = yt
    yn[:, :, :, 0] = 1-yt
    
    return yn


def save_image_and_maskt_humbs(x_i, y_i, x_f, y_f, vol_name, thumbs_dir):

    z_i = np.where(y_i > 0)
    z_f = np.where(y_f > 0)

    if len(z_i[0]) < 10:

        print('Segmentation mask smaller than 10; image index: ') #, image_index)

    else:

        n_rows, n_cols = 3, 4

        fig, ax = plt.subplots(figsize=(20, 13), nrows=n_rows, ncols=n_cols)

        x_min, x_max, y_min, y_max, z_min, z_max = z_i[0].min(), z_i[0].max(), z_i[1].min(), z_i[1].max(), z_i[2].min(), z_i[ 2].max()

        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)

        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(x_i[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 5)
        plt.imshow(x_i[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 9)
        plt.imshow(x_i[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(y_i[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 6)
        plt.imshow(y_i[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 10)
        plt.imshow(y_i[SX // 2, :, :], cmap='gray')

        x_min, x_max, y_min, y_max, z_min, z_max = z_f[0].min(), z_f[0].max(), z_f[1].min(), z_f[1].max(), z_f[2].min(), z_f[2].max()

        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)

        plt.subplot(n_rows, n_cols, 3)
        plt.imshow(x_f[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 7)
        plt.imshow(x_f[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 11)
        plt.imshow(x_f[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 4)
        plt.imshow(y_f[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 8)
        plt.imshow(y_f[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 12)
        plt.imshow(y_f[SX // 2, :, :], cmap='gray')

    fig.savefig(thumbs_dir + vol_name + '.png')
    plt.close(fig)




def create_rough_mask_old(x, percentile=95, closing_rad=3):
    
    seg= np.uint(x>np.nanpercentile(x, percentile))
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    largest_component= -1
    largest_size= -1
    
    for i in range(1, n_component + 1):
        
        if np.sum(seg == i) > largest_size:
            largest_component= i
            largest_size= np.sum(seg==i)
            
    if largest_component==-1:
        print('Segmentation empty! Returning NaN')
        seg= np.nan
    else:
        seg= np.uint(seg==largest_component)
        seg = sitk.GetImageFromArray(seg)
        seg = sitk.BinaryMorphologicalClosing(seg, closing_rad)
        seg = sitk.GetArrayFromImage(seg)
        
    return seg




def create_rough_mask(x, percentile=95, closing_rad=3):
    
    seg= np.uint(x>np.nanpercentile(x, percentile))
    
    seg = sitk.GetImageFromArray(seg)
    
    seg = sitk.BinaryMorphologicalClosing(seg, 1)
    
    seg = sitk.BinaryMorphologicalOpening(seg, 1)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    seg = c_filter.Execute(seg)
    
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    largest_component= -1
    largest_size= -1
    
    for i in range(1, n_component + 1):
        
        if np.sum(seg == i) > largest_size:
            largest_component= i
            largest_size= np.sum(seg==i)
            
    if largest_component==-1:
        print('Segmentation empty! Returning NaN')
        seg= np.nan
    else:
        seg= np.uint(seg==largest_component)
        seg = sitk.GetImageFromArray(seg)
        seg = sitk.BinaryMorphologicalClosing(seg, closing_rad)
        seg = sitk.GetArrayFromImage(seg)
        
    return seg



def create_rough_mask_staple(x, y, percentile_vector=np.array([93, 95, 98]), closing_rad_vector=np.array([15,20,30])):
    
    segmentations= list()
    perfs= list()
    
    for percentile in percentile_vector:
        for closing_rad in closing_rad_vector:
            
            seg= np.uint(x>np.nanpercentile(x, percentile))
            
            c_filter = sitk.ConnectedComponentImageFilter()
            c_filter.FullyConnectedOn()
            
            seg = sitk.GetImageFromArray(seg)
            seg = c_filter.Execute(seg)
            seg = sitk.GetArrayFromImage(seg)
            
            n_component = seg.max()
            
            largest_component= -1
            largest_size= -1
            
            for i in range(1, n_component + 1):
                
                if np.sum(seg == i) > largest_size:
                    largest_component= i
                    largest_size= np.sum(seg==i)
                    
            if largest_component==-1:
                print('Segmentation empty!')
            else:
                seg= np.uint(seg==largest_component)
                seg = sitk.GetImageFromArray(seg)
                seg = sitk.BinaryMorphologicalClosing(seg, int(closing_rad))
                segmentations.append(seg)
                temp= sitk.GetArrayFromImage(seg)
                perfs.append(dk_seg.dice(y, temp))
    
    foregroundValue = 1
    threshold = 0.95
    seg_staple = sitk.STAPLE(segmentations, foregroundValue) > threshold
    
    seg= sitk.GetArrayFromImage(seg_staple)
    
    return seg, perfs
    


def empty_folder(this_folder):
    
    for file in os.listdir(this_folder):
        file_path = os.path.join(this_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)




def save_img_slice_and_seg_boundary(x, y, slc_no, vol_name, thumbs_dir, y2, slc_in=2, markersize=1):

    b= dk_seg.seg_2_boundary_3d(y)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
        
    b= dk_seg.seg_2_boundary_3d(y2)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[1][slc_sel].astype(np.int)
    
    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
    
    plt.subplot(1, 1, 1)
    plt.axis('off')
    
    plt.imshow(x[:,:,slc_no], cmap='gray')
    
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    
    fig.savefig(thumbs_dir + vol_name + '.png')
    plt.axis('off')
    plt.close(fig)
    
    



def compute_seg_vol_to_surface_ratio(y):
    
    b= dk_seg.seg_2_boundary_3d(y>0.5)
    
    vol= np.sum(y>0.5)
    
    surf= np.sum(b)
    
    return vol/surf



def compute_seg_vol_to_diamater_ratio(y):
    
    b= dk_seg.seg_2_boundary_3d(y>0.5)
    b= np.where(b>0)
    b= np.vstack((b[0], b[1], b[2])).T
    
    D = pdist(b)
    D = squareform(D)
    diameter= D.max()
    
    vol= np.sum(y>0.5)
    
    return vol, diameter, vol/diameter





def weight_matrix(lx, ly, lz, sigx, sigy, sigz, n_channel=1):
    
    wx= np.exp( - ( np.arange(1, lx+1)-lx/2 )**2/ sigx**2 )
    wy= np.exp( - ( np.arange(1, ly+1)-ly/2 )**2/ sigy**2 )
    wz= np.exp( - ( np.arange(1, lz+1)-lz/2 )**2/ sigz**2 )
    
    wxy= np.matmul( wx[:,np.newaxis] , wy[np.newaxis,:] )
    wxyz= np.matmul( wxy[:,:,np.newaxis] , wz[np.newaxis,:] )
    
    if n_channel==1:
        W= wxyz
    else:
        W= np.zeros((lx,ly,lz,n_channel))
        for i_channel in range(n_channel):
            W[:,:,:,i_channel]= wxyz.copy()
    
    return W





def save_multiscale_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, iter_num=-1, save_dir= None, figsize=(22, 13)):
    
    n_img= len(my_list)
    
    n_rows= int(np.sqrt(n_img))
    n_cols= n_rows
    while n_rows*n_cols<n_img:
        n_cols+=1
    
    if direction=='X':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    for i in range(len(my_list)):
        plt.subplot(n_rows, n_cols, i+1)
        if i==0:
            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
        else:
            plt.imshow( my_list[i] [:, :, SZ ])
        '''elif i==1:
            plt.imshow( my_list[i] [:, :, SZ ])
        else:
            plt.imshow( my_list[i] [:, :, SZ ]-my_list[1] [:, :, SZ ])'''
    
    plt.tight_layout()
    
    if iter_num>-1:
        fig.savefig(save_dir + 'thumbs_' + direction + '_' + str(image_index) + '_' + str(iter_num) + '.png')
    else:
        fig.savefig(save_dir + 'thumbs_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)






def save_cae_thumbs(x_c, x_r, training_flag, image_index, iteration_count, prediction_dir):

    n_rows, n_cols = 2, 3

    SX, SY, SZ = x_c.shape

    fig, ax = plt.subplots(figsize=(16, 10), nrows=n_rows, ncols=n_cols)

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_r[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(x_r[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(x_r[SX // 2, :, :], cmap='gray')
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    
    plt.close(fig)






def save_dwi_rough_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None):
    
    n_img= len(my_list)
    n_rows, n_cols = 2, (n_img+1)//2
    
    if direction=='X':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    for i in range(len(my_list)):
        plt.subplot(n_rows, n_cols, i+1)
        if i==0:
            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
        else:
            plt.imshow( my_list[i] [:, :, SZ ])
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'DWI_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)






def save_dwi_rough_boundaries( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None, figsize=(8, 8), markersize=1, colors=['.r', '.b', '.g', '.c', '.y', '.k']):
    
    n_rows, n_cols = 1, 1
    
    if direction=='X':
        slc_in= 0
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        slc_in= 1
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        slc_in= 2
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
    plt.axis('off');
    
    for i in range(2, len(my_list)):
        
        y_temp= my_list[i]
        
        b= dk_seg.seg_2_boundary_3d(y_temp)
        z= np.where(b>0)
        slc_all= z[slc_in].astype(np.int)
        slc_sel= slc_all==SZ
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    i= 1
    
    y_temp= my_list[i]
    
    b= dk_seg.seg_2_boundary_3d(y_temp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==SZ
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    fig.savefig(save_dir + 'DWI_boundary_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)











                
def save_co_teaching_boundaries( my_list, slice_ind=-1, direction= 'Z', image_index=None, eval_index= None, train=False, save_dir= None, 
                                figsize=(8, 8), markersize=1, colors=['.r', '.b', '.g', '.c', '.y', '.k']):
    
    n_rows, n_cols = 1, 1
    
    if direction=='X':
        slc_in= 0
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        slc_in= 1
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        slc_in= 2
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
    plt.axis('off');
    
    for i in range(2, len(my_list)):
        
        y_temp= my_list[i]
        
        b= dk_seg.seg_2_boundary_3d(y_temp)
        z= np.where(b>0)
        slc_all= z[slc_in].astype(np.int)
        slc_sel= slc_all==SZ
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    i= 1
    
    y_temp= my_list[i]
    
    b= dk_seg.seg_2_boundary_3d(y_temp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==SZ
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    if train:
        fig.savefig(save_dir + 'X_ct_train_' + direction + '_' + str(image_index) + '_' + str(eval_index) + '.png')
    else:
        fig.savefig(save_dir + 'X_ct_test_' + direction + '_' + str(image_index) + '_' + str(eval_index) + '.png')
    
    plt.close(fig)







def resample_imtar_to_imref(im_tar, im_ref, resampling_method= sitk.sitkLinear, match_ref_pixeltype=False):
    
    if match_ref_pixeltype:
        pixeltype= im_ref.GetPixelIDValue()
    else:
        pixeltype= im_tar.GetPixelIDValue()
    
    I = sitk.Image(im_ref.GetSize(), pixeltype)
    I.SetSpacing(im_ref.GetSpacing())
    I.SetOrigin(im_ref.GetOrigin())
    I.SetDirection(im_ref.GetDirection())
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(I)
    resample.SetInterpolator( resampling_method )
    resample.SetTransform(sitk.Transform())
    
    I = resample.Execute(im_tar)
    
    return I
    






def save_dwi_gold_boundaries( my_list, image_index=None, save_dir= None, markersize=1, colors=['.r', '.b']):
    
    n_rows, n_cols = 1, 3
    
    fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    i_fig= 0
    
    for direction in ['Z', 'X', 'Y']:
        
        if direction=='X':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list[i] , [2,1,0] )
        elif direction=='Y':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list[i] , [0,2,1] )
        elif direction=='Z':
            slc_in= 2
            pass
        
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        
        plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
        plt.axis('off');
        
        for i in range(1, len(my_list)):
            
            y_temp= my_list[i]
            
            b= dk_seg.seg_2_boundary_3d(y_temp)
            z= np.where(b>0)
            slc_all= z[slc_in].astype(np.int)
            slc_sel= slc_all==SZ
            x_sel= z[0][slc_sel].astype(np.int)
            y_sel= z[1][slc_sel].astype(np.int)
            plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'X_dwi_seg_gold_' + str(image_index) + '.png')
    
    plt.close(fig)













def number_n_size_of_cc(seg):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    size= np.zeros(n_component)
    
    for i in range(n_component):
        
        size[i]= np.sum(seg==i+1)
    
    return n_component, size

#def save_data_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None):
#    
#    n_img= len(my_list)
#    n_rows, n_cols = 2, (n_img+1)//2
#    
#    if direction=='X':
#        for i in range(len(my_list)):
#            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
#    elif direction=='Y':
#        for i in range(len(my_list)):
#            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
#    elif direction=='Z':
#        pass
#    else:
#        print('Direction not valid')
#        return None
#    
#    if slice_ind==-1:
#        y_t= my_list[1]
#        z = np.where(y_t > 0)
#        SZ = ( z[2].min() + z[2].max() ) // 2
#        
#    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
#    
#    for i in range(len(my_list)):
#        plt.subplot(n_rows, n_cols, i+1)
#        if i==0:
#            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
#        elif i==1:
#            plt.imshow( my_list[i] [:, :, SZ ])
#        else:
#            plt.imshow( my_list[i] [:, :, SZ ]-my_list[1] [:, :, SZ ])
#    
#    fig.savefig(save_dir + 'CPSP_' + direction + '_' + str(image_index) + '.png')
#    
#    plt.close(fig)




def nbr_sum_6(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            shift_x, shift_y, shift_z= 1, 0, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 0, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 1, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, -1, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 0, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 0, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
                        
    return y



def nbr_sum_8(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            shift_x, shift_y, shift_z= 1, 1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, 1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, -1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, -1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, -1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, -1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
                       
    return y






def nbr_sum_26(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            for shift_x in range(-1, 2):
                for shift_y in range(-1, 2):
                    for shift_z in range(-1, 2):
                        y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            
            y-= x
            
    return y


def nbr_sum_124(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            for shift_x in range(-2, 3):
                for shift_y in range(-2, 3):
                    for shift_z in range(-2, 3):
                        y[2:-2,2:-2,2:-2]+= x[2+shift_x:a-2+shift_x, 2+shift_y:b-2+shift_y, 2+shift_z:c-2+shift_z]
            
            y-= x
            
    return y





def estimate_ECE_and_MCE(seg_true, seg_pred, N=10, plot_save_path=''):
    
    ECE_curve= np.zeros((N,3))
    
    for i in range(N):
        
        p0= i/N
        p1= (i+1)/N
        
        mask= np.logical_and(seg_pred>p0, seg_pred<p1)
        
        if mask.sum()>0:
            
            pos= mask*seg_true
            
            mean_p= seg_pred[mask].mean()
            mean_q= pos.sum()/ mask.sum()
            frac =  mask.mean()
            
            ECE_curve[i,:]= mean_p, mean_q, frac
        
    ECE= np.sum(ECE_curve[:,2] * np.abs(ECE_curve[:,0]- ECE_curve[:,1] ))
    MCE= np.abs(ECE_curve[:,0]- ECE_curve[:,1] ).max()
    
    if not plot_save_path is None:
        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot( ECE_curve[:,0], ECE_curve[:,1], '.')
        fig.savefig(plot_save_path)
        plt.close(fig)
    
    return ECE, MCE, ECE_curve








def estimate_ECE_and_MCE_masked(seg_true, seg_pred, error_mask, N=10, plot_save_path=''):
    
    ECE_curve= np.zeros((N,3))
    
    for i in range(N):
        
        p0= i/N
        p1= (i+1)/N
        
        mask= np.logical_and( np.logical_and(seg_pred>p0, seg_pred<p1), error_mask )
        
        if mask.sum()>0:
            
            pos= mask*seg_true
            
            mean_p= seg_pred[mask].mean()
            mean_q= pos.sum()/ ( mask.sum() + 1e-6 )
            frac =  mask.mean()
            
            ECE_curve[i,:]= mean_p, mean_q, frac
        
    frac_sum= np.sum(ECE_curve[:,2])
    ECE_curve[:,2]/= (frac_sum+1e-6)
    
    ECE= np.sum(ECE_curve[:,2] * np.abs(ECE_curve[:,0]- ECE_curve[:,1] ))
    MCE= np.abs(ECE_curve[:,0]- ECE_curve[:,1] ).max()
    
    if not plot_save_path is None:
        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot( ECE_curve[:,0], ECE_curve[:,1], '.')
        fig.savefig(plot_save_path)
        plt.close(fig)
    
    return ECE, MCE, ECE_curve





def estimate_RegECE_curve(e_org, s_org, scales=[1], N=10, rejection_percentile=99.5):
    
    e_org= np.abs(e_org)
    
    ENCE_best= np.inf
    
    for scale in scales:
        
        e= e_org.copy()
        s= s_org.copy()
        
        ind_s= s<np.percentile(s, rejection_percentile)
        ind_e= e<np.percentile(e, rejection_percentile)
        
        ind= np.logical_and(ind_s, ind_e)
        
        e= e[ind]
        s= s[ind]
        
        s*= e.max()/s.max()*scale
        
        s0, sN= s.min(), s.max()
        
        RegECE_curve= np.zeros((N,3))
        
        del_s= (sN- s0)/N
        
        for i in range(N):
            
            p0= s0 + i * del_s
            p1= p0 + del_s
            
            mask= np.logical_and(s>=p0, s<p1)
            
            if mask.sum()>0:
                
                mean_p= s[mask].mean()
                mean_q= e[mask].mean()
                frac =  mask.mean()
                
                RegECE_curve[i,:]= mean_p, mean_q, frac
            
        ENCE= np.sum( RegECE_curve[:,2] * np.abs(RegECE_curve[:,0]- RegECE_curve[:,1] ) )
        
        if ENCE<ENCE_best:
            ENCE_best= ENCE
            RegECE_curve_best= RegECE_curve
        
    return ENCE_best, RegECE_curve_best







def estimate_RegECE_curve_noscale(e_org, s_org, N=10, rejection_percentile=99.5):
    
    e_org= np.abs(e_org)
    
    e= e_org.copy()
    s= s_org.copy()
    
    # print(s.mean(), e.mean())
    
    ind_s= s<np.percentile(s, rejection_percentile)
    ind_e= e<np.percentile(e, rejection_percentile)
    
    ind= np.logical_and(ind_s, ind_e)
    
    e= e[ind]
    s= s[ind]
    
    # s0, sN= s.min(), s.max()
    s0, sN= min(s.min(), e.min()) , max( s.max(), e.max())
    
    RegECE_curve= np.zeros((N,3))
    
    del_s= (sN- s0)/N
    
    for i in range(N):
        
        p0= s0 + i * del_s
        p1= p0 + del_s
        
        mask= np.logical_and(s>=p0, s<p1)
        
        if mask.sum()>0:
            
            mean_p= s[mask].mean()
            mean_q= e[mask].mean()
            frac =  mask.mean()
            
            RegECE_curve[i,:]= mean_p, mean_q, frac
        
    ENCE= np.sum( RegECE_curve[:,2] * np.abs(RegECE_curve[:,0]- RegECE_curve[:,1] ) )
    
    return ENCE, RegECE_curve









def error_cdf(e_org, N=1000):
    
    assert(e_org.min()>=0)
    
    e= e_org.copy()
    
    e0, eN= e.min(), e.max()
    
    err_cdf= np.zeros((N,2))
    
    del_e= (eN- e0)/N
    
    for i in range(N):
        
        err_cdf[i,:]= i*del_e, np.mean(e<=i*del_e)
    
    return err_cdf





# def regression_calibration_curve(s_org, e_org, N=1000, rejection_percentile=99.5):
    
#     e_org= np.abs(e_org)
    
#     e= e_org.copy()
#     s= s_org.copy()
    
#     ind_s= s<np.percentile(s, rejection_percentile)
#     ind_e= e<np.percentile(e, rejection_percentile)
    
#     ind= np.logical_and(ind_s, ind_e)
    
#     e= e[ind]
#     s= s[ind]
    
#     s0, sN= 0, 1
    
#     RegECE_curve= np.zeros((N,3))
    
#     del_s= (sN- s0)/N
    
#     for i in range(N):
        
#         p0= s0 + i * del_s
#         p1= p0 + del_s
        
#         mask= np.logical_and(s>=p0, s<p1)
        
#         if mask.sum()>0:
            
#             mean_p= s[mask].mean()
#             # mean_p= np.mean( s<=p1)
#             # mean_q= np.mean(e<=p1)
#             mean_q= e[mask].mean()
#             mean_r= mask.mean()
            
#             RegECE_curve[i,:]= mean_p, mean_q, mean_r
    
#     return RegECE_curve






def estimate_RegROC_curve(e, s, max_X=1.0, DL= np.logspace(-5, 5, 1000)):
    
    PICP= np.zeros(DL.shape)
    MPIW= np.zeros(DL.shape)
    
    for idl in range(len(DL)):
        dl= DL[idl]
        yl= -dl*s
        yu= dl*s
        PICP[idl]= np.mean( np.logical_and(e>=yl, e<=yu) )
        MPIW[idl]= np.mean(yu-yl)
    
    assert(MPIW.max()>max_X)
    
    ind_temp= np.where(MPIW<max_X)[0]
    
    PICP= PICP[ind_temp]
    MPIW= MPIW[ind_temp]
    MPIW/= MPIW.max()
    
    return MPIW, PICP, metrics.auc(MPIW, PICP)
    
    




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



def seg_2_anulus(mask_orig, radius= 2.0):
    
    mask_copy= mask_orig.copy()
    
    size_x, size_y, size_z= mask_copy.shape
    mask= np.zeros((size_x+20, size_y+20, size_z+20))
    mask[10:10+size_x, 10:10+size_y, 10:10+size_z]= mask_copy
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    anulus= dist_image
    
    anulus= anulus[10:10+size_x, 10:10+size_y, 10:10+size_z]
    
    return anulus

















def register_jhu(my_t2, my_mk, jh_t2, jh_mk, jh_lb):
    
    my_t2_np= sitk.GetArrayFromImage( my_t2)
    my_mk_np= sitk.GetArrayFromImage( my_mk)
    
    my_t2_mk_np= my_t2_np * my_mk_np
    my_t2_mk= sitk.GetImageFromArray(my_t2_mk_np)
    
    my_t2_mk.SetDirection(my_mk.GetDirection())
    my_t2_mk.SetOrigin(my_mk.GetOrigin())
    my_t2_mk.SetSpacing(my_mk.GetSpacing())
    
    fixed_image= my_t2_mk
    
    jh_t2_np= sitk.GetArrayFromImage( jh_t2)
    jh_mk_np= sitk.GetArrayFromImage( jh_mk)
    
    jh_t2_mk_np= jh_t2_np * (jh_mk_np>200)
    jh_t2_mk= sitk.GetImageFromArray(jh_t2_mk_np)
    
    jh_t2_mk.SetDirection(jh_mk.GetDirection())
    jh_t2_mk.SetOrigin(jh_mk.GetOrigin())
    jh_t2_mk.SetSpacing(jh_mk.GetSpacing())
    
    moving_image= jh_t2_mk
    
    moving_image.SetDirection( fixed_image.GetDirection() )
    jh_lb.SetDirection( fixed_image.GetDirection() )
    
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(initial_transform)
    
    moving_image_2= resample.Execute(moving_image)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    
    final_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(final_transform)
    
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    
    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                sitk.Cast(moving_image_2, sitk.sitkFloat32))
    
    final_transform_v = sitk.Transform(final_transform)
    
    '''resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(final_transform_v)
    
    moving_image_5= resample.Execute(moving_image_2)
    sitk.WriteImage(moving_image_5 , reg_dir+'moving_image_reg.mhd')'''
    
    final_transform_v.AddTransform(initial_transform)
    
    '''resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetTransform(final_transform_v)
    
    moving_image_5= resample.Execute(jh_lb)
    sitk.WriteImage(moving_image_5 , reg_dir+'lb_image_reg.mhd')'''
        
    tx= initial_transform
    tx.AddTransform(final_transform)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetTransform(tx)
    
    out_image= resample.Execute(jh_lb)
    
    return out_image






























def save_all_data_thumbs( X_vol, Y_vol, save_dir= None):
    
    n_vol, SX, SY, SZ, n_channel= X_vol.shape
    n_vol, SX, SY, SZ, n_class  = Y_vol.shape
    
    n_rows, n_cols = 1, n_channel+1
    
    for i_vol in range(n_vol):
        
        fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
        i_fig= 0
        
        for i_channel in range(n_channel):
            
            i_fig+= 1
            plt.subplot(n_rows, n_cols, i_fig)
            
            plt.imshow( X_vol[i_vol, :, :, SZ//2 , i_channel], cmap='gray')
            plt.axis('off');
            
        seg= np.zeros( (SX, SY, SZ) )
        
        for i_seg in range(1,n_class):
            
            seg+= i_seg*Y_vol[i_vol,:,:,:,i_seg]
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        plt.imshow( seg[:, :, SZ//2], vmin=0, vmax= n_class)
        plt.axis('off');
        
        plt.tight_layout()
        
        fig.savefig(save_dir + 'thumb_data' + str(i_vol) + '.png')
        
        plt.close(fig)






def compute_AUC(D_te, D_od, p):
    
    fpr= np.zeros(len(p))
    tpr= np.zeros(len(p))
    
    for ip in range(len(p)):
        
        fpr[ip]= np.mean(D_te>p[ip])
        tpr[ip]= np.mean(D_od>p[ip])
    
    return metrics.auc(fpr, tpr), fpr, tpr
    















def save_img_and_seg_boundaries( my_list_orig, image_index=None, save_dir= None, markersize=1, colors=['.r', '.b']):
    
    n_rows, n_cols = 1, 3
    
    fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    i_fig= 0
    my_list= [None]*len(my_list_orig)
    
    for direction in ['Z', 'X', 'Y']:
        
        if direction=='X':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list_orig[i].copy() , [2,1,0] )
        elif direction=='Y':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list_orig[i].copy() , [0,2,1] )
        elif direction=='Z':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= my_list_orig[i].copy()
            pass
        
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        
        plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
        plt.axis('off');
        
        for i in range(1, len(my_list)):
            
            y_temp= my_list[i]
            
            b= dk_seg.seg_2_boundary_3d(y_temp)
            z= np.where(b>0)
            slc_all= z[slc_in].astype(np.int)
            slc_sel= slc_all==SZ
            x_sel= z[0][slc_sel].astype(np.int)
            y_sel= z[1][slc_sel].astype(np.int)
            plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'X_' + str(image_index) + '.png')
    
    plt.close(fig)






def save_pred_thumbs_gray(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 3, 3
    
    z = np.where(y_t_c > 0)
    SX = ( z[0].min() + z[0].max() ) // 2
    SY = ( z[1].min() + z[1].max() ) // 2
    SZ = ( z[2].min() + z[2].max() ) // 2
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ], perc_low) , np.percentile(x_c[:, :, SZ], perc_hi)
    plt.imshow(x_c[:, :, SZ], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 4)
    vmin, vmax= np.percentile(x_c[:, SY, :], perc_low) , np.percentile(x_c[:, SY, :], perc_hi)
    plt.imshow(x_c[:, SY, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 7)
    vmin, vmax= np.percentile(x_c[SX, :, :], perc_low) , np.percentile(x_c[SX, : , :], perc_hi)
    plt.imshow(x_c[SX , :, :], cmap='gray', vmin= vmin, vmax= vmax)
    '''plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')'''
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_t_c[:, SY, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX, :, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX, :, :], vmin=0, vmax= n_class-1, cmap='gray')
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)










def save_pred_thumbs_colored(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 3, 3
    
    z = np.where(y_t_c > 0)
    SX = ( z[0].min() + z[0].max() ) // 2
    SY = ( z[1].min() + z[1].max() ) // 2
    SZ = ( z[2].min() + z[2].max() ) // 2
    
    z= np.zeros(x_c.shape)
    z= y_t_c-y_a_c+2
    z[np.logical_and(y_t_c==0,y_a_c==0)]=0

    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[:, :, SZ], perc_low) , np.percentile(x_c[:, :, SZ], perc_hi)
    plt.imshow(x_c[:, :, SZ], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[:, SY, :], perc_low) , np.percentile(x_c[:, SY, :], perc_hi)
    plt.imshow(x_c[:, SY, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 7)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[SX, :, :], perc_low) , np.percentile(x_c[SX, : , :], perc_hi)
    plt.imshow(x_c[SX , :, :], cmap='gray', vmin= vmin, vmax= vmax)
    '''plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')'''
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(y_t_c[:, :, SZ], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 5)
    plt.axis('off')
    plt.imshow(y_t_c[:, SY, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX, :, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(z[:, :, SZ])
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(z[:, SY, :])
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(z[SX, :, :])
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)





def save_pred_thumbs_colored_uncert(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 3, 3
    
    z = np.where(y_t_c > 0)
    SX = ( z[0].min() + z[0].max() ) // 2
    SY = ( z[1].min() + z[1].max() ) // 2
    SZ = ( z[2].min() + z[2].max() ) // 2
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[:, :, SZ], perc_low) , np.percentile(x_c[:, :, SZ], perc_hi)
    plt.imshow(x_c[:, :, SZ], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 4)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[:, SY, :], perc_low) , np.percentile(x_c[:, SY, :], perc_hi)
    plt.imshow(x_c[:, SY, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 7)
    plt.axis('off')
    vmin, vmax= np.percentile(x_c[SX, :, :], perc_low) , np.percentile(x_c[SX, : , :], perc_hi)
    plt.imshow(x_c[SX , :, :], cmap='gray', vmin= vmin, vmax= vmax)
    '''plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')'''
    plt.subplot(n_rows, n_cols, 2)
    plt.axis('off')
    plt.imshow(y_t_c[:, :, SZ], vmin=0, vmax= n_class-1, cmap='gray')
    plt.subplot(n_rows, n_cols, 5)
    plt.axis('off')
    plt.imshow(y_t_c[:, SY, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX, :, :], vmin=0, vmax= n_class-1, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ], vmin=0, vmax= 1, cmap='OrRd')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY, :], vmin=0, vmax= 1, cmap='OrRd')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX, :, :], vmin=0, vmax= 1, cmap='OrRd')
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + 'uncert.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + 'uncert.png')
    plt.close(fig)







def save_multi_annot(x, y, image_index, thumbs_dir, perc_low= 5, perc_hi= 95):
    
    n_channel= x.shape[-1]
    n_class  = y.shape[-2]
    n_annot  = y.shape[-1]
    
    n_rows= n_class+1
    n_cols= max(n_annot, n_channel)
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    for i in range(n_cols):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off')
        if i<n_channel:
            vmin, vmax= np.percentile(x[:, :, i], perc_low) , np.percentile(x[:, :, i], perc_hi)
            plt.imshow(x[:, :, i], cmap='gray', vmin= vmin, vmax= vmax)
    
    for i in range(n_rows-1):
        for j in range(n_cols):
            plt.subplot(n_rows, n_cols, n_cols*(i+1)+j+1 )
            plt.axis('off')
            if j<n_annot:
                plt.imshow(y[:, :, i, j], vmin= 0, vmax= 1)
            
    plt.tight_layout()
    
    fig.savefig(thumbs_dir + 'X_' + str(image_index) + '.png')
    
    plt.close(fig)



def confusion_matrix(Y1, Y2):
    
    CM= np.zeros( ( Y1.shape[1], Y1.shape[1] ) )
    
    y1= np.argmax(Y1, axis=1)
    y2= np.argmax(Y2, axis=1)
    
    for i in range(Y1.shape[1] ):
        for j in range(Y1.shape[1] ):
            
            CM[i,j]= np.sum( np.logical_and( y1==i, y2==j ) )
            
    return CM













def compute_kappa(M):
    
    ''' function to compute Cohen's kappa (unweighted)
    see Wikipedia'''
    
    # to avoid division by zero
    M= M+1
    
    N= np.sum(M)
    n1= np.sum(M, axis=0)
    n2= np.sum(M, axis=1)
    
    pe= 0.0
    for i in range(M.shape[0]):
        pe+= n1[i]*n2[i]
    pe/= N**2
    
    po= np.sum(np.diag(M))/N
    
    kappa= 1 - (1-po) / (1-pe)
    
    return kappa


def Wilcoxon_Karimi(x1,x2, w_table):
    
    diff= np.abs(x2-x1)
    sign= np.sign(x2-x1)
    
    nz_ind= np.where(diff>0)[0]
    Nr= len(nz_ind)
    
    if Nr>5:
        
        diff= diff[nz_ind]
        sign= sign[nz_ind]
        
        arg_sort= np.argsort(diff)
        diff= diff[arg_sort]
        sign= sign[arg_sort]
        
        R= np.zeros(Nr)
        
        start= 0
        stop= 1
        
        while stop<Nr:
            
            if diff[stop] == diff[start] and stop<Nr-1:
                stop+= 1
            elif diff[stop] > diff[start]:
                R[start:stop]= (start+stop+1)/2
                start= stop
                stop= start+1
            elif stop==Nr-1:
                stop+= 1
                R[start:]= (start+stop+1)/2
        
        if R[-1]==0:
            R[-1]= Nr
        
        '''W= np.abs( np.sum( R*sign ) )
        significant= W>w_table[Nr,:]'''
        
        W= np.sum( R[sign>0]  )
        significant= np.zeros(w_table.shape[1])
        significant[0::2]= W<w_table[Nr,0::2]
        significant[1::2]= W>w_table[Nr,1::2]
    
    else:
        significant= np.zeros(w_table.shape[1])
    
    return significant



def trk_density_2_prob(density, q_percentile):
    
    assert(density.min()==0)
    
    if np.sum(density)==0:
        
        probability= np.zeros(density.shape)
    
    else:
        
        non_zeros= density[density>0]
        
        percntl = np.percentile(non_zeros, q_percentile)
        
        probability= density/percntl
        probability[density>=percntl]= 1
        
    return probability
    






def save_slice_and_seg_boundaries( img_np, seg_np, save_dir= None, name_prefix=' ', markersize=2, colors=['.r', '.b']):
    
    # n_img= np.sum( np.sum(seg_np, axis=0), axis=0).sum()
    # n_rows= int( np.ceil(np.sqrt(n_img)) )
    # n_cols= n_img//n_rows+1
    # fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    # slices= np.where( np.sum( np.sum(seg_np, axis=0), axis=0)>0 )
    
    n_img= img_np.shape[-1]
    if n_img>4:
        n_rows= int( np.floor(np.sqrt(n_img)) )-1
        n_cols= n_img//n_rows+1
    else:
        n_rows= 1
        n_cols= n_img
    fig, ax = plt.subplots(figsize=(32,16), nrows=n_rows, ncols=n_cols)
    
    # print(n_img)
    b= dk_seg.seg_2_boundary_3d(seg_np)
    z= np.where(b>0)
    
    for i in range(n_img):
        
        plt.subplot(n_rows, n_cols, i+1)
        
        plt.imshow( img_np[:, :, i ], cmap='gray')
        plt.axis('off');
        
        slc_all= z[2].astype(np.int)
        slc_sel= slc_all==i
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[0],  markersize=markersize)
    
    for i in range(n_img,n_rows*n_cols):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off');
    
    plt.tight_layout()
    
    fig.savefig(save_dir + name_prefix + '.png')
    
    plt.close(fig)





def save_slice_and_seg_boundaries_w_results( img_np, seg_np, y_res, DSC, save_dir= None, name_prefix=' ', markersize=1, colors=['.r', '.b']):
    
    # n_img= np.sum( np.sum(seg_np, axis=0), axis=0).sum()
    # n_rows= int( np.ceil(np.sqrt(n_img)) )
    # n_cols= n_img//n_rows+1
    # fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    # slices= np.where( np.sum( np.sum(seg_np, axis=0), axis=0)>0 )
    
    n_img= img_np.shape[-1]
    n_rows= int( np.ceil(np.sqrt(n_img)) )
    n_cols= n_img//n_rows+1
    fig, ax = plt.subplots(figsize=(24,12), nrows=n_rows, ncols=n_cols)
    
    # print(n_img)
    b= dk_seg.seg_2_boundary_3d(seg_np)
    z= np.where(b>0)
    
    b_res= dk_seg.seg_2_boundary_3d(y_res)
    z_res= np.where(b_res>0)
    
    for i in range(n_img):
        
        plt.subplot(n_rows, n_cols, i+1)
        
        plt.imshow( img_np[:, :, i ], cmap='gray')
        plt.axis('off');
        
        slc_all= z[2].astype(np.int)
        slc_sel= slc_all==i
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[0],  markersize=markersize)
        
        slc_all= z_res[2].astype(np.int)
        slc_sel= slc_all==i
        x_sel= z_res[0][slc_sel].astype(np.int)
        y_sel= z_res[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[1],  markersize=markersize)
        
        plt.title(str(round(DSC[i], 3)))
    
    for i in range(n_img,n_rows*n_cols):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off');
    
    plt.tight_layout()
    
    fig.savefig(save_dir + name_prefix + '.png')
    
    plt.close(fig)




def save_slice( img_np, save_dir= None, name_prefix=' '):
    
    n_img= img_np.shape[-1]
    
    if n_img>4:
        n_rows= int( np.floor(np.sqrt(n_img)) )-1
        n_cols= n_img//n_rows+1
    else:
        n_rows= 1
        n_cols= n_img
    
    fig, ax = plt.subplots(figsize=(24,12), nrows=n_rows, ncols=n_cols)
    
    for i in range(n_img):
        
        plt.subplot(n_rows, n_cols, i+1)
        
        plt.imshow( img_np[:, :, i ], cmap='gray', vmin= np.percentile(img_np[:, :, i ],10), vmax= np.percentile(img_np[:, :, i ],90))
        plt.axis('off');
        
        plt.title(str(i))
    
    for i in range(n_img,n_rows*n_cols):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off');
    
    plt.tight_layout()
    
    fig.savefig(save_dir + name_prefix + '.png')
    
    plt.close(fig)







def bland_altman_plot(data1, data2, markersize=2, outlier=10, x_lim=-1, y_lim=-1, std=1.96, first_method_is_reference=False):
    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    if first_method_is_reference:
        mean      = data1
    else:
        mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    good_ind= np.logical_and( diff< md + outlier*sd , diff> md - outlier*sd )
    
    data1= data1[good_ind]
    data2= data2[good_ind]
    
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    plt.figure()
    
    plt.scatter(mean, diff, s=markersize)
    plt.axhline(md,           color='gray', linestyle='-')
    plt.axhline(md + std*sd, color='gray', linestyle='--')
    plt.axhline(md - std*sd, color='gray', linestyle='--')
    
    if not x_lim==-1:
        plt.xlim([0,x_lim])
    if not y_lim==-1:
        plt.ylim([-y_lim,y_lim])
    
    # plt.title('Bland-Altman Plot')
    plt.show()
    
    











def unify_segmentation(seg_list, threshold=0.5):
    
    for i, seg in enumerate(seg_list):
        
        seg_img= nib.load(seg)
        seg_np= seg_img.get_fdata()
        
        if i==0:
            
            seg_affine= seg_img.affine
            seg_combined= np.zeros(seg_np.shape)
        
        seg_combined[seg_np>threshold]= i+1
    
    seg_combined = nib.Nifti1Image(seg_combined, seg_affine)
    
    return seg_combined
























def wasserstein_distance_dk(y_tr, y_prd):
    
    mask= np.logical_or(y_tr>0, y_prd>0)
    
    mask_x, mask_y, mask_z= np.where(mask>0)
    
    y_tr_nz= y_tr[mask>0]
    y_prd_nz= y_prd[mask>0]
    
    y_tr_cumsum= np.cumsum(y_tr_nz)
    y_prd_cumsum= np.cumsum(y_prd_nz)
    
    EM_dist= np.zeros(len(y_tr_cumsum))
    
    for i in range( len(y_tr_cumsum) ):
        
        ind_tr=  np.argmin(y_tr_cumsum>y_tr_cumsum[i])
        ind_prd= np.argmax(y_prd_cumsum>y_tr_cumsum[i])
        
        if i>0 and ind_prd==0:
            ind_prd= np.argmin(y_prd_cumsum<y_tr_cumsum[i])
            
        loc_tr=  [mask_x[ind_tr], mask_y[ind_tr], mask_z[ind_tr]]
        loc_prd= [mask_x[ind_prd], mask_y[ind_prd], mask_z[ind_prd]]
        
        EM_dist[i]= np.linalg.norm( np.array(loc_tr) - np.array(loc_prd) ) * y_tr_nz[i]
    
    # return EM_dist, EM_dist.sum()/y_tr_cumsum[-1]
    return EM_dist.sum() / y_tr.sum()




def wasserstein_distance_normalize_dk(y_tr, y_prd):
    
    mask= np.logical_or(y_tr>0, y_prd>0)
    
    mask_x, mask_y, mask_z= np.where(mask>0)
    
    y_tr_nz= y_tr[mask>0]
    y_prd_nz= y_prd[mask>0]
    
    y_tr_nz/=  y_tr_nz.sum()
    y_prd_nz/= y_prd_nz.sum()
    
    y_tr_cumsum= np.cumsum(y_tr_nz)
    y_prd_cumsum= np.cumsum(y_prd_nz)
    
    EM_dist= np.zeros(len(y_tr_cumsum))
    
    for i in range( len(y_tr_cumsum) ):
        
        ind_tr=  np.argmax(y_tr_cumsum>y_tr_cumsum[i])
        ind_prd= np.argmax(y_prd_cumsum>y_tr_cumsum[i])
        
        loc_tr=  [mask_x[ind_tr], mask_y[ind_tr], mask_z[ind_tr]]
        loc_prd= [mask_x[ind_prd], mask_y[ind_prd], mask_z[ind_prd]]
        
        EM_dist[i]= np.linalg.norm( np.array(loc_tr) - np.array(loc_prd) ) * y_tr_nz[i]
    
    # return EM_dist, EM_dist.sum()/y_tr_cumsum[-1]
    return EM_dist.sum() / y_tr.sum()





def wasserstein_distance_1d_dk(y_tr, y_prd, normalize=False):
    
    sx, sy, sz= y_tr.shape
    
    assert y_tr.shape==y_prd.shape
    assert y_tr.min()>=0
    assert y_prd.min()>=0
    
    res= np.zeros(sx*sy+sx*sz+sx*sy)
    i_res= -1
    
    for ix in range(sx):
        for iy in range(sy):
            p= y_tr[ix,iy,:]
            q= y_prd[ix,iy,:]
            if normalize:
                if p.max()>0 and q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p/p.sum(),q/q.sum())
            else:
                if p.max()>0 or q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p,q)
                    
    
    for ix in range(sx):
        for iz in range(sz):
            p= y_tr[ix,:,iz]
            q= y_prd[ix,:,iz]
            if normalize:
                if p.max()>0 and q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p/p.sum(),q/q.sum())
            else:
                if p.max()>0 or q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p,q)
    
    for iy in range(sy):
        for iz in range(sz):
            p= y_tr[:,iy,iz]
            q= y_prd[:,iy,iz]
            if normalize:
                if p.max()>0 and q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p/p.sum(),q/q.sum())
            else:
                if p.max()>0 or q.max()>0:
                    i_res+= 1
                    res[i_res]= wasserstein_distance(p,q)
    
    return res[:i_res+1].mean()














def save_tractseg_results(x, yt, yp, slx, sly, slz, perc_low= 5, perc_hi= 95, markersize=1, save_address=None):
    
    n_rows, n_cols = 1, 3
    
    SX, SY, SZ = x.shape
    
    fig, ax = plt.subplots(figsize=(16, 8), nrows=n_rows, ncols=n_cols)
    
    slc_in= 2
    b= dk_seg.seg_2_boundary_3d(yt)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slz
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(yp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slz
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[1][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 1)
    
    vmin, vmax= np.percentile(x[:, :, slz], perc_low) , np.percentile(x[:, :, slz], perc_hi)
    
    plt.imshow(x[:,:,slz], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    plt.axis('off')
    
    slc_in= 1
    b= dk_seg.seg_2_boundary_3d(yt)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==sly
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[2][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(yp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==sly
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[2][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 2)
    
    vmin, vmax= np.percentile(x[:, sly, :], perc_low) , np.percentile(x[:, sly, :], perc_hi)
    
    plt.imshow(x[:,sly,:], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    plt.axis('off')
    
    slc_in= 0
    b= dk_seg.seg_2_boundary_3d(yt)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slx
    x_sel= z[1][slc_sel].astype(np.int)
    y_sel= z[2][slc_sel].astype(np.int)
    
    b= dk_seg.seg_2_boundary_3d(yp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slx
    x_sel2= z[1][slc_sel].astype(np.int)
    y_sel2= z[2][slc_sel].astype(np.int)
    
    plt.subplot(n_rows, n_cols, 3)
    
    vmin, vmax= np.percentile(x[slx, :, :], perc_low) , np.percentile(x[slx, : , :], perc_hi)
    
    plt.imshow(x[slx,:,:], cmap='gray', vmin= vmin, vmax= vmax)
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    plt.axis('off')
    
    plt.tight_layout()
    
    fig.savefig(save_address)
    plt.close(fig)
    
    
    




































