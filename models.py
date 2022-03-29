# -*- coding: utf-8 -*-
"""

Models for segmentation

@author: davoo
"""


import numpy as np
import tensorflow as tf




def my_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001):
    
    feat_fine = [None] * (depth - 1)
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        
        out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        
        # if level == 3:
        #     out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        # else:
        #     out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
           
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
           
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            inp = inp + inp_1 + inp_0  ###
            
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''
        
    return output, output_s
































def cost_dice_forground(Y, predicter, loss_type= 'sorensen', smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    inse = tf.reduce_sum(output * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)
    
    cost= 1.0 - dice
    
    return cost





def cost_dice_forground_2D(Y, predicter, loss_type= 'sorensen', smooth = 1e-5):
    
    target = Y[:, :, :, 1]
    output = predicter[:, :, :, 1]
    inse = tf.reduce_sum(output * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)
    
    cost= 1.0 - dice
    
    return cost



def cost_dice_2D(Y, predicter, loss_type= 'sorensen', smooth = 1e-5):
    
    target = Y
    output = predicter
    inse = tf.reduce_sum(output * target)
    
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
        
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)

    cost= 1.0 - dice
    
    return cost






def cost_dice_multi_forground(Y, predicter, n_foreground=2, loss_type='sorensen', smooth=1e-5):
    
    cost= n_foreground
    
    for i_f in range(n_foreground):
        
        target = Y[:, :, :, :, i_f+1]
        output = predicter[:, :, :, :, i_f+1]
        inse = tf.reduce_sum(output * target)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output)
            r = tf.reduce_sum(target * target)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output)
            r = tf.reduce_sum(target)
        dice = (2. * inse + smooth) / (l + r + smooth)
        # dice_whole = tf.reduce_mean(dice)
        
        cost -= dice
        
    return cost








def cost_dice_multi_forground_log(Y, predicter, n_foreground=2, loss_type='sorensen', smooth=1e-3, log_smooth=1e-3):
    
    cost= 0
    
    for i_f in range(n_foreground):
        
        target = Y[:, :, :, :, i_f+1]
        output = predicter[:, :, :, :, i_f+1]
        inse = tf.reduce_sum(output * target)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output)
            r = tf.reduce_sum(target * target)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output)
            r = tf.reduce_sum(target)
        dice = (2. * inse + smooth) / (l + r + smooth)
        # dice_whole = tf.reduce_mean(dice)
        
        cost -= tf.log(dice + log_smooth)
        
    return cost











def cost_dice_full(Y, predicter, n_class= 2, loss_type='sorensen', smooth = 1e-5):
    
    cost = n_class
    
    for dice_channel in range(n_class):
        
        target = Y[:, :, :, :, dice_channel]
        output = predicter[:, :, :, :, dice_channel]
        inse = tf.reduce_sum(output * target )
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output)
            r = tf.reduce_sum(target * target)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output)
            r = tf.reduce_sum(target)
        dice = (2. * inse + smooth) / (l + r + smooth)
        # dice_whole = tf.reduce_mean(dice, name='dice_coe')
        
        cost -= dice
    
    return cost





def cost_x_entropy(Y, logit_f):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_f, labels=Y))
    
    return cost










def cost_x_entropy_noisy(Y, logit_f, M, T, LX, LY, LZ, n_class):
    
    cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=logit_f, labels=Y)
    cost1 = tf.multiply( cost1, (M-1) )
    cost1 = tf.reduce_mean( cost1 )
    
    cost2 = tf.nn.softmax_cross_entropy_with_logits(logits=logit_f, labels=Y)
    cost2 = tf.reshape(cost2, [-1, LX*LY*LZ, n_class ])
    cost2 = tf.matmul( cost2, T )
    cost2 = tf.multiply( cost2, M )
    cost2 = tf.reduce_mean( cost2 )
    
    cost = cost1 + cost2
    
    return cost













def cost_uncertainty_aleatoric(Y, logit_f, logit_s, EPS, T, loss_type='sorensen', smooth = 1e-5):
    
    cost= 0
    for iT in range(T):
        
       Eps= EPS[:,:,:,:,:,iT]
       #logit_f_n  = logit_f + Eps * tf.exp( logit_s )
       logit_f_n  = logit_f + Eps * tf.exp( tf.tile(logit_s, [1,1,1,1,2]) )
       logit_f_n_softmax= tf.nn.softmax(logit_f_n)
       
       target = Y[:, :, :, :, 1]
       output = logit_f_n_softmax[:, :, :, :, 1] * tf.exp( - logit_s[:, :, :, :, 0] )
       inse = tf.reduce_sum(output * target)
       if loss_type == 'jaccard':
           l = tf.reduce_sum(output * output)
           r = tf.reduce_sum(target * target)
       elif loss_type == 'sorensen':
           l = tf.reduce_sum(output)
           r = tf.reduce_sum(target)
       dice = (2. * inse + smooth) / (l + r + smooth)
       #dice_whole = tf.reduce_mean(dice)
       
       cost+= - dice + tf.reduce_mean( logit_s )
       
    return cost
  
    
    
def cost_uncertainty_epistemic(Y, predicter, logit_s, loss_type='sorensen', smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1] * tf.exp( - logit_s[:, :, :, :, 0] )
    inse = tf.reduce_sum(output * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)
    
    cost= - dice + tf.reduce_mean( logit_s )
    
    return cost




def cost_tvesrky(Y, predicter, smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    inse = tf.reduce_sum(output * target, )
    diff1= tf.reduce_sum((1-output) * target)
    diff2= tf.reduce_sum(output * (1-target))
    cost = 1- inse/(inse+0.6*diff1+0.4*diff2+smooth)
    
    return cost




def cost_mae(Y, predicter):
    
    target = Y[:, :, :, :, :]
    output = predicter[:, :, :, :, :]
    cost = tf.reduce_mean( tf.abs( output - target ) )
    
    return cost



def cost_lq(Y, predicter, q=0.7):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    cost = tf.reduce_sum( tf.multiply(target, 1- tf.pow( tf.multiply(target,output) , q ) ) / q )
    
    return cost






def my_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    for i in range(len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    return inp






def my_reg_net_SH(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    for i in range(len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        if i<len(n_feat_vec)-2:
            inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
            inp = tf.nn.dropout(inp, p_keep_hidden)
        else:
            inp = tf.matmul(inp, W_fc) + b_fc
            inp = tf.nn.dropout(inp, p_keep_hidden)
    
    return inp






def my_reg_net_nonReLU(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    for i in range(len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        if i<len(n_feat_vec)-2:
            inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        else:
            inp = tf.matmul(inp, W_fc) + b_fc
        
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    return inp




def my_2Dreg_net_nonReLU(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    n_feat_in= n_feat_vec[1]
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp = tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-2):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    n_feat_in=  n_feat_vec[-2]
    n_feat_out= n_feat_vec[-1]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_out')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_out')
    out = tf.matmul(inp, W_fc) + b_fc
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_unc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_unc')
    unc = tf.matmul(inp, W_fc) + b_fc
    
    return out, unc







def my_2Dreg_net_nonReLU_2branch(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    ###########################################################################
    inp = X
    
    n_feat_in= n_feat_vec[1]
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp = tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-2):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    n_feat_in=  n_feat_vec[-2]
    n_feat_out= n_feat_vec[-1]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_out')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_out')
    out = tf.matmul(inp, W_fc) + b_fc
    
    ###########################################################################
    inp = X
    
    n_feat_in= n_feat_vec[1]
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp = tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-2):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu_'+str(i))
        
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    n_feat_in=  n_feat_vec[-2]
    n_feat_out= n_feat_vec[-1]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_unc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_unc')
    unc = tf.matmul(inp, W_fc) + b_fc
    
    return out, unc
















def my_2Dreg_net_nonReLU_2branch_datapred(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    ###########################################################################
    inp = X
    
    n_feat_in= n_feat_vec[1]
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp = tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-2):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    n_feat_in=  n_feat_vec[-2]
    n_feat_out= n_feat_vec[-1]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_out')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_out')
    out = tf.matmul(inp, W_fc) + b_fc
    
    ###########################################################################
    inp = X[:,:,::2]
    
    n_feat_in= n_feat_vec[1]//2
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp = tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-2):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu_'+str(i))
        
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    inp_out= tf.concat((inp, out), axis= -1)
    
    n_feat_in=  n_feat_vec[-2]+1
    n_feat_out= n_feat_vec[1]//2*n_feat_vec[0]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_unc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_unc')
    sig_pred = tf.matmul(inp_out, W_fc) + b_fc
    
    sig_pred= tf.reshape(sig_pred, [-1,n_feat_vec[0], n_feat_vec[1]//2])
    
    return out, sig_pred



def my_2Dreg_net_ReLU(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    n_feat_in= n_feat_vec[1]
    n_feat_out= n_feat_vec[2]
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    inp= tf.reshape(inp, [-1, n_feat_vec[0]*n_feat_vec[2]])
    
    for i in range(3, len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        
        if i<len(n_feat_vec)-2:
            inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
            inp = tf.nn.dropout(inp, p_keep_hidden)
        else:
            inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    return inp























def my_RL(X, Y, n_RL, n_sig, n_fod):
    
    Y_est = Y
   
    H = tf.Variable(tf.truncated_normal([n_sig, n_fod], stddev= np.sqrt(2.0/n_sig)), name='H')
    
    for i in range(n_RL):
        
        numer = tf.matmul( tf.transpose(H), tf.transpose(X))
        denom = tf.matmul( tf.matmul( tf.transpose(H), H), tf.transpose(Y_est) )
        
        Y_est= tf.math.multiply( Y_est, tf.math.divide( numer, denom ) )
    
    return Y_est


















def my_net_return_fmaps(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth )
    
    # if write_summaries:
    #     tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            # name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            # name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                # name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    # name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            # name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            # name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            # name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            # name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if True: #level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    feat_cors = [None] * (depth - 1)
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        # name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        # if write_summaries:
        #     tf.summary.histogram(name_w, W_deconv)
        #     tf.summary.histogram(name_b, b_deconv)
        #     tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        # name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        # if write_summaries:
        #     tf.summary.histogram(name_w, W_1)
        #     tf.summary.histogram(name_b, b_1)
        #     tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            # name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            # name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            # name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            # name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
        
        feat_cors[level] = inp
        
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''

    return output, output_s, feat_fine, feat_cors






























def my_net_frcnn(X, ks_0, depth, n_feat_0, n_channel, n_class, n_anchor, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth - 1)
    
    # if write_summaries:
    #     tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            # name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            # name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                # name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    # name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            # name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
                # tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            # name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            # name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            # name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        # name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        # if write_summaries:
        #     tf.summary.histogram(name_w, W_deconv)
        #     tf.summary.histogram(name_b, b_deconv)
        #     tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        # name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        # if write_summaries:
        #     tf.summary.histogram(name_w, W_1)
        #     tf.summary.histogram(name_b, b_1)
        #     tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            # name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            # name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            # name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_2)
            #     tf.summary.histogram(name_b, b_2)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            # name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            # if write_summaries:
            #     tf.summary.histogram(name_w, W_3)
            #     tf.summary.histogram(name_b, b_3)
            #     tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
    
    ks= 1
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_prob'
    name_b = 'b_prob'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class*n_anchor], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class*n_anchor]), name=name_b)
    prob = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_reg'
    name_b = 'b_reg'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 6*n_anchor], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[6*n_anchor]), name=name_b)
    reg = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
        
    return prob, reg










def u_net(X, depth, n_feat_0, n_channel, n_class, bias_init=0.001):
    
    
    feat_fine= [None]*(depth-1)
    
    # Convolution Layers
    
    inp= X
    
    for level in range(depth):
        
        inp_0= inp
        
        strd= 1
        ks= 5
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==0:
                    n_f_in=  n_channel
                else:
                    n_f_in= n_feat_0 * 2**level
            else:
                n_f_in= n_feat_0 * 2**level
            
            n_f_out= n_feat_0 * 2**level
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv) + '_down'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp + inp_0
        
        if level<depth-1:
            
            feat_fine[level]= inp
            
            strd= 2
            ks= 2
            
            n_f_in = n_feat_0 * 2**level
            n_f_out= n_feat_0 * 2**(level+1)
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_down'
            name_b= 'b_'+ str(level) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        
    
    # DeConvolution Layers
    
    for level in range(depth-2,-1,-1):
        
        ks= 2
        strd= 2
        
        if level==depth-2:
            n_f_out= n_feat_0 * 2**level
            n_f_in = n_feat_0 * 2**(level+1)
        else:
            n_f_out= n_feat_0 * 2**(level+1)
            n_f_in = n_feat_0 * 2**(level+2)
        
        n_l= n_f_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(level) + '_up'
        name_b= 'b_'+ str(level) + '_up'
        W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_out,n_f_in], stddev=s_dev), name=name_w)
        b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, in_shape[4]//2])
        Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
        Deconv= tf.nn.relu(tf.add(Deconv, b_deconv))
        
        inp_0= Deconv
        
        if level==depth-2:
            inp_0= tf.tile(inp_0, [1,1,1,1,2])
        
        inp= tf.concat([feat_fine[level], Deconv], 4)
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==depth-2:
                    n_f_in= n_feat_0 * 2**(level+1)
                else:
                    n_f_in= n_feat_0 * 2**(level+1) * 3 / 2
            else:
                n_f_in= n_feat_0 * 2**(level+1)
            
            n_f_out= n_feat_0 * 2**(level+1)
            
            strd= 1
            ks= 5
            
            n_l= n_channel*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv)  + '_up'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv)  + '_up'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,int(n_f_in),int(n_f_out)], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp+ inp_0
    
    n_f_in = n_feat_0 * 2**(level+1)
    
    n_l= n_f_in*ks**3
    s_dev= np.sqrt(2.0/n_l)
    name_w= 'W_out'
    name_b= 'b_out'
    W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_class], stddev=s_dev), name=name_w)
    b_1= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
    
    return output













def v_net(X, depth, n_feat_0, n_channel, n_class, bias_init=0.001):
    
    
    feat_fine= [None]*(depth-1)
    
    # Convolution Layers
    
    inp= X
    
    for level in range(depth):
        
        inp_0= inp
        
        strd= 1
        ks= 5
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==0:
                    n_f_in=  n_channel
                else:
                    n_f_in= n_feat_0 * 2**level
            else:
                n_f_in= n_feat_0 * 2**level
            
            n_f_out= n_feat_0 * 2**level
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv) + '_down'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp + inp_0
        
        if level<depth-1:
            
            feat_fine[level]= inp
            
            strd= 2
            ks= 2
            
            n_f_in = n_feat_0 * 2**level
            n_f_out= n_feat_0 * 2**(level+1)
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_down'
            name_b= 'b_'+ str(level) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        
    
    # DeConvolution Layers
    
    for level in range(depth-2,-1,-1):
        
        ks= 2
        strd= 2
        
        if level==depth-2:
            n_f_out= n_feat_0 * 2**level
            n_f_in = n_feat_0 * 2**(level+1)
        else:
            n_f_out= n_feat_0 * 2**(level+1)
            n_f_in = n_feat_0 * 2**(level+2)
        
        n_l= n_f_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(level) + '_up'
        name_b= 'b_'+ str(level) + '_up'
        W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_out,n_f_in], stddev=s_dev), name=name_w)
        b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, in_shape[4]//2])
        Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
        Deconv= tf.nn.relu(tf.add(Deconv, b_deconv))
        
        inp_0= Deconv
        
        if level==depth-2:
            inp_0= tf.tile(inp_0, [1,1,1,1,2])
        
        inp= tf.concat([feat_fine[level], Deconv], 4)
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==depth-2:
                    n_f_in= n_feat_0 * 2**(level+1)
                else:
                    n_f_in= n_feat_0 * 2**(level+1) * 3 / 2
            else:
                n_f_in= n_feat_0 * 2**(level+1)
            
            n_f_out= n_feat_0 * 2**(level+1)
            
            strd= 1
            ks= 5
            
            n_l= n_channel*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv)  + '_up'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv)  + '_up'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,int(n_f_in),int(n_f_out)], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp+ inp_0
    
    n_f_in = n_feat_0 * 2**(level+1)
    
    n_l= n_f_in*ks**3
    s_dev= np.sqrt(2.0/n_l)
    name_w= 'W_out'
    name_b= 'b_out'
    W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_class], stddev=s_dev), name=name_w)
    b_1= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
    
    return output










    






def v_net_return_fmaps(X, depth, n_feat_0, n_channel, n_class, bias_init=0.001):
    
    
    feat_fine= [None]*(depth-1)
    
    # Convolution Layers
    
    inp= X
    
    for level in range(depth):
        
        inp_0= inp
        
        strd= 1
        ks= 5
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==0:
                    n_f_in=  n_channel
                else:
                    n_f_in= n_feat_0 * 2**level
            else:
                n_f_in= n_feat_0 * 2**level
            
            n_f_out= n_feat_0 * 2**level
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv) + '_down'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp + inp_0
        
        if level<depth-1:
            
            feat_fine[level]= inp
            
            strd= 2
            ks= 2
            
            n_f_in = n_feat_0 * 2**level
            n_f_out= n_feat_0 * 2**(level+1)
            
            n_l= n_f_in*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_down'
            name_b= 'b_'+ str(level) + '_down'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_f_out], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        
    
    # DeConvolution Layers
    
    feat_cors = [None] * (depth - 1)
    
    for level in range(depth-2,-1,-1):
        
        ks= 2
        strd= 2
        
        if level==depth-2:
            n_f_out= n_feat_0 * 2**level
            n_f_in = n_feat_0 * 2**(level+1)
        else:
            n_f_out= n_feat_0 * 2**(level+1)
            n_f_in = n_feat_0 * 2**(level+2)
        
        n_l= n_f_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(level) + '_up'
        name_b= 'b_'+ str(level) + '_up'
        W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_out,n_f_in], stddev=s_dev), name=name_w)
        b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, in_shape[4]//2])
        Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
        Deconv= tf.nn.relu(tf.add(Deconv, b_deconv))
        
        inp_0= Deconv
        
        if level==depth-2:
            inp_0= tf.tile(inp_0, [1,1,1,1,2])
        
        inp= tf.concat([feat_fine[level], Deconv], 4)
        
        if level==0:
            n_conv= 1
        elif level==1:
            n_conv= 2
        else:
            n_conv= 3
        
        for i_conv in range(n_conv):
            
            if i_conv==0:
                if level==depth-2:
                    n_f_in= n_feat_0 * 2**(level+1)
                else:
                    n_f_in= n_feat_0 * 2**(level+1) * 3 / 2
            else:
                n_f_in= n_feat_0 * 2**(level+1)
            
            n_f_out= n_feat_0 * 2**(level+1)
            
            strd= 1
            ks= 5
            
            n_l= n_channel*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level)+ '_' + str(i_conv)  + '_up'
            name_b= 'b_'+ str(level)+ '_' + str(i_conv)  + '_up'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,int(n_f_in),int(n_f_out)], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_f_out]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        inp= inp+ inp_0
        
        feat_cors[level] = inp
    
    n_f_in = n_feat_0 * 2**(level+1)
    
    n_l= n_f_in*ks**3
    s_dev= np.sqrt(2.0/n_l)
    name_w= 'W_out'
    name_b= 'b_out'
    W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_f_in,n_class], stddev=s_dev), name=name_w)
    b_1= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
    
    return output, feat_fine, feat_cors








    











def DCI_CNN(X, n_feat_vec, ks_vec, strd_vec, n_channel, n_class, p_keep_conv, bias_init=0.001):
    
    inp= X
    
    for level in range(len(n_feat_vec)-1):
        
        n_feat_in  = n_feat_vec[level]
        n_feat_out = n_feat_vec[level+1]
        ks = ks_vec[level]
        strd = strd_vec[level]
        
        n_l = n_feat_in * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_init'
        name_b = 'b_' + str(level) + '_init'
        
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
        #inp = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1)
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        
    return inp










def DCI_RESCNN(X, n_feat_vec, n_res_vec, ks_vec, strd_vec, n_channel, n_class, p_keep_conv, bias_init=0.0):
    
    inp= X
    
    for level in range(len(n_feat_vec)-1):
        
        n_feat_in  = n_feat_vec[level]
        n_feat_out = n_feat_vec[level+1]
        ks = ks_vec[level]
        strd = strd_vec[level]
        n_res  = n_res_vec[level]
        
        n_l = n_feat_in * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_init'
        name_b = 'b_' + str(level) + '_init'
        
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        inp_res= inp
        
        for i_res in range(n_res):
            
            n_feat_in  = n_feat_vec[level+1]
            n_feat_out = n_feat_vec[level+1]
            ks = ks_vec[level]
            strd = 1
            
            n_l = n_feat_in * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_' + str(i_res)
            name_b = 'b_' + str(level) + '_' + str(i_res)
            
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
            
            inp_res = tf.nn.relu(tf.add(tf.nn.conv3d(inp_res, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp_res= tf.nn.dropout(inp_res, p_keep_conv)
            
        inp= inp+ inp_res
    
    return inp






def ViT_seg(X, T, n_patch, d_patch, d_emb, ViT_depth, n_head, d_emb_h, n_class=2, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  d_patch
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        if use_token:
            pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        else:
            pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
    
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        # out = tf.reshape(inp, [-1, n_patch*d_emb])
        
        # n_feat_in=  n_patch*d_emb
        # n_feat_out= n_patch*d_emb//10
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        # out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        
        # n_feat_in=  n_patch*d_emb//10
        # n_feat_out= n_class
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        # out = tf.matmul(out, W_oout) + b_out
        
        out= inp
        
        n_feat_in=  d_emb
        n_feat_out= d_patch
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        out = tf.transpose(out, perm=[0,2,1])
        
        n_feat_in=  n_patch
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        out = tf.matmul(out, W_oout) + b_out
        
    
    return out, A_list, pos_emb









def ViT_seg_full_patch(X, T, n_patch, d_patch, d_emb, ViT_depth, n_head, d_emb_h, n_class=2, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  d_patch
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        if use_token:
            pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        else:
            pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
        
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        n_feat_in=  d_emb
        n_feat_out= d_patch
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_0' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_0' )
        
        out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
        
        
        out = tf.reshape(out, tf.stack((-1, d_patch*n_patch, 1)))
                
        n_feat_in=  1
        n_feat_out= 10
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_1' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_1' )
        
        out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        
        n_feat_in=  10
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_2' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_2' )
        
        out = tf.matmul(out, W_oout) + b_out
        
        d_out= int( round( n_patch**(1/3) * d_patch**(1/3)) )
        
        out = tf.reshape(out, tf.stack((-1, d_out, d_out, d_out, n_class)))
        
    
    return out#, A_list












def ViT_det_full_patch(X, T, pos_emb, n_patch, d_patch, d_patch_2, d_emb, ViT_depth, n_head, d_emb_h, n_class=2, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  d_patch
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        # if use_token:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        # else:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
        
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        n_feat_in=  d_emb
        n_feat_out= d_patch_2
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_0' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_0' )
        
        out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
        
        out = tf.reshape(out, tf.stack((-1, d_patch_2*n_patch, 1)))
                
        n_feat_in=  1
        n_feat_out= 10
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_1' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_1' )
        
        out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        
        n_feat_in=  10
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_2' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_2' )
        
        out = tf.matmul(out, W_oout) + b_out
        
        d_out= int( round( n_patch**(1/3) * d_patch_2**(1/3)) )
        
        out = tf.reshape(out, tf.stack((-1, d_out, d_out, d_out, n_class)))
        
    
    return out, A_list




















def ViT_FCN_seg(X, T, n_feat_vec, str_vec, n_patch, d_patch, d_emb, L_patch, ViT_depth, n_head, d_emb_h, 
                n_class=2, use_token=False, residuals=True, layer_norm=True, p_keep=1.0, w_std= 0.002, bias_init= 0.0):
    
    
    print(X)
    
    # inp, _ = my_net(X, 3, 3, 28, 1, n_feat_vec[1], p_keep, bias_init=0.001)
    
    inp = X
    
    print('inp ', inp)
    
    for i in range(1,len(n_feat_vec)-1):
        
        n_feat_in=  n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        ks= 3
        strd= str_vec[i]
        
        n_l= n_feat_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i)
        name_b= 'b_'+ str(i)
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        if i<len(n_feat_vec)-2:
            new_inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        else:
            new_inp =            tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1)
        
        if n_feat_in==n_feat_out:
            inp= inp + new_inp
        else:
            inp= new_inp
        
        print(inp)
     
    
    inp = tf.transpose(inp, perm=[0,4,1,2,3])
    inp = tf.reshape(inp, [-1, 512, 27])
    inp = tf.transpose(inp, perm=[0,2,1])
    
    
    
    
    n_feat_in=  d_patch
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
    
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        # out = tf.reshape(inp, [-1, n_patch*d_emb])
        
        # n_feat_in=  n_patch*d_emb
        # n_feat_out= n_patch*d_emb//10
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        # out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        
        # n_feat_in=  n_patch*d_emb//10
        # n_feat_out= n_class
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        # out = tf.matmul(out, W_oout) + b_out
        
        out= inp
        
        n_feat_in=  d_emb
        n_feat_out= d_patch
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        out = tf.transpose(out, perm=[0,2,1])
        
        n_feat_in=  n_patch
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        out = tf.matmul(out, W_oout) + b_out
        
    
    return out




















def ViT_MLP_seg(X, T, n_patch, d_patch, d_emb, ViT_depth, n_head, d_emb_h, n_class=2, use_token=False, 
            residuals=True, layer_norm=True, p_keep=1.0, w_std= 0.002):
    
    
    inp = X
    
    n_feat_in=  d_patch
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    
    for i_depth in range(ViT_depth):
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
    
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        # out = tf.reshape(inp, [-1, n_patch*d_emb])
        
        # n_feat_in=  n_patch*d_emb
        # n_feat_out= n_patch*d_emb//10
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        # out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        
        # n_feat_in=  n_patch*d_emb//10
        # n_feat_out= n_class
        
        # W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        # b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        # out = tf.matmul(out, W_oout) + b_out
        
        out= inp
        
        n_feat_in=  d_emb
        n_feat_out= d_patch
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_1' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_1' )
        
        out = tf.nn.relu( tf.matmul(out, W_oout) + b_out )
        out = tf.transpose(out, perm=[0,2,1])
        
        n_feat_in=  n_patch
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_2' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_2' )
        
        out = tf.matmul(out, W_oout) + b_out
        
    
    return out





def pos_embed_fixed(n_patch, d_emb, C=10**4):
    
    pos_emb= np.zeros( (n_patch, d_emb) )
    
    for i in range(n_patch):
        for j in range(d_emb):
            
            if i%2==0:
                pos_emb[i,j]= np.sin( i/ ( C**( (j//2)/d_emb ) ) )
            else:
                pos_emb[i,j]= np.cos( i/ ( C**( (j//2)/d_emb ) ) )
   
    return pos_emb









def DTI_Transformer(X, T, pos_emb, n_inp, d_emb, ViT_depth, n_head, d_emb_h, n_channel, L, n_class=6, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  n_channel
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    inp = tf.nn.dropout(inp, p_keep)
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        # if use_token:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        # else:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            Q = tf.nn.dropout(Q, p_keep)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            K = tf.nn.dropout(K, p_keep)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            V = tf.nn.dropout(V, p_keep)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        new_inp = tf.nn.dropout(new_inp, p_keep)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
        
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_0' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_0' )
        
        # out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
        out = tf.matmul(inp, W_oout) + b_out
        
        out = tf.reshape(out, tf.stack((-1, L, L, L, n_class)))
        
    
    return out, A_list

















def DTI_Transformer_plusplus(X, T, pos_emb, Spos_emb, Dpos_emb, n_inp, d_emb, ViT_depth, n_head, d_emb_h, n_channel, L, n_class=6, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  n_channel
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    inp = tf.nn.dropout(inp, p_keep)
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        # if use_token:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        # else:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            Q = tf.nn.dropout(Q, p_keep)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            K = tf.nn.dropout(K, p_keep)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            V = tf.nn.dropout(V, p_keep)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        new_inp = tf.nn.dropout(new_inp, p_keep)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
        
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_0' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_0' )
        
        # out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
        out = tf.matmul(inp, W_oout) + b_out
        
        out = tf.reshape(out, tf.stack((-1, L, L, L, n_class)))
        
    
    
    
    
    ####
    
    S= X
    
    n_feat_in=  6
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='SW_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='Sb_fc')
    
    S = tf.nn.relu( tf.matmul(S, W_fc) + b_fc )
    S = tf.nn.dropout(S, p_keep)
    
    if use_token:
        S= tf.concat((T, S), axis=1)
    
    if learn_embd:
        S= S + Spos_emb
   
    for i_depth in range(2):
        
        if layer_norm:
            S= tf.contrib.layers.layer_norm(S)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='SW_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(S, W_q)
            Q = tf.nn.dropout(Q, p_keep)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='SW_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(S, W_k)
            K = tf.nn.dropout(K, p_keep)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='SW_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(S, W_v)
            V = tf.nn.dropout(V, p_keep)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='SA_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            # A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                new_inp= SA
            else:
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            S= S + new_inp
        else:
            S= new_inp
                    
        if layer_norm:
            S= tf.contrib.layers.layer_norm(S)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='SW_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='Sb_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(S, W_l) + b_l )
        new_inp = tf.nn.dropout(new_inp, p_keep)
        
        if residuals:
            S= S + new_inp
        else:
            S= new_inp
                
    
    
    ####
    
    D= out
    
    n_feat_in=  6
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='DW_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='Db_fc')
    
    D = tf.nn.relu( tf.matmul(D, W_fc) + b_fc )
    D = tf.nn.dropout(D, p_keep)
    
    if use_token:
        D= tf.concat((T, D), axis=1)
    
    # if learn_embd:
    #     D= D + Dpos_emb
   
    for i_depth in range(2):
        
        if layer_norm:
            D= tf.contrib.layers.layer_norm(D)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='DW_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(D, W_q)
            Q = tf.nn.dropout(Q, p_keep)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='DW_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(D, W_k)
            K = tf.nn.dropout(K, p_keep)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='DW_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(D, W_v)
            V = tf.nn.dropout(V, p_keep)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='DA_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            # A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                new_inp= SA
            else:
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            D= D + new_inp
        else:
            D= new_inp
                    
        if layer_norm:
            D= tf.contrib.layers.layer_norm(D)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='DW_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='Db_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(D, W_l) + b_l )
        new_inp = tf.nn.dropout(new_inp, p_keep)
        
        if residuals:
            D= D + new_inp
        else:
            D= new_inp
    
    
    D= tf.reshape(D, tf.stack((-1, n_inp, d_emb)))
    
    
    inp= S+D
    
    ######
    
    n_feat_in=  d_emb
    n_feat_out= n_class
    
    W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_n_n' )
    b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_n_n' )
    
    # out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
    out = tf.matmul(inp, W_oout) + b_out
    
    out = tf.reshape(out, tf.stack((-1, L, L, L, n_class)))
    
    
    
    
    
    return out, A_list
































def fast_2d_net(inp, Conv_n_feat, Conv_kern_size, Conv_stride, Pool_kern_size, Pool_stride, \
                FC_n_feat, Deconv_n_feat, Deconv_kern_size, Deconv_stride, \
                p_keep_conv=1.0, p_keep_fc=1.0, n_channel=1, n_class=1, bias_init=0.001):
    
    for i_conv in range(len(Conv_n_feat)):
        
        ks= Conv_kern_size[i_conv]
        n_feat_in  = n_channel if i_conv==0 else Conv_n_feat[i_conv-1]
        n_feat_out = Conv_n_feat[i_conv]
        strd= Conv_stride[i_conv]
        
        n_l = n_feat_in * ks ** 2
        s_dev = np.sqrt(2.0 / n_l)
        
        name_w = 'W_' + str(i_conv) + '_init'
        name_b = 'b_' + str(i_conv) + '_init'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
        
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        ks=   Pool_kern_size[i_conv]
        strd= Pool_stride[i_conv]
        inp= tf.nn.max_pool2d(inp, ks, strd, padding='SAME')
        
    inp= tf.layers.flatten(inp)
    
    for i_fc in range(len(FC_n_feat)-1):
        
        n_feat_in=  FC_n_feat[i_fc]
        n_feat_out= FC_n_feat[i_fc+1]
        
        s_dev = np.sqrt(2.0 / n_feat_in)
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= s_dev), name='FC_w_' + str(i_fc) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='FC_b_' + str(i_fc) )
        
        inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        inp = tf.nn.dropout(inp, p_keep_fc)
    
    inp= tf.reshape(inp, [-1, int(FC_n_feat[-1]**.5), int(FC_n_feat[-1]**.5), 1])
    
    for i_deconv in range(len(Deconv_n_feat)):
        
        n_feat_in  = 1 if i_deconv==0 else Deconv_n_feat[i_deconv-1]
        n_feat_out = Deconv_n_feat[i_deconv]
        ks= Deconv_kern_size[i_deconv]
        
        n_l = n_feat_in * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(i_deconv) + '_up'
        name_b = 'b_' + str(i_deconv) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, n_feat_out, n_feat_in], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat_in]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, n_feat_out])
        inp = tf.nn.conv2d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 1], padding='SAME')
        inp = tf.nn.relu(tf.add(inp, b_deconv))
        inp= tf.nn.dropout(inp, p_keep_conv)
    
    
    return inp









# Conv_kern_size= [3, 3, 3, 3]
# Conv_n_feat = [16, 32, 64, 128]
# Conv_stride= [2,2,2,2]

# def shelf_net_2d(inp, Conv_kern_size, Conv_stride,  \
#                 Deconv_stride, \
#                 p_keep_conv=1.0, p_keep_fc=1.0, n_channel=1, n_class=1, bias_init=0.001):
    
#     ks= Conv_kern_size[0]
#     n_feat_in  = n_channel
#     n_feat_out = Conv_n_feat[0]
#     strd= Conv_stride[0]
#     n_l = n_feat_in * ks ** 2
#     s_dev = np.sqrt(2.0 / n_l)
#     name_w = 'W_0_init'
#     name_b = 'b_0_init'
#     W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
#     b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
#     cov_inp_1 = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
    
#     ks= Conv_kern_size[1]
#     n_feat_in  = Conv_n_feat[0]
#     n_feat_out = Conv_n_feat[1]
#     strd= Conv_stride[1]
#     n_l = n_feat_in * ks ** 2
#     s_dev = np.sqrt(2.0 / n_l)
#     name_w = 'W_1_init'
#     name_b = 'b_1_init'
#     W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
#     b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
#     cov_inp_2 = tf.nn.relu(tf.add(tf.nn.conv2d(cov_inp_1, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
    
#     ks= Conv_kern_size[2]
#     n_feat_in  = Conv_n_feat[1]
#     n_feat_out = Conv_n_feat[2]
#     strd= Conv_stride[2]
#     n_l = n_feat_in * ks ** 2
#     s_dev = np.sqrt(2.0 / n_l)
#     name_w = 'W_2_init'
#     name_b = 'b_2_init'
#     W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
#     b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
#     cov_inp_3 = tf.nn.relu(tf.add(tf.nn.conv2d(cov_inp_2, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
    
#     ks= Conv_kern_size[3]
#     n_feat_in  = Conv_n_feat[2]
#     n_feat_out = Conv_n_feat[3]
#     strd= Conv_stride[3]
#     n_l = n_feat_in * ks ** 2
#     s_dev = np.sqrt(2.0 / n_l)
#     name_w = 'W_3_init'
#     name_b = 'b_3_init'
#     W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_in, n_feat_out], stddev=s_dev), name=name_w)
#     b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
#     cov_inp_4 = tf.nn.relu(tf.add(tf.nn.conv2d(cov_inp_3, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     inp= tf.nn.dropout(inp, p_keep_conv)
    
#     ks=   Pool_kern_size[i_conv]
#     strd= Pool_stride[i_conv]
#     inp= tf.nn.max_pool2d(inp, ks, strd, padding='SAME')
        
#     inp= tf.layers.flatten(inp)
    
#     for i_fc in range(len(FC_n_feat)-1):
        
#         n_feat_in=  FC_n_feat[i_fc]
#         n_feat_out= FC_n_feat[i_fc+1]
        
#         s_dev = np.sqrt(2.0 / n_feat_in)
#         W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= s_dev), name='FC_w_' + str(i_fc) )
#         b_l = tf.Variable(tf.zeros([n_feat_out]), name='FC_b_' + str(i_fc) )
        
#         inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
#         inp = tf.nn.dropout(inp, p_keep_fc)
    
#     inp= tf.reshape(inp, [-1, int(FC_n_feat[-1]**.5), int(FC_n_feat[-1]**.5), 1])
    
#     for i_deconv in range(len(Deconv_n_feat)):
        
#         n_feat_in  = 1 if i_deconv==0 else Deconv_n_feat[i_deconv-1]
#         n_feat_out = Deconv_n_feat[i_deconv]
#         ks= Deconv_kern_size[i_deconv]
        
#         n_l = n_feat_in * ks ** 3
#         s_dev = np.sqrt(2.0 / n_l)
#         name_w = 'W_' + str(i_deconv) + '_up'
#         name_b = 'b_' + str(i_deconv) + '_up'
#         W_deconv = tf.Variable(tf.truncated_normal([ks, ks, n_feat_out, n_feat_in], stddev=s_dev), name=name_w)
#         b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat_in]), name=name_b)
#         in_shape = tf.shape(inp)
#         out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, n_feat_out])
#         inp = tf.nn.conv2d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 1], padding='SAME')
#         inp = tf.nn.relu(tf.add(inp, b_deconv))
#         inp= tf.nn.dropout(inp, p_keep_conv)
    
    
#     return inp















        





def style_transfer_cae(X, ks_0, n_feat_vec, n_enrich_vec, strd_vec, n_channel, p_keep_conv, bias_init=0.001):
    
    
    '''if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))'''
    
    inp = X
    
    F_list = [None] * (len(n_feat_vec)*2)
    i_F_list= -1
    
    # Enc.
    
    for i in range(len(n_feat_vec)):
        
        if i==0:
            n_feat_in= n_channel
        else:
            n_feat_in= n_feat_vec[i-1]
        n_feat_out= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**2
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_down'
        name_b= 'b_'+ str(i) + '_down'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd,strd, 1], padding='SAME'), b_1))
        inp = tf.nn.dropout(inp, p_keep_conv)
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            n_feat_in= n_feat_out= n_feat_vec[i]
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**2
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_down'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_down'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
                inp = tf.nn.dropout(inp, p_keep_conv)
            
            inp+= inp0
        
        i_F_list+= 1
        F_list[i_F_list]= inp
        
    for i in range(len(n_feat_vec)-1,-1,-1):
        
        if i==0:
            n_feat_out= n_channel
        else:
            n_feat_out= n_feat_vec[i-1]
        n_feat_in= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**2
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_up'
        name_b= 'b_'+ str(i) + '_up'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_out,n_feat_in], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, n_feat_out])
        if i>0:
            inp = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(inp, W_1, out_shape, strides=[1, strd,strd, 1], padding='SAME'), b_1))
        else:
            inp = tf.add(tf.nn.conv2d_transpose(inp, W_1, out_shape, strides=[1, strd,strd, 1], padding='SAME'), b_1)
        
        if i>0:
            inp = tf.nn.dropout(inp, p_keep_conv)
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            if i==0:
                n_feat_in= n_feat_out= n_channel
            else:
                n_feat_in= n_feat_out= n_feat_vec[i-1]
                
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**2
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_up'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_up'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                if i>0:
                    inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd,strd, 1], padding='SAME'), b_1))
                else:
                    inp = tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd,strd, 1], padding='SAME'), b_1)
                
                if i>0:
                    inp = tf.nn.dropout(inp, p_keep_conv)
            
            inp+= inp0
        
        i_F_list+= 1
        F_list[i_F_list]= inp
            
    
    return inp, F_list
























def my_net_2d_style_transfer(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.0):
    
    
    F_list = [None] * (depth*2)
    i_F_list= -1
    
    feat_fine = [None] * (depth - 1)
    
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(X, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(X, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 2
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 2
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv2d(inp_0, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 3)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_3, strides=[1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_3, strides=[1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
        
        
        i_F_list+= 1
        F_list[i_F_list]= inp
    
    
    
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 2
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] // 2])
        Deconv = tf.nn.conv2d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 3)
        
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 2
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_3, strides=[1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 2
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_3, strides=[1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_1 + inp_0  ###
            
        i_F_list+= 1
        F_list[i_F_list]= inp
    
    
    n_l = n_feat * ks ** 2
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv2d(inp, W_1, strides=[1, 1, 1, 1], padding='SAME'), b_1)
    
    # n_l = n_feat * ks ** 2
    # s_dev = np.sqrt(2.0 / n_l)
    # name_w_s = 'W_out_s'
    # name_b_s = 'b_out_s'
    # W_1_s = tf.Variable(tf.truncated_normal([ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    # b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    # output_s = tf.add(tf.nn.conv2d(inp, W_1_s, strides=[1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    
    return output, F_list



































def style_transfer_cae_3D(X, ks_0, n_feat_vec, n_enrich_vec, strd_vec, n_channel, p_keep_conv, bias_init=0.001):
    
    inp = X
    
    F_list = [None] * (len(n_feat_vec)*2)
    i_F_list= -1
    
    # Enc.
    
    for i in range(len(n_feat_vec)):
        
        if i==0:
            n_feat_in= n_channel
        else:
            n_feat_in= n_feat_vec[i-1]
        n_feat_out= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_down'
        name_b= 'b_'+ str(i) + '_down'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        inp = tf.nn.dropout(inp, p_keep_conv)
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            n_feat_in= n_feat_out= n_feat_vec[i]
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**3
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_down'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_down'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd, strd, 1], padding='SAME'), b_1))
                inp = tf.nn.dropout(inp, p_keep_conv)
            
            inp+= inp0
        
        i_F_list+= 1
        F_list[i_F_list]= inp
        
    for i in range(len(n_feat_vec)-1,-1,-1):
        
        if i==0:
            n_feat_out= n_channel
        else:
            n_feat_out= n_feat_vec[i-1]
        n_feat_in= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_up'
        name_b= 'b_'+ str(i) + '_up'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_out,n_feat_in], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, n_feat_out])
        if i>0:
            inp = tf.nn.relu(tf.add(tf.nn.conv3d_transpose(inp, W_1, out_shape, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        else:
            inp = tf.add(tf.nn.conv3d_transpose(inp, W_1, out_shape, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1)
        
        if i>0:
            inp = tf.nn.dropout(inp, p_keep_conv)
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            if i==0:
                n_feat_in= n_feat_out= n_channel
            else:
                n_feat_in= n_feat_out= n_feat_vec[i-1]
                
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**3
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_up'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_up'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                if i>0:
                    inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
                else:
                    inp = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1)
                
                if i>0:
                    inp = tf.nn.dropout(inp, p_keep_conv)
            
            inp+= inp0
        
        i_F_list+= 1
        F_list[i_F_list]= inp
            
    
    return inp, F_list














def hist_classify_2d(X, ks_0, n_feat_vec, n_enrich_vec, n_channel, n_class, p_keep_conv, bias_init=0.001):
    
    inp = X
    
    for i in range(len(n_feat_vec)):
        
        if i==0:
            n_feat_in= n_channel
        else:
            n_feat_in= n_feat_vec[i-1]
        n_feat_out= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**2
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_down'
        name_b= 'b_'+ str(i) + '_down'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd,strd, 1], padding='SAME'), b_1))
        inp = tf.nn.dropout(inp, p_keep_conv)
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            n_feat_in= n_feat_out= n_feat_vec[i]
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**2
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_down'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_down'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv2d(inp, W_1, strides=[1, strd, strd, 1], padding='SAME'), b_1))
                inp = tf.nn.dropout(inp, p_keep_conv)
            
            inp+= inp0
            
            # print(i,inp)
            
        
    n_final_feat= 2048
    
    inp = tf.reshape(inp, [-1, n_final_feat])
    
    n_feat_in= n_final_feat
    n_feat_out= 200
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu_1')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu_1')
    
    inp = tf.matmul(inp, W_fc) + b_fc
    
    
    n_feat_in= 200
    n_feat_out= n_class
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fcu_2')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fcu_2')
    
    inp = tf.matmul(inp, W_fc) + b_fc
            
    
    return inp









