# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

# MLP_CONV2D
# input：[B, N, C]
# output : [B, N ,C]
def Edge_conv (inputs, output_dims, k = 8, scope='edge_conv', pooling ='max_pooling', activation_fn=tf.nn.relu,
               is_training=True, use_bn=False ,use_ibn=False, bn_decay=None, weight_decay=0.0):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        feature, _ = get_edge_feature(inputs, k=k, idx=None)  # 拿到KNN 个点的特征 [B，N，K，2*C]
        feature = conv2d(feature, output_dims, [1, 1],
                        padding='VALID', scope='dgcnn_conv', is_training=is_training, bn=use_bn, ibn=use_ibn,
                        weight_decay=weight_decay, activation_fn = activation_fn)  # 不进行参数的 L2正则

        if pooling == 'max_pooling':
            output = tf.reduce_max(feature, axis=-2)  # [B, N, C]
        else:
            output = att_pooling(feature, scope='attentive_pooling')  # [B, N, C]

    return output

def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

#################################################################################
#  High Pass Graph Filter
#################################################################################
def High_Pass_Graph_Filter(point_cloud, k = 350, dist = 0.5 , sigma = 2.0):
#  Input
#  point cloud :  [B, N, C]
#  k : the number of sampled point of each batch
#  dist : distance threshold
#  simga : Gaussian kernel variation

# Output
# Edge_points : Sampled points [B, N, k]

    B = point_cloud.get_shape().as_list()[0]  # Batch 数目
    N = point_cloud.get_shape().as_list()[1]  # N 是点的数目

    adj = pairwise_distance(point_cloud)
    zero = tf.zeros_like(adj)
    one = tf.ones_like(adj)
    mask = tf.where(adj <= dist, x=one, y=zero)

    variation = adj / (sigma * sigma * -1)
    W_fake = tf.exp(variation)  # 求出高斯函数值。注意对角线元素本来均为0，但 e 的0次幂会让对角线元素变为1 (后续要处理掉)
    W = mask * W_fake
    I = tf.eye(N, batch_shape=[B])
    W = W - I  # 去掉对角线上的 1
    sum_W = tf.reduce_sum(W, axis=-1, keepdims=True)  # 求出每行元素的和
    normalization = tf.tile(sum_W, [1, 1, N])  # 扩展维度
    normalization = tf.where(tf.not_equal(normalization, 0), normalization, one)  # 防止某个点没有找到近邻点，在标准化时 0 变成 被除数 (Inf)
    A = W / normalization  # 标准化后获取 传输矩阵A
    H_A = I - A  # 生成 高通图滤波器 # [B, N, N]
    Filtered_signal = tf.matmul(H_A, point_cloud)  # 使用 高通滤波器去提取信号频域特性  [B, N, C]     !!!!!!   显存 容易爆炸  !!!!!!!
    L2_square = tf.reduce_sum(tf.square(Filtered_signal), axis=-1, keep_dims=True)  # 求出每个点在 L2范数下的频域的幅值
    Value, index = tf.nn.top_k(tf.squeeze(L2_square, [2]), k=k)  # 提取幅值最大的K个点作为 Edge points

    idx_bais = tf.reshape(tf.range(0, B), [B, 1]) * N
    idx_bais_tile = tf.tile(idx_bais, [1, k])
    index_new = index + idx_bais_tile  # 接上索引信息
    point_cloud_reshape = tf.reshape(point_cloud, [B * N, -1])  # 把输入由 原来的 [B, N, C ] 变成 [B*N, C]

    new_point = tf.gather_nd(point_cloud_reshape, tf.reshape(index_new, [B * k, -1]))  # [B*k，C] 选取最终的 k个点，作为高频点
    Edge_points = tf.reshape(new_point, [B, k, -1])  # [B,N,C] 输出经过高通图滤波器采样后的点云

    return Edge_points


def High_Pass_Graph_Filter0(point_cloud, k = 256, dist = 0.5, sigma = 2.0):
#  Input
#  point cloud :  [B, N, C]
#  k : the number of sampled point of each batch
#  dist : distance threshold
#  simga : Gaussian kernel variation

# Output
# Edge_points : Sampled points [B, N, k]
    B = point_cloud.get_shape().as_list()[0]  # Batch 数目
    N = point_cloud.get_shape().as_list()[1]  # N 是点的数目

    adj = pairwise_distance(point_cloud)
    zero = tf.zeros_like(adj)
    one = tf.ones_like(adj)
    mask = tf.where(adj <= dist, x=one, y=zero)
    variation = adj / (sigma * sigma * -1)

    W_fake = tf.exp(variation)  # 求出高斯函数值。注意对角线元素本来均为0，但 e 的0次幂会让对角线元素变为1 (后续要处理掉)
    W = mask * W_fake
    I = tf.eye(N, batch_shape=[B])
    W = W - I  # 去掉对角线上的 1
    sum_W = tf.reduce_sum(W, axis=-1, keepdims=True)  # 求出每行元素的和
    normalization = tf.tile(sum_W, [1, 1, N])  # 扩展维度
    normalization = tf.where(tf.not_equal(normalization, 0), normalization, one)  # 防止某个点没有找到近邻点，在标准化时 0 变成 被除数 (Inf)
    A = W / normalization  # 标准化后获取 传输矩阵A
    H_A = I - A  # 生成 高通图滤波器 # [B, N, N]
    Filtered_signal = tf.matmul(H_A, point_cloud)  # 使用 高通滤波器去提取信号频域特性  [B, N, C]     !!!!!!   显存 容易爆炸  !!!!!!!
    L2_square = tf.reduce_sum(tf.square(Filtered_signal), axis=-1, keep_dims=True)  # 求出每个点在 L2范数下的频域的幅值
    Value, index = tf.nn.top_k(tf.squeeze(L2_square, [2]), k=N)  # 提取幅值最大的K个点作为 Edge points
    # 获取 高频 和 低频 点的索引
    #high_idx = index[:,0:k]
    low_idx = index[:,k:N]
    idx_bais = tf.reshape(tf.range(0, B), [B, 1]) * N
    #high_idx_bais = tf.tile(idx_bais, [1, k])
    low_idx_bais = tf.tile(idx_bais, [1, N-k])
    # high_index_new = high_idx + high_idx_bais  # 高频索引
    low_index_new = low_idx + low_idx_bais  # 低频索引

    point_cloud_reshape = tf.reshape(point_cloud, [B * N, -1])  # 把输入由 原来的 [B, N, C ] 变成 [B*N, C]
    # high_new_point = tf.gather_nd(point_cloud_reshape, tf.reshape(high_index_new, [B * k, -1]))  # [B*k，C] 选取最终的 k个点，作为高频点
    low_new_point = tf.gather_nd(point_cloud_reshape, tf.reshape(low_index_new, [B * (N-k), -1]))  # [B*（N-k)，C] 选取最终的 N-k个点, 作为低频点

    #edge_points = tf.reshape(high_new_point, [B, k, -1])  # [B,k,C] 输出经过高通图滤波器采样后的点云
    flat_points = tf.reshape(low_new_point, [B, (N-k), -1])  # [B,k,C] 输出经过高通图滤波器采样后的点云

    return  flat_points

##################################################################################
                           # gather_features
##################################################################################
# input : [B,N,C],
# idx : [B,k]
# output : [B,N,k]

def gather_features(input, idx):
    B, N, C = [i.value for i in input.get_shape()]
    _, k = [j.value for j in idx.get_shape()]

    idx_bais = tf.reshape(tf.range(0, B), [B, 1]) * N
    idx_bais_tile = tf.tile(idx_bais, [1, k])
    index_new = idx + idx_bais_tile  # 接上索引信息
    input_reshape = tf.reshape(input, [B * N, -1])  # 把输入由 原来的 [B,N,C] 变成 [B*N, C]
    new_point = tf.gather_nd(input_reshape, tf.reshape(index_new, [B * k, -1]))
    new_point = tf.reshape(new_point, [B, k, -1])  # [B,N,C] 输出经过高通图滤波器采样后的点云

    return new_point

##################################################################################
# Back projection Blocks
##################################################################################
def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs

from Common.model_utils import gen_1d_grid,gen_grid
def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim*up_ratio
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
            #grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def down_block(inputs,up_ratio,scope='down_block',is_training=True,bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net,[tf.shape(net)[0],up_ratio,-1,tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net, 256, [1, up_ratio],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

# SE-Net (Channel attention model)
# input ： (B, N, 1，C)
# output : (B, N ,1, C)
def SE_NET(input, scope='se_net', is_training=True, bn_decay=None, use_bn = False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c = input.get_shape().as_list()[3]
        feature = tf.reduce_mean(input, axis=[1, 2])  # Global Average Pooling [B C]
        feature = tf.expand_dims(feature, axis=1)  # [B 1 C]
        feature = tf.expand_dims(feature, axis=2)  # [B 1 1 C]
        feature = conv2d(feature, c//16, [1, 1],
                          padding='VALID', scope='SE_0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay)
        feature = conv2d(feature, c, [1, 1],
                          padding='VALID', scope='SE_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)

        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out

# 残差块
# input :  (B, N, 1, C)
# output : (B, N ,1, C)
def Rssidual_Block(input, C_OUT, scope='residual_block', is_training=True,
                   use_ibn=False, use_bn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gamma = 4  # bottleNeck ratio
        x = input
        # RB : Conv + Bn + Relu
        residual = conv2d(input, C_OUT, [1, 1],
                     padding='VALID', scope='bottle_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                     bn_decay=bn_decay)

        residual = conv2d(residual, C_OUT//gamma, [1, 1],
                     padding='VALID', scope='bottle_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                     bn_decay=bn_decay)

        residual = conv2d(residual, C_OUT, [1, 1],
                          padding='VALID', scope='bottle_3', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)  # Conv + bn ， 最后一层 没有 Relu

        y = x + residual
        y = tf.nn.relu(y)
    return y, residual   # 返回 输出 和 当前块的残差值


def Chain_Residual_Block(input, output=128, block_num=4, scope='chain_residual_block',
                         is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        identity = input
        sum_residual = identity  # 存储每个链的残差

        for i in range(block_num):
            if i == 0:
                feature, residual = Rssidual_Block(identity, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = residual
            else:
                feature, residual = Rssidual_Block(feature, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = tf.concat([sum_residual, residual], axis=-1)  # concat 所有链的 residuals  [n * 128]

        sum_residual = SE_NET(sum_residual, scope='se_net',is_training=is_training)  # 对残差进行 attention
        sum_residual = conv2d(sum_residual, output, [1, 1],
                              padding='VALID', scope='layer_compress', is_training=is_training, bn=use_bn, ibn=use_ibn,
                              bn_decay=bn_decay, activation_fn=None)  # [128]  No Relu
        out = identity + sum_residual
        out = tf.nn.relu(out)
    return out

def Feature_Up_Sampling_CRB(input, up_ratio, output=256, block_num=4, layer=3, scope='CRB',
                            is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # add 2D girds
        net = input
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1, tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        C = input.get_shape().as_list()[-1]
        # align feature
        if C != output:
            net = conv2d(net, output, [1, 1], padding='VALID', scope='Adjust_layer',
                              is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 使用 Conv + Relu
        # self-attention
        net = attention_unit(net, is_training=is_training)
        for i in range(layer):
            net = Chain_Residual_Block(net, output, block_num, scope='chain_residual_block%d' % i,
                                           is_training=is_training, use_bn=use_bn, use_ibn=use_ibn, bn_decay=bn_decay)  # [B, N, 128]
    return net

def RCB_conv(input, k, output=256, block_num=4, layer=3, scope='CRB', is_training=True,
             use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        y, idx = get_edge_feature(input, k=k, idx=None)  # [B N K 2*C]
        for i in range(layer):
            y = Chain_Residual_Block(y, output, block_num, scope='chain_residual_block%d' % i,
                                     is_training=is_training, use_bn=use_bn, use_ibn=use_ibn, bn_decay=bn_decay)  # [B, N, 128]

        y = conv2d(y, output, [1, 1], padding='VALID', scope='Adjust_layer', activation_fn = None,
                   is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 使用 Conv without Relu

        # attentive_pooling
        #y = att_pooling(y, scope='attentive_pooling') # [B, N, C]
        # max_pooling

        y = tf.reduce_max(y, axis=-2)  # [B, N, C]

        return y, idx

# Attentive Pooling
# Input
# feature_set : [B N K C]  dout : 输出 channel数目

def att_pooling(feature_set, scope = 'attentive_pooling'):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size, num_points, num_neigh, d = [i.value for i in feature_set.get_shape()]
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])  # [B*N, K, C]
        #att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False)  # 不使用 激活，BN ,偏置项
        att_activation = fully_connected(f_reshaped, d, scope='attentive_pooling_FC',activation_fn=None)
        att_scores = tf.nn.softmax(att_activation, axis=1)  # 计算当前特征通道C_i内，K个近邻点的贡献程度(即权重)
        f_agg = f_reshaped * att_scores  # 加权
        f_agg = tf.reduce_sum(f_agg, axis=1) # [B*N, C] 求和
        # f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])  # [B, N, 1, C]
        f_agg = tf.reshape(f_agg, [batch_size, num_points, d])  # [B, N, C]

    return f_agg

def feature_extraction_RCB(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 32

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, growth_rate, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = RCB_conv(l0_features, k=knn, output= growth_rate * 2, scope="layer1",
                                       is_training=is_training,bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # 96

        l2_features = conv1d(l1_features, comp, 1,  padding='VALID', scope='layer2_prep',
                             is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)

        l2_features, l2_idx  = RCB_conv(l2_features, k=knn, output= growth_rate * 4, scope="layer2",
                                       is_training=is_training,bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 224

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = RCB_conv(l3_features, k=knn, output= growth_rate * 4, scope="layer3",
                                       is_training=is_training,bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 352

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l4_idx = RCB_conv(l4_features, k=knn, output= growth_rate * 4, scope="layer4",
                                       is_training=is_training,bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 480

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


def feature_extraction(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l4_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features

def up_projection_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = conv2d(inputs, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='conv0', bn_decay=bn_decay)

        H0 = up_block(L,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_0')

        L0 = down_block(H0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='down_0')
        E0 = L0-L
        H1 = up_block(E0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_1')
        H2 = H0+H1
    return H2

def weight_learning_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape().as_list()[-1]
        grid = gen_1d_grid(tf.reshape(up_ratio,[]))

        out_dim = dim * up_ratio

        ratios = tf.tile(tf.expand_dims(up_ratio,0),[1,tf.shape(grid)[1]])
        grid_ratios = tf.concat([grid,tf.cast(ratios,tf.float32)],axis=1)
        weights = tf.tile(tf.expand_dims(tf.expand_dims(grid_ratios,0),0),[tf.shape(inputs)[0],tf.shape(inputs)[1], 1, 1])
        weights.set_shape([None, None, None, 2])
        weights = conv2d(weights, dim, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_1', bn_decay=None)


        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_2', bn_decay=None)
        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_3', bn_decay=None)

        s = tf.matmul(hw_flatten(inputs), hw_flatten(weights), transpose_b=True)  # # [bs, N, N]

    return tf.expand_dims(s,axis=2)


def coordinate_reconstruction_unit(inputs,scope="reconstruction",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs

# non-local attention (self-attention)
def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

from tf_ops.grouping.tf_grouping import knn_point_2
def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx


def get_KNN_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    dist = point_cloud
    if idx is None:
        dist, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]
        dist = dist[:,:,1:]
    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)

    return point_cloud_neighbors, dist, idx

def dense_conv_att_pooling(feature, n=3,growth_rate=64, k=16, scope='dense_conv_att_pooling',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        # attentive_pooling
        y = att_pooling(y, scope='attentive_pooling') # [B, N, C]
        # max_pooling
        #y = tf.reduce_max(y, axis=-2)  # [B, N, C]
        return y,idx

def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        # attentive_pooling
        #y = att_pooling(y, scope='attentive_pooling') # [B, N, C]
        # max_pooling
        y = tf.reduce_max(y, axis=-2)  # [B, N, C]
        return y,idx

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])   # 一般在画loss,accuary时会用到这个函数
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])  # 一般用来显示训练过程中变量的分布情况

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
