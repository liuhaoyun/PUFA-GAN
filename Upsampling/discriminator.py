# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com

import tensorflow as tf
from Common import ops

class Discriminator(object):
    def __init__(self, opts,is_training, name="Discriminator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.start_number = 32
        #print('start_number:',self.start_number)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs = tf.expand_dims(inputs,axis=2)
            with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(inputs, [self.start_number, self.start_number * 2])
                features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
                features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1],1, 1])], axis=-1)
                features = ops.attention_unit(features, is_training=self.is_training)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(features, [self.start_number * 4, self.start_number * 8])
                features = tf.reduce_max(features, axis=1, name='maxpool_1')

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                outputs = ops.mlp(features, [self.start_number * 8, 1])
                outputs = tf.reshape(outputs, [-1, 1])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return outputs


class Discriminator_DGCNN(object):
    def __init__(self, opts,is_training, name="Discriminator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.output_dims = 32  # 特征维度
        self.k = 20  # 最近邻的个数
        self.weight_decay = 0.0  # 默认没有 L2正则
        #print('start_number:',self.start_number)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                use_bn = False
                use_ibn = False
                # DGCNN
                inputs = tf.expand_dims(inputs, axis=2)  # [B, N, 1, C]

                feature1 = ops.conv2d(inputs, self.output_dims, [1, 1],
                                      padding='VALID', scope='edge_conv1', is_training=self.is_training, bn=use_bn,
                                      ibn=use_ibn, weight_decay=self.weight_decay)  # [B,N,1,C]
                feature1 = tf.squeeze(feature1, axis=2)  # [B, N ,C]
                feature2 = ops.Edge_conv(feature1, self.output_dims * 2, self.k, scope='edge_conv2', use_bn=use_bn,
                                         is_training=self.is_training, weight_decay=self.weight_decay)  # [B,N, C1 = 64]
                feature3 = ops.Edge_conv(feature2, self.output_dims * 2, self.k, scope='edge_conv3',use_bn=use_bn,
                                         is_training=self.is_training, weight_decay=self.weight_decay) # [B,N, C1 = 64]

                feature_total = tf.concat([feature1, feature2, feature3], axis=-1)
                feature_total = tf.expand_dims(feature_total, axis=2)  # [B, N, 1, C]
                feature_total = ops.conv2d(feature_total, 128, [1, 1],
                                      padding='VALID', scope='edge_conv4', is_training=self.is_training, bn=use_bn, ibn=use_ibn,
                                      weight_decay=self.weight_decay)  # [B,N,1,C]
                # Max_pooling
                feature_total = tf.reduce_max(feature_total, axis=1, name='maxpool_0')  # [B,1,C]

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                feature_total = ops.mlp(feature_total, [128, 1])   # [B, 1, 1]
                feature_total = tf.reshape(feature_total, [-1, 1])  # [B,1]

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return feature_total

class HF_Discriminator(object):
    def __init__(self, opts, is_training, name="Discriminator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.start_number = 32  # 特征维度
        self.k = 6  # 最近邻的个数
        self.weight_decay = 0.0  # 默认没有 L2正则
        # print('start_number:',self.start_number)

    def __call__(self, inputs, edge_inputs):

        with tf.variable_scope(self.name, reuse=self.reuse):

            inputs = tf.expand_dims(inputs, axis=2)
            # GAN_1
            with tf.variable_scope('encoder_0_global', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(inputs, [self.start_number, self.start_number * 2])
                features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0_global')
                features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1, 1])], axis=-1)
                features = ops.attention_unit(features, is_training=self.is_training)
            with tf.variable_scope('encoder_1_global', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(features, [self.start_number * 4, self.start_number * 8])
                # Max_pooling
                features = tf.reduce_max(features, axis=1, name='maxpool_1_global')

            # GAN_2
            with tf.variable_scope('encoder_edge', reuse=tf.AUTO_REUSE):

                edge_inputs = tf.expand_dims(edge_inputs, axis=2)  # [B, N, 1, C]
                with tf.variable_scope('encoder_0_edge', reuse=tf.AUTO_REUSE):
                    features1 = ops.mlp_conv( edge_inputs, [self.start_number, self.start_number * 2])
                    features1_global = tf.reduce_max(features1, axis=1, keep_dims=True, name='maxpool_0_edge')
                    features1 = tf.concat([features1, tf.tile(features1_global, [1, tf.shape(edge_inputs)[1], 1, 1])], axis=-1)
                    features1 = ops.attention_unit(features1, is_training=self.is_training)
                with tf.variable_scope('encoder_1_edge', reuse=tf.AUTO_REUSE):
                    features1 = ops.mlp_conv(features1, [self.start_number * 4, self.start_number * 6])
                    # Max_pooling
                    features1 = tf.reduce_max(features1, axis=1, name='maxpool_1_edge')

                output = tf.concat([features1, features], axis=-1)

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                output2 = ops.mlp(output, [self.start_number * 8, self.start_number * 4, 1])  # [B, 1, 1]
                output2 = tf.reshape(output2, [-1, 1])  # [B,1]


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return output2


