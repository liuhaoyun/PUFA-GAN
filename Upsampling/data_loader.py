# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:11 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : data_loader.py

import numpy as np
import h5py
import queue
import threading
from Common import point_operation
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
import tensorflow as tf

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def batch_sampling(input_data,num):
    B,N,C = input_data.shape
    out_data = np.zeros([B,num,C])
    for i in range(B):
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:num]
        out_data[i,...] = input_data[i,idx]
    return out_data

def load_h5_data(h5_filename='', opts=None, skip_rate = 1, use_randominput=True):

# skip_rate  选择数据集中多个点patch 1,： 全部； k ：以 k为步长，等间隔选择 patch

    num_point = opts.num_point
    num_4X_point = int(opts.num_point*4)
    num_out_point = int(opts.num_point*opts.up_ratio)

    print("h5_filename : ",h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d'%num_4X_point][:]
        gt = f['poisson_%d'%num_out_point][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d' % num_point][:]
        gt = f['poisson_%d' % num_out_point][:]

    #name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalization the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))
    return input, gt, data_radius


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher,self).__init__()
        self.queue = queue.Queue(50)  # 50  可能笑了 相当于  只能选  50 * number of batch 个 patch
        self.stopped = False
        self.opts = opts
        self.use_random_input = self.opts.use_non_uniform
        self.input_data, self.gt_data, self.radius_data = load_h5_data(self.opts.train_file,opts=self.opts,use_randominput=self.use_random_input)
        self.batch_size = self.opts.batch_size
        self.sample_cnt = self.input_data.shape[0]
        self.patch_num_point = self.opts.patch_num_point
        self.num_batches = self.sample_cnt//self.batch_size
        print ("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)  # 一： 打乱点云patch 的输入顺序
            self.input_data = self.input_data[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()  # 让 input_data 的改动 不影响 batch_input_data
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
                radius = self.radius_data[start_idx:end_idx].copy()

                if self.use_random_input:
                    new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
                    for i in range(self.batch_size):
                        idx = point_operation.nonuniform_sampling(self.input_data.shape[1], sample_num=self.patch_num_point)  # 随机从 1024 个点 采样 256 个点，每次不重样
                        new_batch_input[i, ...] = batch_input_data[i][idx]
                    batch_input_data = new_batch_input
                if self.opts.augment:
                    batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max) # 二：让几何信息(每个点)在极小范围内浮动
                    batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)  # 三：默认把 input 和 ground truth 绕 Z 轴旋转 相同的随机角度
                    batch_input_data, batch_data_gt, scales = point_operation.random_scale_point_cloud_and_gt(batch_input_data, # 四：把 每个patch 的 input 和 及对应的 ground truth 缩放 scale倍，实现尺度缩放
                                                                                              batch_data_gt,
                                                                                              scale_low=0.8,
                                                                                              scale_high=1.2)

                    radius = radius * scales
                # 最终输出的是 经过所有数据增强方法后的patch版本，而不是每个数据增强方法输出一组增强后的 patch。
                # input [B,N,C] , output [B,N,C]
                self.queue.put((batch_input_data[:,:,:3], batch_data_gt[:,:,:3],radius))
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()  # 先进先出队列

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")





