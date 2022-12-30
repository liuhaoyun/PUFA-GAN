from tf_ops.sampling.tf_sampling import farthest_point_sample , gather_point
import tensorflow as tf
from glob import glob
import numpy as np
import os
from Common import pc_util, ops, model_utils

test_data = '/home/vim/SR_GAN/Test_results/Normalized_ModelNet40/*.xyz'
samples = glob(test_data) # 通过 glob 遍历所有测试数据的路径，以 list 形式存储
len = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__=='__main__':

    with tf.Session() as sess:
        for i in samples:
            temp_dir = '/home/vim/SR_GAN/Test_results/Normalized_ModelNet40/2048/'  # 放在哪里
            point = pc_util.load(samples[len])  # 测试样本点云
            point = tf.convert_to_tensor(point)
            point = tf.expand_dims(point,axis = 0)
            # PU147: (0.001, 2.0), Normalized_ModelNet40: (0.005, 2.0)
            edge_point = ops.Edge_points = ops.High_Pass_Graph_Filter(point, 2048, 0.001, 2.0).eval()
            edge_point = edge_point[0,:,:]

            filepath, tempfilename = os.path.split(i)
            output_path = temp_dir + tempfilename
            # output downsampling point cloud
            np.savetxt(output_path, edge_point, fmt='%.6f')
            len += 1