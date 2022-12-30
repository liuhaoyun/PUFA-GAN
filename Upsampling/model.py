# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
from Upsampling.generator import Generator
from Upsampling.discriminator import HF_Discriminator
from Common.visu_utils import plot_pcd_three_views,point_cloud_three_views
from Common.ops import add_scalar_summary,add_hist_summary, High_Pass_Graph_Filter, High_Pass_Graph_Filter0
from Upsampling.data_loader import Fetcher
from Common import model_utils
from Common import pc_util
from Common.loss_utils import pc_distance,get_uniform_loss,get_repulsion_loss,DH_discriminator_loss,DH_generator_loss,cosine_similar_loss, cosine_similar_loss_var
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
import logging
import os
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np

class Model(object):
  def __init__(self,opts,sess):
      self.sess = sess
      self.opts = opts

  def allocate_placeholders(self):
      self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training') # 不feed_dict 时，默认是真
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size,self.opts.num_point,3])  # 低分辨率 训练 patch 数据
      self.input_y = tf.placeholder(tf.float32, shape=[self.opts.batch_size, int(4*self.opts.num_point),3]) # grond truth 高分辨率 patch
      self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])

  def build_model(self):
      self.G = Generator(self.opts,self.is_training,name='generator')
      self.D = HF_Discriminator(self.opts, self.is_training, name='discriminator')
      # X -> Y
      self.G_y, self.G_feature, self.G_pred_input_x, self.G_Edge_y = self.G(self.input_x)
      # get edge point of ground truth
      N = self.input_y.get_shape().as_list()[1]
      scale_ori = tf.expand_dims(tf.expand_dims(self.pc_radius, axis=-1), axis=-1)
      scale_N = tf.tile(scale_ori, [1, N, 1])
      self.input_y_without_scale = self.input_y / scale_N
      self.edge_input_y_without_scale =  High_Pass_Graph_Filter(self.input_y_without_scale, self.opts.HPGF_Sample_num_point,
                                                                self.opts.HPGF_dist, self.opts.HPGF_sigma)
      scale_k = tf.tile(scale_ori, [1, self.opts.HPGF_Sample_num_point, 1])
      self.edge_input_y = self.edge_input_y_without_scale * scale_k

      # get loss
      self.dis_loss = self.opts.fidelity_w * pc_distance(self.G_y, self.input_y, radius=self.pc_radius)
      self.cycle_loss = self.opts.cycle_w * pc_distance(self.G_pred_input_x, self.input_x, radius=self.pc_radius)

      if self.opts.use_repulse:
          self.repulsion_loss = self.opts.repulsion_w*get_repulsion_loss(self.G_y)
      else:
          self.repulsion_loss = 0
      self.uniform_loss = self.opts.uniform_w * get_uniform_loss(self.G_y)
      self.pu_loss = self.cycle_loss + self.dis_loss + self.uniform_loss +  self.repulsion_loss + tf.losses.get_regularization_loss(scope='generator')

      self.G_gan_loss = self.opts.gan_w*DH_generator_loss(self.D,self.G_y,self.G_Edge_y)
      self.total_gen_loss = self.G_gan_loss + self.pu_loss
      self.D_W_L2 = tf.losses.get_regularization_loss(scope='discriminator')  # 网络参数 L2 loss
      self.D_loss = DH_discriminator_loss(self.D, self.input_y, self.edge_input_y, self.G_y, self.G_Edge_y)
      self.D_loss = self.D_loss + self.D_W_L2  # 判别器 加上 L2正则 loss

      self.setup_optimizer()
      self.summary_all()

      self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]] # 可视化 每个batch 中 第一个点云
      self.visualize_titles = ['input_x', 'fake_y', 'real_y']

  def summary_all(self):

      # summary
      add_scalar_summary('loss/dis_loss', self.dis_loss, collection='gen')
      add_scalar_summary('loss/repulsion_loss', self.repulsion_loss,collection='gen')
      add_scalar_summary('loss/uniform_loss', self.uniform_loss,collection='gen')
      add_scalar_summary('loss/G_loss', self.G_gan_loss,collection='gen')
      add_scalar_summary('loss/cycle_loss', self.cycle_loss, collection='gen')
      add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')

      add_hist_summary('D/true',self.D(self.input_y, self.edge_input_y),collection='dis')
      add_hist_summary('D/fake',self.D(self.G_y, self.G_Edge_y),collection='dis')
      add_scalar_summary('DIS/True', tf.reduce_mean(self.D(self.input_y,self.edge_input_y)), collection='dis')  # average distribution True
      add_scalar_summary('DIS/Fake', tf.reduce_mean(self.D(self.G_y, self.G_Edge_y)), collection='dis')  # average distribution Fake
      add_scalar_summary('loss/D_W_L2', self.D_W_L2, collection='dis')  #  L2 LOSS
      add_scalar_summary('loss/D_Y', self.D_loss,collection='dis')

      self.g_summary_op = tf.summary.merge_all('gen')
      self.d_summary_op = tf.summary.merge_all('dis')

      self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
      self.visualize_x_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
      self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 1])
      self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

  def setup_optimizer(self):

      learning_rate_d = tf.where(
          tf.greater_equal(self.global_step, self.opts.start_decay_step),
          tf.train.exponential_decay(self.opts.base_lr_d, self.global_step - self.opts.start_decay_step,
                                    self.opts.lr_decay_steps, self.opts.lr_decay_rate,staircase=True),
          self.opts.base_lr_d
      )
      learning_rate_d = tf.maximum(learning_rate_d, self.opts.lr_clip)
      add_scalar_summary('learning_rate/learning_rate_d', learning_rate_d,collection='dis')


      learning_rate_g = tf.where(
          tf.greater_equal(self.global_step, self.opts.start_decay_step),
          tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
                                     self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
          self.opts.base_lr_g
      )
      learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
      add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collection='gen')

      # create pre-generator ops
      gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
      gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]  # tf.trainable_variables() 收集所有需要训练的变量 与 GraphKeys.TRAINABLE_VARIABLES 等价

      with tf.control_dependencies(gen_update_ops):
          self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(self.total_gen_loss, var_list=gen_tvars,
                                                                                              colocate_gradients_with_ops=False,  # True 梯度值可以并行在多个GPU上计算
                                                                                              global_step=self.global_step)

       # create pre-discriminator ops
      dis_update_ops = [op1 for op1 in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op1.name.startswith("discriminator")]
      dis_tvars = [var1 for var1 in tf.trainable_variables() if var1.name.startswith("discriminator")]  # tf.trainable_variables()


      with tf.control_dependencies(dis_update_ops):
          self.D_optimizers = tf.train.AdamOptimizer(learning_rate_d, beta1=self.opts.beta).minimize(self.D_loss,
                                                                                                 self.global_step,
                                                                                                 var_list=dis_tvars,
                                                                                                 name='Adam_D_X')

  def train(self):

      self.allocate_placeholders()
      self.build_model()

      self.sess.run(tf.global_variables_initializer())

      fetchworker = Fetcher(self.opts)  # Fetcher为 处理训练数据的类，生成 Fetchworker对象
      fetchworker.start()  #启动多线程

      self.saver = tf.train.Saver(max_to_keep=None)  # 设置为 None或者0, 就是保存全部的模型
      self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)  # 初始化 生成可视化文件路径

      restore_epoch = 0
      if self.opts.restore:
          restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir) # 预加载已有的模型，返回已有模型的 epoch 和 检查点路径， 每个 epoch 存储一次模型
          self.saver.restore(self.sess, checkpoint_path) # Restore则是将训练好的参数提取出来
          #self.saver.restore(self.sess, tf.train.latest_checkpoint(self.opts.log_dir))
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
          tf.assign(self.global_step, restore_epoch * fetchworker.num_batches).eval()
          restore_epoch += 1

      else:
          os.makedirs(os.path.join(self.opts.log_dir, 'plots'))  # 不加载预训练模型，并生成 plots 文件夹
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

      with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:  # 把网络 参数 写入 args.txt中
          for arg in sorted(vars(self.opts)):     # vars(object) 返回对象object的属性和属性值的字典对象
              log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

      step = self.sess.run(self.global_step) # 获取当前 step， 即 Iterations 次数
      start = time()
      # print trainable parameters
      #v_names = [v.name for v in tf.trainable_variables()]
      #print(v_names)
      for epoch in range(restore_epoch, self.opts.training_epoch):  # Epoch.   range（0， 5） 是[0, 1, 2, 3, 4] 没有 5
          logging.info('**** EPOCH %03d ****\t' % (epoch))
          for batch_idx in range(fetchworker.num_batches):  # Iterations

              batch_input_x, batch_input_y,batch_radius = fetchworker.fetch() # 每次从队列中取出一个 batch_size 大小个 patch

              feed_dict = {self.input_x: batch_input_x,
                           self.input_y: batch_input_y,
                           self.pc_radius: batch_radius,
                           self.is_training: True}


              # Update D network
              _,d_loss,d_summary = self.sess.run([self.D_optimizers,self.D_loss,self.d_summary_op],feed_dict=feed_dict)
              self.writer.add_summary(d_summary, step)

              # Update G network
              for i in range(self.opts.gen_update):  # 生成器 更新 N (默认是2) 次， 判别器更新一次
                  # get previously generated images
                  _, g_total_loss, summary = self.sess.run(
                      [self.G_optimizers, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)
                  self.writer.add_summary(summary, step)

              if step % self.opts.steps_per_print == 0:  # 每过 n次 step 打印一次 Loss 及时间。 一般是每个 step 打印一次
                  self.log_string('-----------EPOCH %d Step %d:-------------' % (epoch,step))
                  self.log_string('  G_loss   : {}'.format(g_total_loss)) # 生成器 Loss
                  self.log_string('  D_loss   : {}'.format(d_loss)) # 判别器 Loss
                  self.log_string(' Time Cost : {}'.format(time() - start))
                  start = time()  # 重新计算时间
                  feed_dict = {self.input_x: batch_input_x,
                               self.is_training: False}

                  fake_y_val = self.sess.run([self.G_y], feed_dict=feed_dict)


                  fake_y_val = np.squeeze(fake_y_val) # 从数组的形状中删除单维度条目，即把shape中为 1 的维度去掉
                  image_input_x = point_cloud_three_views(batch_input_x[0]) # [1500, 500]
                  image_fake_y = point_cloud_three_views(fake_y_val[0])
                  image_input_y = point_cloud_three_views(batch_input_y[0, :, 0:3])
                  image_x_merged = np.concatenate([image_input_x, image_fake_y, image_input_y], axis=1) # [1500, 1500 ]
                  image_x_merged = np.expand_dims(image_x_merged, axis=0)
                  image_x_merged = np.expand_dims(image_x_merged, axis=-1)
                  image_x_summary = self.sess.run(self.image_x_summary, feed_dict={self.image_x_merged: image_x_merged})
                  self.writer.add_summary(image_x_summary, step)

              if self.opts.visulize and (step % self.opts.steps_per_visu == 0):   # 每过 n次 step 打印一次 Loss 及时间。 一般是每个 step 打印一次
                  feed_dict = {self.input_x: batch_input_x,
                               self.input_y: batch_input_y,
                               self.pc_radius: batch_radius,
                               self.is_training: False}  # 不训练，单纯为了可视化结果
                  pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
                  pcds = np.squeeze(pcds)  # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                  plot_path = os.path.join(self.opts.log_dir, 'plots',
                                           'epoch_%d_step_%d.png' % (epoch, step))
                  plot_pcd_three_views(plot_path, pcds, self.visualize_titles) # 把可视化的点云写入plots文件中

              step += 1
          if (epoch % self.opts.epoch_per_save) == 0:
              self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
              print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

      fetchworker.shutdown()

  def patch_prediction(self, patch_point):    # patch_point [N, 3]
      # normalize the point clouds
      patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
      patch_point = np.expand_dims(patch_point, axis=0)  # [1,N,3] <-> [B,N,C]
      pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point})  # 把patch 作为输入 进行 测试
      pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
      return pred

  def pc_prediction(self, pc):
      ## get patch seed from farthestsampling
      points = tf.convert_to_tensor(np.expand_dims(pc,axis=0),dtype=tf.float32)
      start= time()
      print('------------------patch_num_point:',self.opts.patch_num_point)
      seed1_num = int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)  # 把每个点云切分成 seed1_num 个 patch
      ## FPS sampling
      seed = farthest_point_sample(seed1_num, points).eval()[0]
      seed_list = seed[:seed1_num]
      print("farthest distance sampling cost", time() - start)
      print("number of patches: %d" % len(seed_list))
      input_list = []
      up_point_list=[]

      patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.patch_num_point)

      for point in tqdm(patches, total=len(patches)):  # tqdm 显示进度条
            up_point = self.patch_prediction(point)
            up_point = np.squeeze(up_point,axis=0)
            input_list.append(point)
            up_point_list.append(up_point)

      return input_list, up_point_list

  def test(self):

      self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
      is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
      Gen = Generator(self.opts, is_training, name='generator')
      self.pred_pc, _ , _ , _= Gen(self.inputs)
      for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
          self.pred_pc, _ , _ , _= Gen(self.pred_pc)

      saver = tf.train.Saver()
      restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
      print(checkpoint_path)
      saver.restore(self.sess, checkpoint_path)

      samples = glob(self.opts.test_data) # 通过 glob 遍历所有测试数据的路径，以 list 形式存储
      point = pc_util.load(samples[0]) # 加载第一个测试样本点云
      self.opts.num_point = point.shape[0] # 测试点云中点的数 ：一般不是 patch (256) 而是整个点云(e.g., 2048)
      out_point_num = int(self.opts.num_point*self.opts.up_ratio)      # 每次只能 上采用 同一尺寸的点云 !!!!  后期需要修改 !!!!!
      for point_path in samples:
          logging.info(point_path) # 打印当前测试点云所在路径
          start = time() # 以秒的形式计算程序运行时间
          pc = pc_util.load(point_path)[:,:3] # 加载测试数据的几何信息 [N, 3]
          pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)  # 把点云归一化， pc 为归一化后的点云

          if self.opts.jitter:
              pc = pc_util.jitter_perturbation_point_cloud(pc[np.newaxis, ...], sigma=self.opts.jitter_sigma,
                                                             clip=self.opts.jitter_max)  #让每个数据点数据进行局部抖动
              pc = pc[0, ...] # 取出当前测试点云

          input_list, pred_list = self.pc_prediction(pc)  # pc ：[N ,3]   input_list ： patch 列表   pred_list： patch 上采样后的列表

          pred_pc = np.concatenate(pred_list, axis=0)
          pred_pc = (pred_pc * furthest_distance) + centroid

          # Graph filter (Optional)
          #batch_num = int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)
          #pred_pc = np.reshape(pred_pc, [batch_num, -1, 3])
          #pred_pc = tf.convert_to_tensor(pred_pc)
          #pred_pc = High_Pass_Graph_Filter0(pred_pc, 7, self.opts.HPGF_dist2, self.opts.HPGF_sigma).eval()

          pred_pc = np.reshape(pred_pc,[-1,3])
          path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
          idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
          pred_pc = pred_pc[idx, 0:3]

          np.savetxt(path[:-4] + '.xyz',pred_pc,fmt='%.6f')

  def log_string(self,msg):
      #global LOG_FOUT
      logging.info(msg)
      self.LOG_FOUT.write(msg + "\n")
      self.LOG_FOUT.flush()











