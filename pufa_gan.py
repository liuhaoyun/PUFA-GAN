import tensorflow as tf
from Upsampling.model import Model
from Upsampling.configs import FLAGS
from datetime import datetime
import os
import logging
import pprint
import h5py
pp = pprint.PrettyPrinter()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def run():
    if FLAGS.phase=='train':
        FLAGS.train_file = os.path.join(FLAGS.data_dir, 'train/PUGAN_poisson_256_poisson_1024.h5')  # 拼接得到训练数据路径
        print('train_file:',FLAGS.train_file)  # 训练文件路径
        if not FLAGS.restore:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")  # Y-year m-month d-day H-hour M-minute，把时间以 string 的方式 输出
            FLAGS.log_dir = os.path.join(FLAGS.log_dir,current_time)
            try:
                os.makedirs(FLAGS.log_dir) # 创建多级目录
            except os.error:
                pass
    else:
        FLAGS.log_dir = os.path.join(os.getcwd(),'model')  # os.getcwd() ：获取当前目录
        FLAGS.test_data = os.path.join(FLAGS.data_dir, 'test/*.xyz')
        FLAGS.out_folder = os.path.join(FLAGS.data_dir,'test/output')
        if not os.path.exists(FLAGS.out_folder):
            os.makedirs(FLAGS.out_folder)
        print('test_data:',FLAGS.test_data)

    print('checkpoints:',FLAGS.log_dir)
    pp.pprint(FLAGS)
    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True  # 允许动态地增加GPU运行内存，不会限制GPU的内存
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS,sess)
        if FLAGS.phase == 'train':
            model.train()
        else:
            model.test()
        # 计算 FLOPs
        #stats_graph(tf.get_default_graph())


def main(unused_argv):
  run()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)  # 调整 logging 日志 输出级别为 INFO
  tf.app.run()

