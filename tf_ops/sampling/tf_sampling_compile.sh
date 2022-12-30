#!/usr/bin/env bash
#/bin/bash

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NSYNC_INC=$TF_INC/external/nsync/public
NVCC=/usr/local/cuda-10.0/bin/nvcc
cudainc=/usr/local/cuda-10.0/include
cudalib=/usr/local/cuda-10.0/lib64/

$NVCC tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /data/lirh/anaconda3/envs/tensorflow3/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC  -I $cudainc -I $NSYNC_INC -lcudart -L $cudalib -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
