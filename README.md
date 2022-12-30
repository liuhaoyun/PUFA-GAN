# PUFA-GAN
"PUFA-GAN: A Frequency-Aware Generative Adversarial Network for 3D Point Cloud Upsampling" is accepted by IEEE TIP 2022.

A. Introduction

This repository is for our IEEE TIP 2022 paper 'PUFA-GAN: A Frequency-Aware Generative Adversarial Network for 3D Point Cloud Upsampling'. The code is modified from PU-GAN.

B. Installation

This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators.
For installing tensorflow, please follow the official instructions in here. The code is tested under TF1.11 (higher version should also work) and Python 3.6 on Ubuntu 18.04.
For compiling TF operators, please check tf_xxx_compile.sh under each op subfolder in code/tf_ops folder. Note that you need to update nvcc, python and tensoflow include library if necessary.

C. Usage

1. Compile the TF operators Follow the above information to compile the TF operators.

2. Train the model: First, you need to download the training patches in HDF5 format from https://drive.google.com/file/d/13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ/view?usp=drive_open and put it in folder data/train. Then run:

  python pufa_gan.py --phase train

3. Evaluate the model: First, you need to download the pretrained model from /log (automatically generated during training), extract it and put it in folder 'model'. Then run:

  python pufa_gan.py --phase test

Note that our pretrained model has been put into /model.

4. Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer https://www.cgal.org/download/linux.html and PU-Net to install this library. 
Then run: 

  cd evaluation_code

  cmake .

  make

  ./evaluation file_name.off file_name.xyz

The second argument is the mesh, and the third one is the predicted points.

D. Cite:

   If PUFA-GAN is useful for your research, please consider citing:
   
@ARTICLE{9961237,
  author={Liu, Hao and Yuan, Hui and Hou, Junhui and Hamzaoui, Raouf and Gao, Wei},
  journal={IEEE Transactions on Image Processing}, 
  title={PUFA-GAN: A Frequency-Aware Generative Adversarial Network for 3D Point Cloud Upsampling}, 
  year={2022},
  volume={31},
  number={},
  pages={7389-7402},
  doi={10.1109/TIP.2022.3222918}}
  
E. Questions

  Please contact 'liuhaoxb@gmail.com'
  
F. Thanks 

  Thank PU-GAN for the source code support.
