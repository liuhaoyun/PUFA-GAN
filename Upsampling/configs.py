import argparse
import os

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train',help="train/test")
parser.add_argument('--log_dir', default='log')
parser.add_argument('--data_dir', default='data')
parser.add_argument('--augment', type=str2bool, default=True)
parser.add_argument('--restore', action='store_true')   #action='store_true'作用，命令行中，显式地打出 --resotre 就为True，不打出 --resotre 就为 False
parser.add_argument('--more_up', type=int, default=2)
parser.add_argument('--training_epoch', type=int, default=131)
parser.add_argument('--batch_size', type=int, default=28)
parser.add_argument('--use_non_uniform', type=str2bool, default=True)
parser.add_argument('--jitter', type=str2bool, default=False)
parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
parser.add_argument('--up_ratio', type=int, default=4)
parser.add_argument('--num_point', type=int, default=256)
parser.add_argument('--patch_num_point', type=int, default=256)
parser.add_argument('--patch_num_ratio', type=int, default=3)
parser.add_argument('--base_lr_d', type=float, default=0.00006)
parser.add_argument('--base_lr_g', type=float, default=0.00009)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--start_decay_step', type=int, default=50000)
parser.add_argument('--lr_decay_steps', type=int, default=50000)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1.0e-6)  #  0.0000035
parser.add_argument('--steps_per_print', type=int, default=1)
parser.add_argument('--visulize', type=str2bool, default=False) # 默认是假
parser.add_argument('--steps_per_visu', type=int, default=100)
parser.add_argument('--epoch_per_save', type=int, default=5)
parser.add_argument('--use_repulse', type=str2bool, default=False)
parser.add_argument('--repulsion_w', default=1.0, type=float, help="repulsion_weight")  # repulsion_loss
parser.add_argument('--fidelity_w', default=100.0, type=float, help="fidelity_weight")  # CD_loss
parser.add_argument('--cycle_w', default=1.0, type=float, help="cycle_weight")  # cycle loss
parser.add_argument('--cosine_w', default=100.0, type=float, help="cosine_weight")  # cosine_loss
parser.add_argument('--edge_w', default=1.0, type=float, help="edge_weight")  # edge_loss
parser.add_argument('--uniform_w', default=10.0, type=float, help="uniform_weight")  # uniform_loss
parser.add_argument('--gan_w', default=0.5, type=float, help="gan_weight")  # gen_loss
parser.add_argument('--global_w', default=0.95, type=float, help="gan_weight")  # global weight for Discriminator
parser.add_argument('--gen_update', default=2, type=int, help="gen_update")
parser.add_argument('--HPGF_Sample_num_point', default=256, type=int, help="HPGF_Sample_num_point")  #  High Pass Graph Filter （HPGF）
parser.add_argument('--HPGF_dist', default=0.5, type=float, help="HPGF_dist")
parser.add_argument('--HPGF_dist2', default=0.001, type=float, help="HPGF_dist2")
parser.add_argument('--HPGF_sigma', default=2.0, type=float, help="HPGF_sigma")


FLAGS = parser.parse_args()

