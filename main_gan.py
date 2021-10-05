import argparse
from core.train_gan import Train_Generative_Models

import torch
import numpy as np
import random
       
parser = argparse.ArgumentParser(description='Training Generative Models')
parser.add_argument('--dataset', type=str, choices=['WF_avg1','WF_avg2','WF_avg4','WF_avg8','WF_avg16', 
                                                    'Dose25', 'Dose50', 'Dose75', 'Dose100'], required=True)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lr-g', default=4e-4, type=float)
parser.add_argument('--lr-c', default=5e-5, type=float)
parser.add_argument('--alpha', default=5, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--gamma', default=10, type=float)
parser.add_argument('--weight-clip', default=10, type=float)
parser.add_argument('--ep', default=30, type=int)
parser.add_argument('--decay-ep', default=10, type=int)
parser.add_argument('--c-iter', default=5, type=int)
parser.add_argument('--mbs', default=64, type=int)
parser.add_argument('--input-channel', default=1, type=int)
parser.add_argument('--patch-size', default=64, type=int)
parser.add_argument('-p', '--print-freq', default=1, type=int)
parser.add_argument('--save-only-final-weights', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

if 'Dose' in args.dataset:

    tr_data_z = 'GAN_train_96x96_Medical_dataset_'+str(args.dataset)+'_noisy_clean_patches.hdf5'
    tr_data_n = 'GAN_train_96x96_Medical_dataset_'+str(args.dataset)+'_noise_patches.hdf5'
        
elif 'WF' in args.dataset:

    tr_data_z = 'GAN_train_96x96_FMD_'+str(args.dataset)+'_noisy_clean_patches.hdf5'
    tr_data_n = 'GAN_train_96x96_FMD_'+str(args.dataset)+'_noise_patches.hdf5'
    
    
train_gan = Train_Generative_Models(args, tr_data_z, tr_data_n)
train_gan.train(args)


















