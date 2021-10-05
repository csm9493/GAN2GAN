import argparse
from core.train_gan2gan import GAN2GAN

import torch
import numpy as np
import random
       
parser = argparse.ArgumentParser(description='Training GAN2GAN')
parser.add_argument('--dataset', type=str, choices=['WF_avg1','WF_avg2','WF_avg4','WF_avg8','WF_avg16', 
                                                    'Dose25', 'Dose50', 'Dose75', 'Dose100'], required=True)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--ep', default=50, type=int)
parser.add_argument('--decay-ep', default=10, type=int)
parser.add_argument('--mbs', default=64, type=int)
parser.add_argument('--iter', default=3, type=int)
parser.add_argument('--patch-size', default=64, type=int)
parser.add_argument('--input-channel', default=1, type=int)
parser.add_argument('--print-only-final-ep', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

if 'Dose' in args.dataset:

    tr_data = 'G2G_train_120x120_Medical_dataset_'+str(args.dataset)+'.hdf5'
    te_data = 'Medical_dataset_'+str(args.dataset)+'_test.hdf5'
        
elif 'WF' in args.dataset:
    
    tr_data = 'G2G_train_120x120_FMD_'+str(args.dataset)+'.hdf5'
    te_data = 'FMD_'+str(args.dataset)+'_test.hdf5'
    
    
for i in range(args.iter+1):
    train_gan = GAN2GAN(args, tr_data, te_data, i)
    train_gan.train(args)


















