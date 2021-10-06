import argparse
from core.test_denoiser import Test_Denoiser

import torch
import numpy as np
import random
       
parser = argparse.ArgumentParser(description='Training GAN2GAN')
parser.add_argument('--dataset', type=str, choices=['WF_avg1','WF_avg2','WF_avg4','WF_avg8','WF_avg16', 
                                                    'Dose25', 'Dose50', 'Dose75', 'Dose100',
                                                   'Gaussian_std15', 'Gaussian_std25', 'Gaussian_std30', 'Gaussian_std50',
                                                   'Mixture_s15', 'Mixture_s25', 'Mixture_s30', 'Mixture_s50',
                                                   'Correlated_std15', 'Correlated_std25'], required=True)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mbs', default=16, type=int)
parser.add_argument('--input-channel', default=1, type=int)
parser.add_argument('--dncnn-b', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

if 'Dose' in args.dataset:

    te_data = 'Medical_dataset_'+str(args.dataset)+'_test.hdf5'
    
    if args.dncnn_b:
        denoiser_path = 'BSD_DnCNN_B_N2C.w'
    else:
        denoiser_path = 'Medical_dataset_'+str(args.dataset)+'_UNet_iter_3.w'
        
elif 'WF' in args.dataset:
    
    te_data = 'FMD_'+str(args.dataset)+'_test.hdf5'
    
    if args.dncnn_b:
        denoiser_path = 'BSD_DnCNN_B_N2C.w'
    else:
        denoiser_path = 'FMD_'+str(args.dataset)+'_DnCNN_iter_7.w'
        
elif 'Gaussian' in args.dataset:
    
    te_data = 'BSD_'+str(args.dataset)+'_test.hdf5'
    
    if args.dncnn_b:
        denoiser_path = 'BSD_DnCNN_B_N2C.w'
    else:
        denoiser_path = 'BSD_'+str(args.dataset)+'_DnCNN_iter_3.w'
        
elif 'Correlated' in args.dataset:
    
    te_data = 'BSD_'+str(args.dataset)+'_test.hdf5'
    
    if args.dncnn_b:
        denoiser_path = 'BSD_DnCNN_B_N2C.w'
    else:
        denoiser_path = 'BSD_'+str(args.dataset)+'_DnCNN_iter_3.w'
        
elif 'Mixture' in args.dataset:
    
    te_data = 'BSD_'+str(args.dataset)+'_test.hdf5'
    
    if args.dncnn_b:
        denoiser_path = 'BSD_DnCNN_B_N2C.w'
    else:
        denoiser_path = 'BSD_'+str(args.dataset)+'_DnCNN_iter_3.w'

train_gan = Test_Denoiser(args, te_data, denoiser_path)
te_loss, psnr, ssim = train_gan.eval()

print('test data: ', te_data, '\t| denoiser : ', denoiser_path, '\t| PSNR: ',psnr,'\t| SSIM: ',ssim)



















