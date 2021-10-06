from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

import numpy as np
import scipy.io as sio
import math

from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Train_Dataset_G2G,Test_Dataset_G2G
from .models import DnCNN, UNet

class Test_Denoiser(object):
    def __init__(self, args, te_data_dir=None, denoiser_dir=None):
        
        self.te_data_dir = te_data_dir
        self.denoiser_dir = denoiser_dir
        self.args = args
        
        if args.dncnn_b:
            # Load DnCNN-B
            self.denoiser = DnCNN(channels=args.input_channel, num_of_layers=20, output_type='linear')
            
        else:

            if 'Dose' in args.dataset:

                self.denoiser = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'linear')

            else:

                self.denoiser = DnCNN(channels=args.input_channel, num_of_layers=17, output_type='linear')
        
        self.denoiser_dir = './pretrained_weights/' + self.denoiser_dir
        self.denoiser.load_state_dict(torch.load(self.denoiser_dir))
        self.denoiser.cuda()
            
        self.mini_batch_size = args.mbs
        
        _transforms_te = [ transforms.ToTensor(),]
        
        self.transform_te = transforms.Compose(_transforms_te)
        
        te_data_loader = Test_Dataset_G2G(self.te_data_dir, self.transform_te)
        self.te_data_loader = DataLoader(te_data_loader, batch_size=self.mini_batch_size, shuffle=False, num_workers=0, drop_last=False)

        self.loss = nn.MSELoss()
        self.loss.cuda()
        
    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = structural_similarity(X, X_hat, data_range=X.max() - X.min())
        
        return test_SSIM
            
    def eval(self):
        """Evaluates denoiser on validation set."""

        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        denoised_img_arr = []
        
        self.denoiser.eval()

        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):
                
                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                source_denoised = self.denoiser(source)

                # Update loss
                loss = self.loss(source_denoised, target)
                loss = loss.cpu().numpy()

                target = target.cpu().numpy()
                source_denoised = source_denoised.cpu().numpy()

                target = np.clip(target, 0, 1)
                source_denoised = np.clip(source_denoised, 0, 1)
                    
                # Compute PSRN
                for i in range(source.shape[0]):
                    loss_arr.append(loss)
                    psnr_arr.append(self.get_PSNR(source_denoised[i,0,:,:], target[i,0,:,:]))
                    ssim_arr.append(self.get_SSIM(source_denoised[i,0,:,:], target[i,0,:,:]))
                    denoised_img_arr.append(source_denoised[i,0,:,:])

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        
        return mean_loss, mean_psnr, mean_ssim
    