from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
import numpy as np
import scipy.io as sio
import math

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import Train_Dataset_G2G,Test_Dataset_G2G
from .models import DnCNN, UNet, Generator

class GAN2GAN(object):
    def __init__(self, args, tr_data_dir=None, te_data_dir=None, iter = 1):
        
        self.tr_data_dir = tr_data_dir
        self.te_data_dir = te_data_dir
        self.iter = iter
        self.args = args
        
        if 'Dose' in args.dataset:
            
            if self.iter == 0:
                
                g2_weight_dir = './weights/' + str(args.dataset) + '_g2.w'
                self.denoiser = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'sigmoid')
                self.denoiser.load_state_dict(torch.load(g2_weight_dir))
                self.denoiser.cuda()
                
            else:

                g1_weight_dir = './weights/' + str(args.dataset) + '_g1.w'
                self.g1 = Generator()
                self.g1.load_state_dict(torch.load(g1_weight_dir))
                self.g1.cuda().eval()
                
                self.denoiser = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'linear')
                self.denoiser.cuda()

                g2_weight_dir = './weights/' + str(args.dataset) + '_UNet_iter_'+str(iter-1)+'.w'
                if self.iter == 1:
                    self.g2 = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'sigmoid')
                else:
                    self.g2 = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'linear')
                    
                self.g2.load_state_dict(torch.load(g2_weight_dir))
                self.g2.cuda().eval()
                
                if self.iter != 1:
                    self.denoiser.load_state_dict(torch.load(g2_weight_dir))
            
            
            self.save_file_name = args.dataset + '_UNet_iter_' + str(iter)
            
        else:
            
            if self.iter == 0:
                
                g2_weight_dir = './weights/' + str(args.dataset) + '_g2.w'
                self.denoiser = DnCNN(channels=args.input_channel, num_of_layers=15,output_type='sigmoid')
                self.denoiser.load_state_dict(torch.load(g2_weight_dir))
                self.denoiser.cuda()
                
            else:

                g1_weight_dir = './weights/' + str(args.dataset) + '_g1.w'
                self.g1 = Generator()
                self.g1.load_state_dict(torch.load(g1_weight_dir))
                self.g1.cuda().eval()

                self.denoiser = DnCNN(channels=args.input_channel, num_of_layers=17, output_type='linear')
                self.denoiser.cuda()
                
                g2_weight_dir = './weights/' + str(args.dataset) + '_DnCNN_iter_'+str(iter-1)+'.w'
                
                if self.iter == 1:
                    self.g2 = DnCNN(channels=args.input_channel, num_of_layers=15,output_type='sigmoid')
                else:
                    self.g2 = DnCNN(channels=args.input_channel, num_of_layers=17,output_type='linear')
                    
                self.g2.load_state_dict(torch.load(g2_weight_dir))
                self.g2.cuda().eval()
                
                if self.iter != 1:
                    self.denoiser.load_state_dict(torch.load(g2_weight_dir))
            
            self.save_file_name = args.dataset + '_DnCNN_iter_' + str(iter)
            
            
        self.mini_batch_size = args.mbs
        self.learning_rate = args.lr
        self.epochs = args.ep
        self.drop_ep = args.decay_ep
        self.crop_size = args.patch_size
        
        if self.iter == 0:
            print ('mini_batch_size : ',self.mini_batch_size)
            print ('learning_rate : ',self.learning_rate)
            print ('epochs : ',self.epochs )
            print ('drop_ep : ',self.drop_ep )
            print ('crop_size : ',self.crop_size)
            print ('total_iter : ',args.iter)

        _transforms_tr = [ transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        ]
        
        _transforms_te = [ transforms.ToTensor(),]
        
        self.transform_tr = transforms.Compose(_transforms_tr)
        self.transform_te = transforms.Compose(_transforms_te)
        
        if self.iter != 0:
            tr_dataset = Train_Dataset_G2G(self.tr_data_dir, self.te_data_dir, self.transform_tr, self.g1, self.g2, self.crop_size)
            self.tr_data_loader = DataLoader(tr_dataset, batch_size=self.mini_batch_size, shuffle=True, num_workers=0, drop_last=True)

        te_data_loader = Test_Dataset_G2G(self.te_data_dir, self.transform_te)
        self.te_data_loader = DataLoader(te_data_loader, batch_size=self.mini_batch_size//4, shuffle=False, num_workers=0, drop_last=False)

        self.psnr_arr = []
        self.ssim_arr = []
        self.denoised_img_arr = []
        self.te_loss_arr = []
        self.tr_loss_arr = []
        
        self.Tensor = torch.cuda.FloatTensor
            
        self.optim = torch.optim.Adam(self.denoiser.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.drop_ep, gamma=0.5)

        self.loss = nn.MSELoss()
        self.loss.cuda()
        
        self.best_psnr = 0
        
        
    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = structural_similarity(X, X_hat, data_range=X.max() - X.min())
        
        return test_SSIM
        
    def save_model(self):

        torch.save(self.denoiser.state_dict(), './weights/'+self.save_file_name + '.w')
            
        return
    
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

                source_denoised = np.clip(source_denoised, 0, 1)
                target = np.clip(target, 0, 1)
                    

                # Compute PSRN
                for i in range(source.shape[0]):
                    loss_arr.append(loss)
                    psnr_arr.append(self.get_PSNR(source_denoised[i,0,:,:], target[i,0,:,:]))
                    ssim_arr.append(self.get_SSIM(source_denoised[i,0,:,:], target[i,0,:,:]))
                    denoised_img_arr.append(source_denoised[i,0,:,:])

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        
        if self.args.save_last_ep:
            self.save_model()
            self.denoised_img_arr = denoised_img_arr.copy()
        else:
            if self.best_psnr <= mean_psnr:
                self.best_psnr = mean_psnr
                self.denoised_img_arr = denoised_img_arr.copy()
                self.save_model()
        
        return mean_loss, mean_psnr, mean_ssim
    
    def after_epoch(self, mean_tr_loss, epoch):
        """Tracks and saves starts after each epoch."""

        mean_te_loss, mean_psnr, mean_ssim = self.eval()

        self.psnr_arr.append(mean_psnr)
        self.ssim_arr.append(mean_ssim)
        self.te_loss_arr.append(mean_te_loss)
        self.tr_loss_arr.append(mean_tr_loss)

        if self.args.print_only_final_ep:
            if epoch+1 == self.epochs:
                print ('G2G Iter : ', self.iter, ' | EPOCH {:d} / {:d}'.format(epoch + 1, self.epochs),'| LR : ', self.optim.param_groups[0]['lr'], '| Tr loss : ', round(mean_tr_loss,4), '| Test loss : ', round(mean_te_loss,4), '| PSNR : ', round(mean_psnr,2), '| SSIM : ', round(mean_ssim,4)) 
        else:
            print ('G2G Iter : ', self.iter, ' | EPOCH {:d} / {:d}'.format(epoch + 1, self.epochs),'| LR : ', self.optim.param_groups[0]['lr'], '| Tr loss : ', round(mean_tr_loss,4), '| Test loss : ', round(mean_te_loss,4), '| PSNR : ', round(mean_psnr,2), '| SSIM : ', round(mean_ssim,4)) 
            
    def train(self, args):
        """Trains denoiser on training set."""
        
        if self.iter == 0:
            
            #Iter = 0 : only evaluate g2
            self.after_epoch(0, self.epochs-1)   
            
            sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.tr_loss_arr, 'te_loss_arr':self.te_loss_arr, 
                                                                          'psnr_arr':self.psnr_arr, 'ssim_arr':self.ssim_arr, 'denoised_img_arr':self.denoised_img_arr})
        else:
            
            for epoch in range(self.epochs):
                
                self.denoiser.train()
                tr_loss = []

                for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                    self.optim.zero_grad()

                    source = source.cuda()
                    target = target.cuda()

                    # Denoise image
                    source_denoised = self.denoiser(source)
                    loss = self.loss(source_denoised, target)

                    loss.backward()
                    self.optim.step()

                    tr_loss.append(loss.detach().cpu().numpy())
                    
                    if args.debug:
                        break

                mean_tr_loss = np.mean(tr_loss)
                self.after_epoch(mean_tr_loss, epoch)    

                sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.tr_loss_arr, 'te_loss_arr':self.te_loss_arr, 
                                                              'psnr_arr':self.psnr_arr, 'ssim_arr':self.ssim_arr, 'denoised_img_arr':self.denoised_img_arr})
                
                if args.debug:
                    break
                


