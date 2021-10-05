from visdom import Visdom
import sys
import random
import time
import datetime
import numpy as np
import scipy.io as sio

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py
import random
import torch
import pywt
from sklearn.feature_extraction import image

class Train_Dataset_GAN(Dataset):
    def __init__(self, path_zi=None,path_ni=None, patch_size=64):
        
        self.patch_size = patch_size
        self.data_name = path_zi

        fd_zi = h5py.File(path_zi, "r")
        fd_ni = h5py.File(path_ni, "r")
        self.z_i = (fd_zi["noisy_patches"])
        self.x_i = (fd_zi["clean_patches"])
        self.n_i = (fd_ni["noise_patches"])
        
        print ('num of noisy dataset : ', self.z_i.shape[0])
        print ('num of noise dataset : ', self.n_i.shape[0])

        self.num_data = min(self.n_i.shape[0], self.z_i.shape[0])
        
    def __len__(self):
        
        return self.num_data

    def __getitem__(self, idx):
        
        if 'Dose' in self.data_name:
            patch = Image.fromarray((self.z_i[idx,:,:] / 0.4))
            noise = Image.fromarray((self.n_i[idx,:,:] / 0.4))
            clean = Image.fromarray((self.x_i[idx,:,:] / 0.4))

        else:
            patch = Image.fromarray((self.z_i[idx,:,:]))
            noise = Image.fromarray((self.n_i[idx,:,:]))
            clean = Image.fromarray((self.x_i[idx,:,:]))
            
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            patch, output_size=(self.patch_size, self.patch_size))
        patch = tvF.crop(patch, i, j, h, w)
        noise = tvF.crop(noise, i, j, h, w)
        clean = tvF.crop(clean, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            patch = tvF.hflip(patch)
            noise = tvF.hflip(noise)
            clean = tvF.hflip(clean)

        # Random vertical flipping
        if random.random() > 0.5:
            patch = tvF.vflip(patch)
            noise = tvF.vflip(noise)
            clean = tvF.vflip(clean)

        patch = tvF.to_tensor(patch)
        noise = tvF.to_tensor(noise)
        noise = (noise - torch.mean(noise))
        clean = tvF.to_tensor(clean)

        return patch, clean, noise
    

class Train_Dataset_G2G():

    def __init__(self,_tr_data_dir=None, _te_data_dir = None, _transform=None, _g1 = None, _g2 = None, _crop_size = 64):

        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.transform = _transform
        self.crop_size = _crop_size
        
        self.G1 = _g1
        self.G2 = _g2

        self.data = h5py.File('./data/' + self.tr_data_dir, "r")
        self.clean_arr = self.data["clean_patches"]
        self.noisy_arr = self.data["noisy_patches"]
        self.num_data = self.data["clean_patches"].shape[0]

        self.Tensor = torch.cuda.FloatTensor
               
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        if 'Dose' in self.tr_data_dir:
            noisy_img = Image.fromarray((self.noisy_arr[index,:,:] / 0.4))
            clean_img = Image.fromarray((self.clean_arr[index,:,:] / 0.4))
        else:
            noisy_img = Image.fromarray((self.noisy_arr[index,:,:]))
            clean_img = Image.fromarray((self.clean_arr[index,:,:]))

        if self.transform:
            
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(noisy_img, output_size=(self.crop_size, self.crop_size))
            noisy_img = tvF.crop(noisy_img, i, j, h, w)
            clean_img = tvF.crop(clean_img, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                clean_img = tvF.hflip(clean_img)
                noisy_img = tvF.hflip(noisy_img)

            # Random vertical flipping
            if random.random() > 0.5:
                clean_img = tvF.vflip(clean_img)
                noisy_img = tvF.vflip(noisy_img)

            clean_img = tvF.to_tensor(clean_img)
            noisy_img = tvF.to_tensor(noisy_img)
            
            with torch.no_grad():

                noise = self.Tensor(2, 128, 1, 1).normal_(0, 1)
                noise = Variable(noise)

                input_tensor = self.Tensor(2, 1, self.crop_size, self.crop_size)
                noisy_inputs = Variable(input_tensor.copy_(noisy_img.view(1,1,self.crop_size,self.crop_size)))

                n_hat = self.G1(noise)
                x_hat = self.G2(noisy_inputs)

                z_hat = x_hat + n_hat

                source = z_hat[0].view(1,self.crop_size,self.crop_size).detach()
                target = z_hat[1].view(1,self.crop_size,self.crop_size).detach()

            return source, target



class Test_Dataset_G2G():

    def __init__(self,_tedata_dir=None, _transform=None):

        self.te_data_dir = _tedata_dir
        self.transform = _transform

        self.data =  h5py.File('./data/' + self.te_data_dir, "r")
        self.clean_arr = self.data["clean_images"]
        self.noisy_arr = self.data["noisy_images"]
        
        self.num_data = self.data["clean_images"].shape[0]
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        
        # Load PIL image
        
        if 'Dose' in   self.te_data_dir:
            source = Image.fromarray((np.clip(self.clean_arr[index,:,:]/0.4, 0, 2)))
            target = Image.fromarray((np.clip(self.noisy_arr[index,:,:]/0.4, 0, 2)))
        else:
            source = Image.fromarray((self.clean_arr[index,:,:]))
            target = Image.fromarray((self.noisy_arr[index,:,:]))
            
        if self.transform:
            source = self.transform(source)
            
        target = tvF.to_tensor(target)
    
        return source, target

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    


