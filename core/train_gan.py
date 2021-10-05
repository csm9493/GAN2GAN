import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import math
import numpy as np
import math
import scipy.io as sio

from .utils import Dataset_GAN, AverageMeter, ProgressMeter, LambdaLR
from .models import Critic, Generator, DnCNN, weights_init_normal, UNet

class Train_Generative_Models():
    
    def __init__(self, args, tr_data_z, tr_data_n):
        
        self.g1 = Generator()
        self.c1 = Critic(args.input_channel)
        self.c2 = Critic(args.input_channel)
        
        if 'WF' in args.dataset:
            print ('use DnCNN for g1 and g3')
            self.g2 = DnCNN(channels=1, num_of_layers=15,output_type='sigmoid')
            self.g3 = DnCNN(channels=1,output_type='linear')
        else:
            print ('use UNet for g1 and g3')
            self.g2 = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'sigmoid', residual = True)
            self.g3 = UNet(in_channels=args.input_channel,out_channels=args.input_channel, output_activation = 'linear', residual = False)
        
        self.criterion_L1 = torch.nn.L1Loss()
        
        self.g1.apply(weights_init_normal)
        self.g2.apply(weights_init_normal)
        self.g3.apply(weights_init_normal)
        self.c1.apply(weights_init_normal)
        self.c2.apply(weights_init_normal)
        
        self.save_weight_name = './weights/' + str(args.dataset) 
        self.save_file_name = './result_data/' + str(args.dataset) + '_GAN_result.mat'
        self.tr_data_z = './data/' + tr_data_z
        self.tr_data_n = './data/' + tr_data_n
        
        self.n_epochs = args.ep
        self.critic_iter = args.c_iter
        self.mini_batch_size = args.mbs
        
        self.g_loss1_alpha = args.alpha
        self.g_loss2_beta = args.beta
        self.cycle_loss_gamma = args.gamma

        self.weight_clip = args.weight_clip
        self.patch_size = args.patch_size
        
        # Optimizers & LR schedulers
        self.optimizer_g1 = torch.optim.Adam(self.g1.parameters(),lr=args.lr_g, betas=(0.5, 0.999))
        self.lr_scheduler_g1 = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g1, lr_lambda=LambdaLR(self.n_epochs, 0, args.decay_ep).step)
        self.optimizer_g2 = torch.optim.Adam(self.g2.parameters(),lr=args.lr_g, betas=(0.5, 0.999))
        self.lr_scheduler_g2 = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g2, lr_lambda=LambdaLR(self.n_epochs, 0, args.decay_ep).step)
        self.optimizer_g3 = torch.optim.Adam(self.g3.parameters(),lr=args.lr_g, betas=(0.5, 0.999))
        self.lr_scheduler_g3 = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g3, lr_lambda=LambdaLR(self.n_epochs, 0, args.decay_ep).step)

        self.optimizer_c1 = torch.optim.RMSprop(self.c1.parameters(), lr=args.lr_c)
        self.optimizer_c2 = torch.optim.RMSprop(self.c2.parameters(), lr=args.lr_c)

        self.g1.cuda()
        self.g2.cuda()
        self.g3.cuda()
        self.c1.cuda()
        self.c2.cuda()

        train_dataset = Train_Dataset_GAN(path_zi=self.tr_data_z,path_ni=self.tr_data_n, patch_size=self.patch_size,)
        self.dataloader = DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        # Inputs & targets memory allocation
        self.Tensor = torch.cuda.FloatTensor
        self.input_A = self.Tensor(self.mini_batch_size, args.input_channel, self.patch_size, self.patch_size)
        self.input_B = self.Tensor(self.mini_batch_size, args.input_channel, self.patch_size, self.patch_size)
        self.noise = self.Tensor(self.mini_batch_size, 128,1,1)
        
        print ('save_weight_name : ', self.save_weight_name)
        print ('save_file_name : ', self.save_file_name)
        print ('tr_data_file_name_zi : ', self.tr_data_z)
        print ('tr_data_file_name_ni : ', self.tr_data_n)
        print ('alpha : ', self.g_loss1_alpha)
        print ('beta : ', self.g_loss2_beta)
        print ('gamma : ', self.cycle_loss_gamma)
        print ('lr_g : ', args.lr_g)
        print ('lr_critic : ', args.lr_c)
        print ('weight_clip : ', self.weight_clip)
        
    def train(self, args):

        W_c1_arr = []
        W_c2_arr = []
        L_g1_arr = []
        L_g2_arr = []
        L_cycle_arr = []
        PSNR_arr = []
        mean_n_arr = []
        std_n_arr = []
        mean_n_hat_arr = []
        std_n_hat_arr = []
        
        for epoch in range(self.n_epochs):

            W_c1 = AverageMeter('W_c1', ':.2f')
            W_c2 = AverageMeter('W_c2', ':.2f')
            L_g1 = AverageMeter('L_g1', ':.2f')
            L_g2 = AverageMeter('L_g2', ':.2f')
            L_cycle = AverageMeter('L_cycle', ':.2f')
            PSNR = AverageMeter('PSNR', ':.2f')
            mean_n = AverageMeter('mean_n', ':.2f')
            std_n = AverageMeter('std_n', ':.2f')
            mean_n_hat = AverageMeter('mean_n_hat', ':.2f')
            std_n_hat = AverageMeter('std_n_hat', ':.2f')

            progress = ProgressMeter(
                len(self.dataloader),
                [W_c1, W_c2, L_g1, L_g2, L_cycle, PSNR, mean_n, std_n, mean_n_hat, std_n_hat],
                prefix="Epoch: [{}]".format(epoch))

            for i, (patch, clean, noise) in enumerate(self.dataloader):
                # Set model input

                patch = Variable(self.input_A.copy_(patch))
                noise = Variable(self.input_B.copy_(noise))
                
                self.noise.resize_(self.mini_batch_size, 128).normal_(0, 1)
                copied_noise = self.noise.clone()
                noisev_g1 = Variable(self.noise)
                noisev_g2 = Variable(copied_noise.resize_(self.mini_batch_size, 128,1,1))

                for d_iter in range(self.critic_iter):
                    # Train discriminator
                    self.optimizer_c1.zero_grad()

                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    c1_loss_real = self.c1(noise)
                    c1_loss_real = c1_loss_real.mean()

                    # Train with fake images
                    n_hat = self.g1(noisev_g2)
                    c1_loss_fake = self.c1(n_hat)
                    c1_loss_fake = c1_loss_fake.mean()

                    c_loss = -(c1_loss_real - c1_loss_fake)
                    Wasserstein_C1 = c1_loss_real - c1_loss_fake
                    c_loss.backward()
                    self.optimizer_c1.step()

                    for p in self.c1.parameters():
                        p.data.clamp_(-self.weight_clip, self.weight_clip)

                for d_iter in range(self.critic_iter):
                    # Train discriminator
                    self.optimizer_c2.zero_grad()

                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    c2_loss_real = self.c2(patch)
                    c2_loss_real = c2_loss_real.mean()

                    # Train with fake images

                    x_hat = self.g2(patch)
                    n_hat = self.g1(noisev_g2)
                    c2_loss_fake = self.c2(x_hat+n_hat)
                    c2_loss_fake = c2_loss_fake.mean()

                    c_loss = -(c2_loss_real -c2_loss_fake)
                    Wasserstein_C2 = c2_loss_real - c2_loss_fake
                    c_loss.backward()
                    self.optimizer_c2.step()

                    for p in self.c2.parameters():
                        p.data.clamp_(-self.weight_clip, self.weight_clip)
                        
                ###### Generators g1, g2 and g3 ######
                self.optimizer_g1.zero_grad()
                self.optimizer_g2.zero_grad()
                
                # GAN loss
                n_hat = self.g1(noisev_g2)
                g_loss1 = self.c1(n_hat)
                g_loss1 = -g_loss1.mean() 

                x_hat = self.g2(patch)
                z_hat = x_hat+n_hat

                g_loss2 = self.c2(z_hat)
                g_loss2 = -g_loss2.mean() 

                z_tilde = self.g3(x_hat)

                cycleloss = self.criterion_L1(patch, z_tilde) 

                # Total loss
                loss_total = g_loss1* self.g_loss1_alpha + g_loss2 * self.g_loss2_beta + cycleloss * self.cycle_loss_gamma
                loss_total.backward()
                
                self.optimizer_g1.step()
                self.optimizer_g2.step()
                self.optimizer_g3.step()

                W_c1.update(Wasserstein_C1.item(), patch.size(0))
                W_c2.update(Wasserstein_C2.item(), patch.size(0))
        
                L_g1.update(g_loss1.item(), patch.size(0))
                L_g2.update(g_loss2.item(), patch.size(0))
                L_cycle.update(cycleloss.item(), patch.size(0))
                
                mean_n_hat.update((n_hat).mean().item()*255, patch.size(0))
                std_n_hat.update((n_hat).std().item()*255, patch.size(0))
                
                mean_n.update((noise).mean().item()*255, patch.size(0))
                std_n.update((noise).std().item()*255, patch.size(0))

                mse_x_hat_x = ((clean.cuda()-x_hat)**2).mean().item()
                PSNR.update(10 * math.log10(1/mse_x_hat_x), patch.size(0))
                
                if i % args.print_freq == 0:
                    progress.display(i)
                    
                if args.debug:
                    break
                
            # Update learning rates
            self.lr_scheduler_g1.step(epoch)
            self.lr_scheduler_g2.step(epoch)
            self.lr_scheduler_g3.step(epoch)

            #save generator netG_A2B
            if args.save_only_final_weights:
                torch.save(self.g1.state_dict(), self.save_weight_name +'_g1.w')
                torch.save(self.g2.state_dict(), self.save_weight_name +'_g2.w')
            else:
                torch.save(self.g1.state_dict(), self.save_weight_name +'_g1.w')
                torch.save(self.g2.state_dict(), self.save_weight_name +'_g2.w')
                torch.save(self.g1.state_dict(), self.save_weight_name +'_g1_ep' + str(epoch+1) + '.w')
                torch.save(self.g2.state_dict(), self.save_weight_name +'_g2_ep'+ str(epoch+1)  + '.w')
                
            #save results
            W_c1_arr.append(W_c1.avg)
            W_c2_arr.append(W_c2.avg)
            L_g1_arr.append(L_g1.avg)
            L_g2_arr.append(L_g2.avg)
            L_cycle_arr.append(L_cycle.avg)
            PSNR_arr.append(PSNR.avg)
            mean_n_arr.append(mean_n.avg)
            std_n_arr.append(std_n.avg)
            mean_n_hat_arr.append(mean_n_hat.avg)
            std_n_hat_arr.append(std_n_hat.avg)
            
            sio.savemat(self.save_file_name, {"W_c1_arr": np.array(W_c1_arr), "W_c2_arr": np.array(W_c2_arr), "L_g1_arr": np.array(L_g1_arr), "L_g2_arr": np.array(L_g2_arr), "L_cycle_arr": np.array(L_cycle_arr), 
                                             "PSNR_arr": np.array(PSNR_arr), "mean_n_arr": np.array(mean_n_arr), "std_n_arr": np.array(std_n_arr), "mean_n_hat_arr": np.array(mean_n_hat_arr), "std_n_hat_arr": np.array(std_n_hat_arr), })
            