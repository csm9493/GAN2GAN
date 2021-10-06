#! /bin/bash
GPU=1

#Evaluate pre-trained GAN2GAN denoisers for Gaussian noise
DATASET='Gaussian_std15'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Gaussian_std25'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Gaussian_std30'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Gaussian_std50'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 

#Evaluate pre-trained GAN2GAN denoisers for Mixture noise
DATASET='Mixture_s15'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Mixture_s25'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='Mixture_s30'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Mixture_s50'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET

#Evaluate pre-trained GAN2GAN denoisers for Correlated noise
DATASET='Correlated_std15'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='Correlated_std25'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 

#Evaluate pre-trained GAN2GAN denoisers for Medical dataset
DATASET='Dose25'          
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='Dose50'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Dose75'          
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET 
DATASET='Dose100'          
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET

#Evaluate pre-trained GAN2GAN denoisers for WF dataset
DATASET='WF_avg1'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='WF_avg2'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='WF_avg4'          
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='WF_avg8'           
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET
DATASET='WF_avg16'          
CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET


#Evaluate pre-trained DnCNN-B denoisers

# DATASET='Gaussian_std15'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Gaussian_std25'          
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Gaussian_std30'          
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Gaussian_std50'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b

# DATASET='Mixture_s15'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Mixture_s25'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Mixture_s30'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Mixture_s50'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b

# DATASET='Correlated_std15'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Correlated_std25'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b


# DATASET='Dose25'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Dose50'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Dose75'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='Dose100'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b

# DATASET='WF_avg1'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='WF_avg2'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='WF_avg4'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='WF_avg8'          
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
# DATASET='WF_avg16'           
# CUDA_VISIBLE_DEVICES=$GPU python main_test.py --dataset $DATASET --dncnn-b
