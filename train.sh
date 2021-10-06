#! /bin/bash
GPU=0

# # # Train Generative Models for FMD WF dataset
# DATASET='WF_avg1'           #['WF_avg1','WF_avg2','WF_avg4','WF_avg8', WF_avg16]
# # CUDA_VISIBLE_DEVICES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
# CUDA_VISIBLE_DEVICES=$GPU python main_gan2gan.py --dataset $DATASET --mbs 128 --iter 3 --print-only-final-ep

# Train Generative Models for Medical dataset
DATASET='Dose25'   #['Dose25','Dose50','Dose75','Dose100']
CUDA_VISIBLE_DEVICES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEVICES=$GPU python main_gan2gan.py --dataset $DATASET --mbs 4 --iter 3 --print-only-final-ep

# Train Generative Models for Medical dataset
DATASET='Gaussian_std15'   #['Gaussian_std15','Gaussian_std25','Gaussian_std30','Gaussian_std50']
CUDA_VISIBLE_DEVICES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEVICES=$GPU python main_gan2gan.py --dataset $DATASET --mbs 4 --iter 3 --print-only-final-ep

# Train Generative Models for Medical dataset
DATASET='Mixture_std15'   #['Mixture_std15','Mixture_std25', 'Mixture_std30', 'Mixture_std50']
CUDA_VISIBLE_DEVICES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEVICES=$GPU python main_gan2gan.py --dataset $DATASET --mbs 4 --iter 3 --print-only-final-ep

# Train Generative Models for Medical dataset
DATASET='Correlated_std15'   #['Correlated_std15','Correlated_std25']
CUDA_VISIBLE_DEVICES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEVICES=$GPU python main_gan2gan.py --dataset $DATASET --mbs 4 --iter 3 --print-only-final-ep




