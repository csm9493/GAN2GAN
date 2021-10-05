#! /bin/bash
GPU=0

# # Train Generative Models for FMD WF dataset
DATASET='WF_avg1'           #['WF_avg1','WF_avg2','WF_avg4','WF_avg8', WF_avg16]
CUDA_VISIBLE_DEIVCES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEIVCES=$GPU python main_gan2gan.py --dataset $DATASET --print-only-final-ep

# Train Generative Models for Medical dataset
DATASET='Dose25'   #['Dose25','Dose50','Dose75','Dose100'
CUDA_VISIBLE_DEIVCES=$GPU python main_gan.py --dataset $DATASET --save-only-final-weights
CUDA_VISIBLE_DEIVCES=$GPU python main_gan2gan.py --dataset $DATASET --print-only-final-ep

