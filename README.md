# GAN2GAN

The official code of GAN2GAN: Generative Noise Learning for Blind Denoising with Single Noisy Images (ICLR 2021) [[arxiv]](https://arxiv.org/abs/1905.10488)

## Quick Start

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### 1) Download FMD(FW)[1] and Medical dataset from [[this google drive link]]()

#### 2) Locate 'Medical_images' and 'Real_FM_images' to './data' directory

```
./data
      /Medical_images 
      /Real_FM_images 
      ...
```
#### 3) Generate FMD WF dataset for training GAN and GAN2GAN
##### 3-1) Run below ipython files sequentically (you can choice 'data_type' in each ipython file)

```
./data
      /Generate_FMD_WF_train_dataset.ipynb 
      /Generate_FMD_WF_GAN_train_dataset_96x96.ipynb 
      /Generate_FMD_WF_G2G_train_dataset_120x120.ipynb
      /Generate_FMD_WF_test_dataset.ipynb
```

#### 4) Generate Medical dataset for training GAN and GAN2GAN
##### 4-1) Run below ipython files sequentically (you can choice 'data_type' in each ipython file)

```
./data
      /Generate_Medical_dataset_GAN_train_dataset_96x96.ipynb
      /Generate_Medical_dataset_G2G_train_dataset_120x120.ipynb
      /Generate_Medical_test_dataset.ipynb
```

### 3. Train GAN and GAN2GAN
#### 1) After generating datasets, run train.sh. It contains scripts to train GAN and GAN2GAN using a specific type of dataset.
#### 2) If training is done, experimental results for GAN and GAN2GAN will be saved in './result_data/'. You can analyze the experimental results using pre-made ipython files.
#### 3) Also, all weights will be saved in './weights/'.

### 4. Evaluate pretrained denoiser by GAN2GAN using custom test images

Download Pretrained weights from [[this google drive link]]()

## Citation

```
@inproceedings{cha2020gan2gan,
  title={GAN2GAN: Generative Noise Learning for Blind Denoising with Single Noisy Images},
  author={Cha, Sungmin and Park, Taeeon and Kim, Byeongjoon and Baek, Jongduk and Moon, Taesup},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

## Reference

[1] A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images [[arxiv]](https://arxiv.org/abs/1812.10366)

