# GAN2GAN

The official code of GAN2GAN: Generative Noise Learning for Blind Denoising with Single Noisy Images (ICLR 2021) [[arxiv]](https://arxiv.org/abs/1905.10488)

## Quick Start

### 1. Requirements

```
$ pip install -r requirements.txt
$ mkdir pretrained_weights weights
```

### 2. Prepare Datasets

1) Download BSD[1], FMD[2] and Medical dataset from [[BSD68 dataset]](https://drive.google.com/file/d/10CJDhU9iYp3Ca_T1gLdzrg4Zl2Jmw7Lj/view?usp=sharing), [[FMD WF dataset]](https://drive.google.com/file/d/10T9uJv0ah_kCVvpjt4OCh0Rc5fqLqwvk/view?usp=sharing) and [[medical dataset]](https://drive.google.com/file/d/10MI6R3vkwBKrtHhW2TKOPs56dzH8p_5t/view?usp=sharing) respectively.
2) Locate 'Medical_images', 'Real_FM_images' and 'BSD68' in './data' directory.

```
./data
      /BSD68 
      /Real_FM_images 
      /Medical_images 
```

3) Generate FMD WF dataset for training GAN and GAN2GAN.

: Run below ipython files sequentically (you can choice 'data_type' among {'WF_avg1', 'WF_avg2', 'WF_avg4', 'WF_avg8', 'WF_avg16'} in each ipython file)

```
./data
      /Generate_FMD_WF_train_dataset.ipynb 
      /Generate_FMD_WF_GAN_train_dataset_96x96.ipynb 
      /Generate_FMD_WF_G2G_train_dataset_120x120.ipynb
      /Generate_FMD_WF_test_dataset.ipynb
```

4) Generate Medical dataset for training GAN and GAN2GAN.

: Run below ipython files sequentically (you can choice 'data_type' among {'Dose25', 'Dose50', 'Dose75', 'Dose100'} in each ipython file)

```
./data
      /Generate_Medical_dataset_GAN_train_dataset_96x96.ipynb
      /Generate_Medical_dataset_G2G_train_dataset_120x120.ipynb
      /Generate_Medical_test_dataset.ipynb
```

### 3. Train GAN and GAN2GAN
1) After generating datasets, run 'train.sh'. It contains scripts to train GAN and GAN2GAN using a specific type of dataset.
2) If training is done, experimental results for GAN and GAN2GAN will be saved in './result_data/'. You can analyze the experimental results using pre-made ipython files.
3) Also, all trained weights will be saved in './weights/'.

### 4. Evaluate pretrained denoiser by GAN2GAN on test images

1) Download Pretrained weights from [[this google drive link]](https://drive.google.com/file/d/103YjwKT5ZnB4Z_NKZhlh8X3SrM6k_BDr/view?usp=sharing), and then locate all weights in './pretrained_weights/'.
2) Generate test image datasets of all noise types by running below ipython files.

```
./data
      /Generate_Medical_test_dataset.ipynb
      /Generate_FMD_WF_test_dataset.ipynb
      /Generate_BSD_test_dataset.ipynb
```

3) Run 'test.sh'.
4) Experimental results are shown in below tables.

#### Synthetic noise datasets
|               | Gaussian Noise |               |               |               | Mixture Noise |               |               |               | Correlated Noise |               |
|:-------------:|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:----------------:|:-------------:|
|   PSNR /SSIM  |    std = 15    |    std = 25   |    std = 30   |    std = 50   |     s = 15    |     s = 25    |     s = 30    |     s = 50    |     std = 15     |    std = 25   |
| DnCNN-B (N2C) |  31.36 /0.8821 | 28.83 /0.8109 | 27.98 /0.7781 | 25.69 /0.6681 | 39.62 /0.9749 | 37.22 /0.9607 | 30.49 /0.8620 | 30.12 /0.8521 |   30.82 /0.8997  | 27.36 /0.8233 |
|   G2G (Ours)  |  31.36 /0.8845 | 28.87 /0.8107 | 27.87 /0.7744 | 25.65 /0.6778 | 42.46 /0.9889 | 39.65 /0.9812 | 30.41 /0.8562 | 29.93 /0.8450 |   31.21 /0.8976  | 27.50 /0.8188 |

#### Real noise datasets
|               |    Medical    |               |               |               |       WF      |               |               |               |               |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|   PSNR /SSIM  |   Dose = 25   |   Dose = 50   |   Dose = 75   |   Dose = 100  |    Avg = 1    |    Avg = 2    |    Avg = 4    |    Avg = 8    |    Avg = 16   |
| DnCNN-B (N2C) | 35.50 /0.6055 | 38.48 /0.7440 | 40.09 /0.8113 | 41.19 /0.8513 | 25.43 /0.3691 | 28.36 /0.5256 | 31.32 /0.6909 | 34.63 /0.8122 | 37.82 /0.9121 |
|   G2G (Ours)  | 46.72 /0.9700 | 48.06 /0.9748 | 49.20 /0.9732 | 48.82 /0.9717 | 32.73 /0.8157 | 32.86 /0.7806 | 33.79 /0.8134 | 35.22 /0.8316 | 38.82 /0.9148 |
## QnA
### 1. Where is the code for Smooth Noisy Patch Extraction (Eqn (4) in the paper)?

: Check smooth_area_detector() in './data/Generate_FMD_WF_GAN_train_dataset_96x96.ipynb'

### 2. How to train GAN and GAN2GAN using custom training dataset?

: You can easily customize ipynb files in './data/' to generate custom training dataset. Note that a lambda (the hyperparameter for controlling the level of homoneousity) for smooth_area_detector() should be selected carefully. Our empirical recommendation is to set a lambda that can extract at least 10,000 noise patches.

### 3. What is the proper number of iterations for GAN2GAN?

: We emprically found that 'iter = 3' achieves the best results in various datasets and additional iterations for GAN2GAN only helps to improve the performance of WF.

## Citation

```
@inproceedings{
      cha2021gangan,
      title={{\{}GAN{\}}2{\{}GAN{\}}: Generative Noise Learning for Blind Denoising with Single Noisy Images},
      author={Sungmin Cha and Taeeon Park and Byeongjoon Kim and Jongduk Baek and Taesup Moon},
      booktitle={International Conference on Learning Representations},
      year={2021},
      url={https://openreview.net/forum?id=SHvF5xaueVn}
}
```

## Reference

[1] A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images [[arxiv]](https://arxiv.org/abs/1812.10366)

[2] The Berkeley Segmentation Dataset and Benchmark [[link]](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
