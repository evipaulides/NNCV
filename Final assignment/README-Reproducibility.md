# Neural Networks for Computer Vision - U-Net-Based Semantic Segmentation with Out-Of-Distribution Detection
A baseline U-Net enhanced for semantic segmentation of Cityscapes images.
Comparing out-of-distribution detection methods using the enhanced U-Net as a fixed backbone.

This repository contains code to support the submitted report: ***'Comparing OOD Detection Methods for U-Net-based Semantic Segmentation of Cityscapes'***


## Project overview

This project proposes three out-of-distribution detection methods for the Cityscapes data set. The aim is to perform out-of-distribution detection based on the statistical distribution of the training data, rather than manual threshold based.

The baseline U-Net is first enhanced with data augmentation and dropout after which it is trained on performing the semantic segmentation task. After training, the distance statistics are saved and later used for out-of-distribution detection.

**Key contributions include:**

- Mahalanobis distance based: We model the distribution of in-distribution features as a multivariate Gaussian with mean $\boldsymbol{\mu}$ and covariance $\Sigma$, estimated from the training data.
- Mahalanobis distance + Extreme Value Theory: we extend this approach with modeling the tail of the Mahalanobis distance distribution using Extreme Value Theory.
- Support Vector Data Description: We enclose in-distribution features within a hypersphere. Using bottleneck features $\mathbf{f}$, Supprt Vector Data Description minimizes the distance to a fixed center $\mathbf{c}$.

## Repository structure
The 'Final assignment' folder is structured the following way:

```bash
Final assignment/
├── submission-unet-EVT-onpeak/
│   ├── Dockerfile
│   ├── model.py
│   └── predict.py
├── submission-unet-EVT/
│   ├── Dockerfile
│   ├── model.py
│   └── predict_ood.py
├── submission-unet-Mahalanobis-onpeak/
│   ├── Dockerfile
│   ├── model.py
│   └── predict.py
├── submission-unet-Mahalanobis/
│   ├── Dockerfile
│   ├── model.py
│   └── predict_ood.py
├── submission-unet-SVDD-onpeak/
│   ├── Dockerfile
│   ├── model.py
│   └── predict.py
├── submission-unet-SVDD/
│   ├── Dockerfile
│   ├── model.py
│   └── predict_ood.py
├── README-Installation.md
├── README-Reproducibility.md
├── README-Slurm.md
├── README-Submission.md
├── download_docker_and_data.sh
├── jobscript_slurm.sh
├── main.sh
├── Dockerfile
├── model.py
├── predict.py
├── predict_ood.py
├── train.py
└── train_SVDD.py

```

## Getting started

### 0. Installation
To get started, follow the instructions from `README-Installation.md` 

### 1. File setup
In order to reproduce the results of the report, it is crucial that the right training, model, prediction and dockerfile are used.

After setting up the repository, there already is a model, training, prediction and dockerfile present. These should be replaced with the right files from this repository. Depending on what results you want to reproduce, the following files should be used.

#### 1.0. File locations
In this repository, there are six folders named `submission-unet-...`. These folders contain the model file, the dockerfile and the prediction file. Depending on what method results you want to reproduce, you use one of the six folder to extract your files from.

#### 1.1. Model.py and Dockerfile
Since the U-Net backbone is fixed for all out-of-distribution detection methods, they all use the same model architecure. Therefore, the existing `model.py` can be replaced with any of the `model.py` files of all six folders.

The `Dockerfile` used should be in line with the aim. For only evaluating semantic segmentation, the Dockerfile of one of the `...-onpeak` folders should be used. For getting the results of the out-of-distribution detection, the Dockerfile of one of other three folders should be used.
Replace your existing Dockerfile with the Dockerfile suited for your task of preference.

#### 1.2. Training: Mahalanobis-based methods
Both the Mahalanobis distance approach and the Mahalanobis distance + Extreme Value Theory need the `train.py` as given in this repository. This ensures that the distance statistics are saved the correct way.

#### 1.3. Training: Support Vector Data Description
The Support Vector Data Description approach needs the `train_SVDD.py` as given in this repository. This ensures that the learned feature representations for the distance statistics are saved the correct way.