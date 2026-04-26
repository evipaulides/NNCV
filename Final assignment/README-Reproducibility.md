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

After setting up the repository, there already is a model, training, prediction and dockerfile present. These should be **replaced** with the right files from this repository. Depending on what results you want to reproduce, determine what files should be used.

#### File locations
In this repository, there are six folders named `submission-unet-...`. These folders contain the model file, the dockerfile and the prediction file. Depending on what method results you want to reproduce, you use one of the six folder to extract your files from.

#### 1.1. Model prediction and Dockerfile
For Mahalanobis distance this folder name includes `Mahalanobis`, for the additional Extreme Value Theory the folder name includes `EVT` and for the Support Vector Data Description the folder name includes `SVDD`.
For each method, there are two folders. For evaluating the out-of-distribution detection, choose the folder with the name ending in just the method name. For evaluating the performance on the peak performance benchmark, choose the folder with the right method name that is ending with `...-onpeak`.

After choosing the right folder, replace your existing model, prediction and dockerfile with the folder's `model.py`, `Dockerfile` and prediction file. The prediction file is either `predict.py` or `predict_ood.py` depending on the task.

#### 1.2. Training: Mahalanobis-based methods
Both the Mahalanobis distance approach and the Mahalanobis distance + Extreme Value Theory need the `train.py` as given in this repository. This ensures that the distance statistics are saved the correct way.

#### 1.3. Training: Support Vector Data Description
The Support Vector Data Description approach needs the `train_SVDD.py` as given in this repository. This ensures that the learned feature representations for the distance statistics are saved the correct way.

### 2. Training the model
After making sure you have correctly replaced all needed documents, you can start training the model. Explaination how to do so can be found in `README-Slurm.md` in this repository.

### 3. Evaluating the trained model
After training the model on the training set, the best checkpoint is saved. Using this, the model can be submitted to the submission server. To do so, please follow the intructions given in `README-Submission.md`.

