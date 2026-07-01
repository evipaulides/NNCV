# Neural Networks for Computer Vision - U-Net-Based Semantic Segmentation with Out-Of-Distribution Detection
A baseline U-Net enhanced for semantic segmentation of Cityscapes images.
Comparing out-of-distribution detection methods using the enhanced U-Net as a fixed backbone.

This repository contains code to support the submitted report: ***'Comparing OOD Detection Methods for U-Net-based Semantic Segmentation of Cityscapes'***

---
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
├── README-Installation.md
├── README-Reproducibility.md
├── README-Slurm.md
├── README-Submission.md
├── download_docker_and_data.sh
├── jobscript_slurm.sh
├── main.sh
├── evaluation_slurm.sh
├── run_eval.sh
├── Dockerfile
├── model.py
├── model_SVDD.py
├── train_mahal.py
├── train_EVT.py
├── train_SVDD.py
├── evaluate_ood.py
├── evaluate_ood_SVDD.py
├── predict.py
├── predict_ood.py
└── predict_ood_SVDD.py

```
---
## Getting started

### 0. Installation
To get started, follow the instructions from `README-Installation.md` 

### 1. File setup
In order to reproduce the results of the report, it is crucial that the right training, model, prediction and dockerfile are used.

After setting up the repository, there already are model, training, prediction and dockerfiles present. And, there are already bashfiles present which are needed to be able to run the code. Depending on what results you want to reproduce, change the `main.sh` to call the right training code.
- For example: `train_mahal.py`, `train_EVT.py` or `strain_SVDD.py`

The second bashfile to adjust is `run_eval.sh`. In this file, you should change the evaluation code that is being called to the corresponding file as well as the trained model weights after `--model-path`.
- For example: `evaluate_ood.py` for mahal and EVT and `evaluate_ood_svdd.py` for SVDD.

Lastly, to be able to compare the encoded feature distances with OOD samples, you can download a pseudo OOD dataset (such as CIFAR-10 data). Make sure to check the filepaths to this data in the evaluation code. 

### 2. Training the model
After making sure you have correctly replaced all needed documents, you can start training the model. Explaination how to do so can be found in `README-Slurm.md` in this repository.

### 3. Evaluating the fitted statistics
In order to check the encoded feature space with the OOD data, run `run_eval.sh` to create plots with the encoded distances on the x-axis and the frequency on the y-axis. This way, it can be checked if the threshold falls within the two distributions.
- If this threshold does not work, the percentile based thresholding can be adjusted in the training code.

### 4. Submitting the trained model
After training the model on the training set, the best checkpoint is saved. Using this, the model can be submitted to the submission server. To do so, please follow the intructions given in `README-Submission.md` and make sure to be connected to the TU/e VPN or WiFi network.

## Contributors
Evi Paulides, e.paulides@student.tue.nl
Neural Networks for Computer Vision (5LSM0)
