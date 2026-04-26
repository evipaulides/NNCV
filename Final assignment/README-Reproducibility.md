# Neural Networks for Computer Vision - U-Net-Based Semantic Segmentation with Out-Of-Distribution Detection
A baseline U-Net enhanced for semantic segmentation of Cityscapes images.
Comparing out-of-distribution detection methods using the enhanced U-Net as a fixed backbone.

This repository contains code to support the submitted report: ***'Comparing OOD Detection Methods for U-Net-based Semantic Segmentation of Cityscapes'***


## Project overview

This project proposes three out-of-distribution detection methods for the Cityscapes data set. The aim is to perform out-of-distribution detection based on the statistical distribution of the training data, rather than manual threshold based.

**Key contributions include:**

- Mehalanobis distance based: We model the distribution of in-distribution features as a multivariate Gaussian with mean $\boldsymbol{\mu}$ and covariance $\Sigma$, estimated from the training data.