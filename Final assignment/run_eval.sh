#!/bin/bash

wandb login

pip install torchmetrics
pip install matplotlib

# model weights in /home/scur2237/NNCV/Final assignment/checkpoints/plot_mahal/best_model-epoch=0110-val_dice=0.7036552280187607.pt

python3 evaluate_ood_svdd.py \
    --model-path "/home/scur2237/NNCV/Final assignment/submission-final-SVDD/model.pt"