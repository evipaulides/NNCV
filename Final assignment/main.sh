wandb login

# Install timm for this environment

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 128 \
    --epochs 200 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-combined-loss" \