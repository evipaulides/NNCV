wandb login

# Install timm for this environment
pip install timm

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dino-unet-training" \