wandb login

# Install timm for this environment
pip install torchmetrics
pip install matplotlib

python3 train_SVDD.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 150 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "training_SVDD" \