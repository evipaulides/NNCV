wandb login

# Install timm for this environment
pip install torchmetrics

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 150 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-aug-drop-EVT" \