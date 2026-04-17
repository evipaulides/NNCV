"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    ColorJitter,
    GaussianBlur,
    RandomHorizontalFlip
)

from model import Model

from scipy.stats import genpareto
import numpy as np


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser

class AugmentedCityscapes(torch.utils.data.Dataset):
    def __init__(self, base_dataset, img_transform, target_transform):
        self.dataset = base_dataset
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # Decide flip ONCE
        do_flip = random.random() < 0.5

        # Apply transforms
        img = self.img_transform(img)
        target = self.target_transform(target)

        if do_flip:
            img = torch.flip(img, dims=[2])      # width dim
            target = torch.flip(target, dims=[2])

        return img, target

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        probs = torch.softmax(logits, dim=1)

        # Create valid mask (ignore void pixels)
        valid_mask = targets != self.ignore_index

        # Replace ignore pixels temporarily so one_hot works
        targets = targets.clone()
        targets[~valid_mask] = 0

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute Dice score per class
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, ce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def multiclass_dice_score(preds, targets, num_classes=19, ignore_index=255, smooth=1e-6):
    """
    preds: [B, H, W]
    targets: [B, H, W]
    """
    dice_scores = []

    valid_mask = targets != ignore_index

    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid_mask
        target_cls = (targets == cls) & valid_mask

        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float()

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))


def fit_ood_statistics(model, dataloader, device, percentile=99):
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # encoder only
            x1 = model.inc(images)
            x2 = model.down1(x1)
            x3 = model.down2(x2)
            x4 = model.down3(x3)
            x5 = model.down4(x4)

            feat = torch.mean(x5, dim=(2, 3))
            features.append(feat.cpu())

    features = torch.cat(features, dim=0)   # [N, 512]

    mean = features.mean(dim=0)

    centered = features - mean
    cov = centered.T @ centered / (features.shape[0] - 1)

    # regularization for numerical stability
    cov += 1e-4 * torch.eye(cov.shape[0])

    inv_cov = torch.inverse(cov)

    # compute train distances
    diff = centered
    left = diff @ inv_cov
    distances = torch.sum(left * diff, dim=1)

    distances_np = distances.numpy()
    u = np.percentile(distances_np, percentile)

    tail = distances_np[distances_np > u] - u

    if len(tail) < 50:
        print("Self Warning: too few tail samples, EVT may be unstable")

    shape, loc, scale = genpareto.fit(tail, floc=0)

    p = 1 - percentile / 100

    gpd_threshold = genpareto.ppf(p, shape, loc=loc, scale=scale)

    final_threshold = u + gpd_threshold

    model.feature_mean.copy_(mean.to(device))
    model.inv_cov.copy_(inv_cov.to(device))
    model.ood_threshold.copy_(torch.tensor(final_threshold).to(device))

    print(f"OOD threshold fitted: {final_threshold:.4f}")

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    img_transform = Compose([
    ToImage(),
    Resize((256, 512)),

    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img_transform_no_aug = Compose([
        ToImage(),
        Resize((256, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    # Load the dataset and make a split for training and validation
    base_train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
    )

    train_dataset = AugmentedCityscapes(
        base_train_dataset,
        img_transform=img_transform,
        target_transform=target_transform
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform_no_aug,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = CombinedLoss(num_classes=19, ignore_index=255, ce_weight=0.5, dice_weight=0.5)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)

    # Training loop
    best_valid_loss = float('inf')
    best_valid_dice = 0.0
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                #"learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            dice_scores = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                predictions_val = outputs.argmax(dim=1)
                batch_dice = multiclass_dice_score(predictions_val, labels)
                dice_scores.append(batch_dice.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            valid_dice = sum(dice_scores) / len(dice_scores)
            scheduler.step(valid_dice)

            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice": valid_dice,
                "learning_rate": scheduler.optimizer.param_groups[0]['lr']
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            #if valid_loss < best_valid_loss:
                #best_valid_loss = valid_loss
            if valid_dice > best_valid_dice:
                best_valid_dice = valid_dice
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_dice={valid_dice:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    print("Fitting OOD detection statistics...")
    fit_ood_statistics(model, train_dataloader, device, percentile=95)

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_dice={valid_dice:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)