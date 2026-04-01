"""
Improved Cityscapes training script for DINOv2 segmentation.

Key upgrades:
- main + auxiliary loss support
- safe ID-preserving augmentations (future OOD-friendly)
- proper low LR for pretrained encoder
- AMP mixed precision
- cosine LR scheduler
- best checkpointing on validation loss
- mIoU logging
- larger crop size
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToDtype,
    ToImage,
    InterpolationMode,
)

from model import DINOv2SegModel


# -----------------------------
# Label conversion
# -----------------------------
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# -----------------------------
# Metrics
# -----------------------------
def mean_iou(logits, target, n_classes=19, ignore_index=255):
    preds = logits.argmax(1)
    ious = []

    for cls in range(n_classes):
        pred_mask = preds == cls
        target_mask = target == cls

        valid = target != ignore_index
        pred_mask = pred_mask & valid
        target_mask = target_mask & valid

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append(intersection / union)

    if len(ious) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(ious).mean()


# -----------------------------
# Args
# -----------------------------
def get_args_parser():
    parser = ArgumentParser("DINOv2 Cityscapes training")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="dinov2-cityscapes")
    parser.add_argument("--crop-size", type=int, default=256)
    return parser


# -----------------------------
# Main
# -----------------------------
def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Transforms (safe for future OOD)
    # -----------------------------
    img_transform = Compose([
        ToImage(),
        RandomResizedCrop(
            size=(args.crop_size, 2*args.crop_size),
            scale=(0.5, 2.0),
            interpolation=InterpolationMode.BILINEAR,
        ),
        RandomHorizontalFlip(),
        ColorJitter(0.2, 0.2, 0.2, 0.05),
        GaussianBlur(kernel_size=3),
        ToDtype(torch.float32, scale=True),
        Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ])

    target_transform = Compose([
        ToImage(),
        RandomResizedCrop(
            size=(args.crop_size, 2*args.crop_size),
            scale=(0.5, 2.0),
            interpolation=InterpolationMode.NEAREST,
        ),
        RandomHorizontalFlip(),
        ToDtype(torch.int64),
    ])

    valid_img_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ])

    valid_target_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    # -----------------------------
    # Datasets
    # -----------------------------
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=valid_img_transform,
        target_transform=valid_target_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = DINOv2SegModel(n_classes=19, freeze_encoder=False).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.proj.parameters(), "lr": 1e-3},
        {"params": model.fuse.parameters(), "lr": 1e-3},
        {"params": model.aspp.parameters(), "lr": 1e-3},
        {"params": model.head.parameters(), "lr": 1e-3},
        {"params": model.aux_head.parameters(), "lr": 1e-3},
    ], weight_decay=0.05)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_miou = 0.0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits, aux_logits = model(images)
                loss_main = criterion(logits, labels)
                loss_aux = criterion(aux_logits, labels)
                loss = loss_main + 0.4 * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for images, labels in valid_loader:
                labels = convert_to_train_id(labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().squeeze(1)

                with torch.cuda.amp.autocast():
                    logits, _ = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_miou += mean_iou(logits, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        val_miou /= len(valid_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "valid_mIoU": val_miou,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[1]["lr"],
        })

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt"),
            )

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
