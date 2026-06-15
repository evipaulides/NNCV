from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model_SVDD import Model
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/home/scur2237/NNCV/Final assignment/submission-final-SVDD/model.pt")
args = parser.parse_args()

# =========================
# Paths
# =========================
MODEL_PATH = args.model_path
OUTPUT_DIR = MODEL_PATH.replace("final_model", "ood_evaluation").rsplit("/", 1)[0] # path: 
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Preprocessing (same idea as training data)
# =========================
transform = Compose([
    ToImage(),
    Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR),
    ToDtype(dtype=torch.float32, scale=True),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# =========================
# Load CIFAR10 (OOD dataset)
# =========================
def load_cifar10():
    return CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )


# =========================
# Compute Mahalanobis distances
# =========================
def compute_distances(model, dataloader, device):
    model.eval()
    distances = []

    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)

            # your model returns (segmentation, distance)
            _, dist = model(img)

            distances.extend(dist.cpu().numpy())

    return np.array(distances)


# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load model
    # -------------------------
    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # -------------------------
    # Load CIFAR10
    # -------------------------
    cifar_dataset = load_cifar10()

    cifar_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # -------------------------
    # Compute OOD distances
    # -------------------------
    print("Computing CIFAR10 OOD distances...")
    cifar_distances = compute_distances(model, cifar_loader, device)

    print(f"Total samples: {len(cifar_distances)}")

    # -------------------------
    # Load threshold from model
    # -------------------------
    threshold = model.ood_threshold.item()
    print(f"OOD threshold: {threshold:.4f}")

    rejection_rate = (cifar_distances > threshold).mean()
    print(f"CIFAR10 rejection rate: {rejection_rate * 100:.2f}%")

    # -------------------------
    # Plot distribution
    # -------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        cifar_distances,
        bins=100,
        density=True,
        alpha=0.7,
        label="CIFAR10 (OOD)"
    )

    plt.axvline(
        threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Threshold = {threshold:.2f}"
    )

    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Density")
    plt.title("OOD Detection: CIFAR10 Distance Distribution")

    plt.legend()

    plot_path = os.path.join(OUTPUT_DIR, "cifar10_ood_hist.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to: {plot_path}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    main()