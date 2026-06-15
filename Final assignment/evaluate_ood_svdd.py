from pathlib import Path
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10, Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from sklearn.decomposition import PCA

from model_SVDD import Model


# =====================================================
# Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    required=True
)

parser.add_argument(
    "--cityscapes-dir",
    type=str,
    default="./data/cityscapes"
)

args = parser.parse_args()


# =====================================================
# Paths
# =====================================================

MODEL_PATH = args.model_path

OUTPUT_DIR = os.path.join(
    os.path.dirname(MODEL_PATH),
    "ood_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# Transform
# =====================================================

transform = Compose([
    ToImage(),
    Resize((256, 512), interpolation=InterpolationMode.BILINEAR),
    ToDtype(torch.float32, scale=True),
    Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),
])


# =====================================================
# Feature extraction
# =====================================================

def extract_features(model, dataloader, device):

    model.eval()

    features = []

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(device)

            x1 = model.inc(images)
            x2 = model.down1(x1)
            x3 = model.down2(x2)
            x4 = model.down3(x3)
            x5 = model.down4(x4)

            feat = x5.flatten(2).mean(dim=2)

            features.append(feat.cpu())

    return torch.cat(features, dim=0)


# =====================================================
# SVDD distances
# =====================================================

def compute_svdd_distances(features, center):

    dist_sq = torch.sum(
        (features - center.unsqueeze(0).cpu()) ** 2,
        dim=1
    )

    return dist_sq.numpy()


# =====================================================
# Main
# =====================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading model...")

    model = Model().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    center = model.svdd_center.detach().cpu()

    radius = model.svdd_radius.item()

    print(f"SVDD radius = {radius:.4f}")

    # =================================================
    # Datasets
    # =================================================

    city_dataset = Cityscapes(
        root=args.cityscapes_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=Compose([
            ToImage(),
            Resize((256, 512), interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64)
        ])
    )

    city_loader = torch.utils.data.DataLoader(
        city_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    cifar_dataset = CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    cifar_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # =================================================
    # Feature extraction
    # =================================================

    print("Extracting Cityscapes features...")

    city_features = extract_features(
        model,
        city_loader,
        device
    )

    print("Extracting CIFAR10 features...")

    cifar_features = extract_features(
        model,
        cifar_loader,
        device
    )

    print("Done.")

    # =================================================
    # Distances
    # =================================================

    city_distances = compute_svdd_distances(
        city_features,
        center
    )

    cifar_distances = compute_svdd_distances(
        cifar_features,
        center
    )

    # =================================================
    # Rejection rates
    # =================================================

    city_rejection = (
        city_distances > radius
    ).mean()

    cifar_rejection = (
        cifar_distances > radius
    ).mean()

    print()
    print(f"Cityscapes rejection rate : {100*city_rejection:.2f}%")
    print(f"CIFAR10 rejection rate    : {100*cifar_rejection:.2f}%")

    # =================================================
    # Distance histogram
    # =================================================

    plt.figure(figsize=(10, 6))

    plt.hist(
        city_distances,
        bins=100,
        density=True,
        alpha=0.5,
        label="Cityscapes (ID)"
    )

    plt.hist(
        cifar_distances,
        bins=100,
        density=True,
        alpha=0.5,
        label="CIFAR10 (OOD)"
    )

    plt.axvline(
        radius,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Radius = {radius:.2f}"
    )

    plt.xlabel("SVDD Distance")
    plt.ylabel("Density")
    plt.title("SVDD Distance Distribution")

    plt.legend()

    hist_path = os.path.join(
        OUTPUT_DIR,
        "svdd_distance_hist.png"
    )

    plt.savefig(
        hist_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {hist_path}")

    # =================================================
    # PCA feature visualization
    # =================================================

    print("Running PCA...")

    all_features = torch.cat(
        [
            city_features,
            cifar_features
        ],
        dim=0
    ).numpy()

    pca = PCA(n_components=2)

    features_2d = pca.fit_transform(
        all_features
    )

    n_city = len(city_features)

    city_2d = features_2d[:n_city]
    cifar_2d = features_2d[n_city:]

    center_2d = pca.transform(
        center.numpy().reshape(1, -1)
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        city_2d[:, 0],
        city_2d[:, 1],
        s=5,
        alpha=0.4,
        label="Cityscapes (ID)"
    )

    plt.scatter(
        cifar_2d[:, 0],
        cifar_2d[:, 1],
        s=5,
        alpha=0.4,
        label="CIFAR10 (OOD)"
    )

    plt.scatter(
        center_2d[:, 0],
        center_2d[:, 1],
        marker="*",
        s=400,
        label="SVDD Center"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title("SVDD Feature Space (PCA)")

    plt.legend()

    pca_path = os.path.join(
        OUTPUT_DIR,
        "svdd_feature_pca.png"
    )

    plt.savefig(
        pca_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {pca_path}")


if __name__ == "__main__":
    main()