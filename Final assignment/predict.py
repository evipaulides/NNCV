"""
Cityscapes Prediction Script for DINOv2 Segmentation Model

This script loads a pre-trained DINOv2 segmentation model, processes input images,
and saves the predicted masks. It is ready for submission to the challenge server.
"""

from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode

from model import DINOv2SegModel

# Fixed paths inside participant container
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
    return img_tensor

# -----------------------------
# Postprocessing
# -----------------------------
def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # pred shape: [1, num_classes, H, W]
    pred_class = torch.argmax(pred, dim=1).squeeze(0)  # remove batch dimension

    # Resize to original image size
    pred_resized = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_class)
    return pred_resized.cpu().numpy().astype(np.uint8)

# -----------------------------
# Main prediction loop
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = DINOv2SegModel(n_classes=19)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # Process all images in the folder
    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = img.size[::-1]  # (height, width)

            img_tensor = preprocess(img).to(device)

            # Forward pass (only main logits)
            pred, _ = model(img_tensor)

            seg_pred = postprocess(pred, original_shape)

            # Save mask
            out_path = Path(OUTPUT_DIR) / img_path.name
            Image.fromarray(seg_pred).save(out_path)

if __name__ == "__main__":
    main()
