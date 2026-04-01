import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Model(nn.Module):
    """
    U-Net-style segmentation model with DINOv2 ViT backbone from timm.
    Compatible with 256x256 input images.
    """
    def __init__(self, in_channels=3, n_classes=19, freeze_encoder=True):
        super().__init__()

        # Patch size of ViT
        self.patch_size = 16

        # DINOv2 backbone (timm)
        self.encoder = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # explicitly 4 feature maps
            img_size=(256, 256)  # must match your input size
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Project encoder channels to manageable number for decoder
        encoder_channels = self.encoder.feature_info.channels()  # list of 4 ints
        self.proj = nn.ModuleList([nn.Conv2d(c, min(c, 256), 1) for c in encoder_channels])

        # U-Net-style decoder
        self.up1 = Up(min(encoder_channels[-1], 256) + min(encoder_channels[-2], 256), 256)
        self.up2 = Up(256 + min(encoder_channels[-3], 256), 128)
        self.up3 = Up(128 + min(encoder_channels[-4], 256), 64)
        self.up4 = Up(64 + min(encoder_channels[0], 256), 64)  # shallowest skip
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # Ensure input is divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input size must be divisible by {self.patch_size}, got {(H,W)}")

        # Encoder features
        feats = self.encoder(x)  # list of 4 feature maps
        feats = [proj(f) for proj, f in zip(self.proj, feats)]

        # Assign features to skip connections
        x1, x2, x3, x4 = feats  # shallowest → deepest

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x1)  # can also just interpolate if you don't want extra skip
        logits = self.outc(x)

        # Optional: interpolate to original input size
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upsample then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample x1 to match spatial size of x2
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv to get desired output channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)