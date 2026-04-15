import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# -----------------------------
# Normalization helper
# -----------------------------
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# -----------------------------
# ASPP (DeepLab-style context)
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(1, 6, 12, 18)):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
                nn.GELU()
            )
        ])

        for r in rates[1:]:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                    norm_layer(out_channels),
                    nn.GELU()
                )
            )

        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        feats = [b(x) for b in self.blocks]
        x = torch.cat(feats, dim=1)
        return self.project(x)


# -----------------------------
# Main Model
# -----------------------------
class DINOv2SegModel(nn.Module):
    def __init__(self, n_classes=19, freeze_encoder=False):
        super().__init__()

        self.patch_size = 16

        # -----------------------------
        # Backbone (DINOv2 ViT-B)
        # -----------------------------
        self.encoder = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # get all 4 feature maps
            img_size=(256, 512)  # must match your input size
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        encoder_channels = self.encoder.feature_info.channels()

        # -----------------------------
        # Projection layers (unify dims)
        # -----------------------------
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 256, 1, bias=False),
                norm_layer(256),
                nn.GELU()
            )
            for c in encoder_channels
        ])

        # -----------------------------
        # Feature fusion
        # -----------------------------
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.GELU()
        )

        # -----------------------------
        # ASPP context module
        # -----------------------------
        self.aspp = ASPP(256, 256)

        # -----------------------------
        # Final segmentation head
        # -----------------------------
        self.head = nn.Conv2d(256, n_classes, 1)

        # -----------------------------
        # Auxiliary head (for training stability)
        # -----------------------------
        self.aux_head = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input size must be divisible by {self.patch_size}, got {(H, W)}")

        # -----------------------------
        # Encoder features
        # -----------------------------
        feats = self.encoder(x)  # list of 4 tensors (same spatial size)

        # Project all to 256 channels
        feats = [proj(f) for proj, f in zip(self.proj, feats)]

        # -----------------------------
        # Feature fusion
        # -----------------------------
        x_fused = torch.cat(feats, dim=1)
        x_fused = self.fuse(x_fused)

        # -----------------------------
        # ASPP context
        # -----------------------------
        x_aspp = self.aspp(x_fused)

        # -----------------------------
        # Main output
        # -----------------------------
        logits = self.head(x_aspp)

        # -----------------------------
        # Auxiliary output (from mid feature)
        # -----------------------------
        aux_logits = self.aux_head(feats[-2])  # second deepest layer

        # Upsample to input resolution
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        aux_logits = F.interpolate(aux_logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits, aux_logits