from __future__ import annotations

import torch
from torch import nn


class TinyBackbone(nn.Module):
    output_channels = 128

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class ConvNeXtBackbone(nn.Module):
    output_channels = 768

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.features = convnext_tiny(weights=weights).features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class DDPFNet(nn.Module):
    """Density and Detection Point Fusion network.

    The shared visual encoder feeds a mass-preserving density head, a point
    localization head, and the frozen feature export consumed by STRFE.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = False,
        export_channels: int | None = None,
    ) -> None:
        super().__init__()
        if backbone == "tiny":
            self.backbone = TinyBackbone()
        elif backbone == "convnext_tiny":
            self.backbone = ConvNeXtBackbone(pretrained)
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}")
        channels = self.backbone.output_channels
        self.export_channels = int(export_channels or channels)
        self.feature_projection = (
            nn.Conv2d(channels, self.export_channels, 1)
            if self.export_channels != channels
            else nn.Identity()
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(),
        )
        self.localization_head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_features = self.backbone(images)
        return {
            "features": self.feature_projection(backbone_features),
            "backbone_features": backbone_features,
            "density": self.density_head(backbone_features),
            "localization_logits": self.localization_head(backbone_features),
        }


# Backwards-compatible name retained for existing checkpoints and imports.
DualHeadPerception = DDPFNet
