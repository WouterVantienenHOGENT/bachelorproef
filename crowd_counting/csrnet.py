"""CSRNet for single-image crowd-density estimation."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class CSRNet(nn.Module):
    """CSRNet with an optional ImageNet-pretrained VGG-16 frontend.

    The output has a spatial stride of 8.  Summing its pixels yields the
    estimated number of people, provided the target map is downsampled with
    count preservation (see ``data.py``).
    """

    def __init__(self, pretrained_frontend: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained_frontend else None
        vgg_features = vgg16(weights=weights).features
        self.frontend = nn.Sequential(*list(vgg_features.children())[:23])
        self.backend = self._make_backend(
            [512, 512, 512, 256, 128, 64], in_channels=512
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_backend()

    @staticmethod
    def _make_backend(channels: list[int], in_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3,
                        padding=2, dilation=2
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_backend(self) -> None:
        for module in [self.backend, self.output_layer]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(image)))
