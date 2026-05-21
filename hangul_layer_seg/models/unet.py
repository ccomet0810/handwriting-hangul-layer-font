from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net for grayscale jamo semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([_match_size(d3, e3), e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([_match_size(d2, e2), e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([_match_size(d1, e1), e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


class ConditionalUNet(nn.Module):
    """U-Net conditioned on Hangul composition indices."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        condition_embed_dim: int = 8,
        l_index_count: int = 1,
        v_index_count: int = 1,
        t_index_count: int = 1,
        vowel_group_count: int = 1,
    ) -> None:
        super().__init__()
        self.l_embedding = nn.Embedding(l_index_count, condition_embed_dim)
        self.v_embedding = nn.Embedding(v_index_count, condition_embed_dim)
        self.t_embedding = nn.Embedding(t_index_count, condition_embed_dim)
        self.vowel_group_embedding = nn.Embedding(vowel_group_count, condition_embed_dim)
        self.condition_channels = condition_embed_dim * 4
        self.unet = UNet(
            in_channels=in_channels + self.condition_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != 4:
            raise ValueError(f"condition must have shape [B, 4], got {tuple(condition.shape)}")
        condition = condition.long()
        condition_vector = torch.cat(
            [
                self.l_embedding(condition[:, 0]),
                self.v_embedding(condition[:, 1]),
                self.t_embedding(condition[:, 2]),
                self.vowel_group_embedding(condition[:, 3]),
            ],
            dim=1,
        )
        condition_map = condition_vector[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.unet(torch.cat([x, condition_map], dim=1))


def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
