"""Lightweight DINOv3 depth head from Raessan/depth_dinov3.

The module layout and parameter names follow
https://github.com/Raessan/depth_dinov3/blob/main/src/model_head.py so the
published ``weights/model.pth`` checkpoint loads without conversion. The only
interface addition is ``forward_features`` for exposing the fused geometry map
used as policy tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_norm(norm: str, channels: int, groups: int = 8) -> nn.Module:
    if norm == "gn":
        return nn.GroupNorm(min(groups, channels), channels)
    return nn.BatchNorm2d(channels)


class DSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "gn", groups: int = 8) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.bn1 = _make_norm(norm, in_channels, groups)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = _make_norm(norm, out_channels, groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class DSDown2(nn.Module):
    def __init__(self, channels: int, norm: str = "gn", groups: int = 8) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False)
        self.bn1 = _make_norm(norm, channels, groups)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = _make_norm(norm, channels, groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class FeatureFusionBlockLite(nn.Module):
    def __init__(
        self,
        channels: int = 160,
        norm: str = "gn",
        groups: int = 8,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = _make_norm(norm, channels, groups)
        self.act = nn.GELU()
        self.res = nn.Sequential(DSConv(channels, channels, norm=norm, groups=groups))
        self.use_skip = use_skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.bn(self.proj(x)))
        if self.use_skip and skip is not None:
            x = x + skip
        return self.res(x)


class DepthHeadLite(nn.Module):
    """Raessan's lightweight DPT-inspired depth head."""

    def __init__(
        self,
        in_ch: int = 384,
        out_size: tuple[int, int] = (640, 640),
        proj0_ch: int = 32,
        proj1_ch: int = 64,
        proj2_ch: int = 96,
        proj3_ch: int = 128,
        common_ch: int = 160,
        dropout: float = 0.2,
        norm: str = "gn",
        groups_gn: int = 8,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.align_corners = align_corners
        self.common_ch = common_ch
        self.drop = nn.Dropout2d(p=dropout)

        self.proj0 = nn.Conv2d(in_ch, proj0_ch, 1, bias=False)
        self.proj1 = nn.Conv2d(in_ch, proj1_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(in_ch, proj2_ch, 1, bias=False)
        self.proj3 = nn.Conv2d(in_ch, proj3_ch, 1, bias=False)
        self.down3 = DSDown2(proj3_ch, norm=norm, groups=groups_gn)

        self.to_c0 = DSConv(proj0_ch, common_ch, norm=norm, groups=groups_gn)
        self.to_c1 = DSConv(proj1_ch, common_ch, norm=norm, groups=groups_gn)
        self.to_c2 = DSConv(proj2_ch, common_ch, norm=norm, groups=groups_gn)
        self.to_c3 = DSConv(proj3_ch, common_ch, norm=norm, groups=groups_gn)

        self.fuse3 = FeatureFusionBlockLite(common_ch, norm=norm, groups=groups_gn, use_skip=False)
        self.fuse2 = FeatureFusionBlockLite(common_ch, norm=norm, groups=groups_gn, use_skip=True)
        self.fuse1 = FeatureFusionBlockLite(common_ch, norm=norm, groups=groups_gn, use_skip=True)
        self.fuse0 = FeatureFusionBlockLite(common_ch, norm=norm, groups=groups_gn, use_skip=True)

        self.h1 = nn.Conv2d(common_ch, common_ch // 2, 3, padding=1)
        self.h2 = nn.Conv2d(common_ch // 2, common_ch // 4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(common_ch // 4, 1, 1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)
        self.h1_bn = _make_norm(norm, common_ch // 2, groups_gn)
        self.h2_bn = _make_norm(norm, common_ch // 4, groups_gn)
        self.min_depth = 1e-3

    def _upsample(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=self.align_corners)

    def forward_features(self, feat_1x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feat_1x.shape
        b0 = self.proj0(feat_1x)
        b1 = self.proj1(feat_1x)
        b2 = self.proj2(feat_1x)
        b3 = self.down3(self.proj3(feat_1x))

        f0 = self.to_c0(self._upsample(b0, (height * 4, width * 4)))
        f1 = self.to_c1(self._upsample(b1, (height * 2, width * 2)))
        f2 = self.to_c2(b2)
        f3 = self.to_c3(b3)

        x = self.fuse3(f3, None)
        x = self.fuse2(self._upsample(x, f2.shape[-2:]), f2)
        x = self.fuse1(self._upsample(x, f1.shape[-2:]), f1)
        x = self.drop(x)
        return self.fuse0(self._upsample(x, f0.shape[-2:]), f0)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.h1_bn(self.h1(features)))
        x = self.drop(x)
        x = self._upsample(x, (x.shape[-2] * 2, x.shape[-1] * 2))
        x = self.relu(self.h2_bn(self.h2(x)))
        x = self.softplus(self.out(x)) + self.min_depth
        return self._upsample(x, self.out_size)

    def forward(self, feat_1x: torch.Tensor) -> torch.Tensor:
        return self.predict_from_features(self.forward_features(feat_1x))
