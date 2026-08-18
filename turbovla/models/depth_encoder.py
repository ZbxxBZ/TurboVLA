from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import DepthEncoderConfig


class MetricDepthEncoder(nn.Module):
    """把 RoboTwin 的真实度量深度图编码成与 RGB patch 一一对齐的 token。"""

    def __init__(self, config: DepthEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # 使用与 DINOv3 相同的 patch 步长，使同一空间位置的 RGB/depth token 可以直接做交叉注意力。
        # [B,3,196,256]
        self.patch_embed = nn.Conv2d(
            in_channels=2,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.token_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        if config.frozen:
            self.requires_grad_(False)

    def _resize(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.shape[-2:] == (self.config.image_size, self.config.image_size):
            return depth
        batch_size, num_views = depth.shape[:2]
        flat_depth = depth.flatten(0, 1)
        # 深度不能使用双线性插值，否则物体边界会生成并不存在的中间距离。
        flat_depth = F.interpolate(
            flat_depth.float(),
            size=(self.config.image_size, self.config.image_size),
            mode="nearest",
        )
        return flat_depth.view(batch_size, num_views, 1, self.config.image_size, self.config.image_size)

    def _normalize(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = self._resize(depth).float()

        # RoboTwin 输出毫米；模型内部统一使用米，便于固定有效范围并跨相机共享编码器。
        if self.config.input_unit == "millimeter":
            depth_m = depth / float(self.config.depth_scale)
        else:
            depth_m = depth

        valid = torch.isfinite(depth_m)
        valid = valid & (depth_m >= self.config.min_depth_m) & (depth_m <= self.config.max_depth_m)
        depth_m = torch.nan_to_num(
            depth_m,
            nan=self.config.min_depth_m,
            posinf=self.config.max_depth_m,
            neginf=self.config.min_depth_m,
        ).clamp(self.config.min_depth_m, self.config.max_depth_m)

        if self.config.use_log_depth:
            depth_m = depth_m.log()
            lower = math.log(self.config.min_depth_m)
            upper = math.log(self.config.max_depth_m)
        else:
            lower = self.config.min_depth_m
            upper = self.config.max_depth_m

        # 映射到 [-1, 1]；无效像素置零，后续同时用 patch mask 阻止其成为 K/V。
        normalized = 2.0 * (depth_m - lower) / (upper - lower) - 1.0
        normalized = normalized.masked_fill(~valid, 0.0)
        return normalized, valid

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.ndim == 4:
            depth = depth.unsqueeze(2)
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError(f"depth must be [B,V,1,H,W] or [B,V,H,W], got {tuple(depth.shape)}")
        if depth.shape[1] != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} depth views, got {depth.shape[1]}")

        normalized, valid_pixels = self._normalize(depth)
        batch_size, num_views = normalized.shape[:2]
        # 无效深度与归一化中点都为 0；显式 validity 通道让卷积能够区分二者。
        encoder_input = torch.cat(
            [normalized, valid_pixels.to(dtype=normalized.dtype)],
            dim=2,
        )
        flat_depth = encoder_input.flatten(0, 1).to(dtype=self.patch_embed.weight.dtype)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            tokens = self.patch_embed(flat_depth).flatten(2).transpose(1, 2)
            tokens = self.dropout(self.token_norm(tokens))

        # 一个 patch 内无效像素比例过高时，将整个 token 标为无效，供 cross-attention 的 K/V mask 使用。
        invalid_fraction = F.avg_pool2d(
            (~valid_pixels.flatten(0, 1)).float(),
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        invalid_mask = invalid_fraction.flatten(1) >= self.config.invalid_threshold
        tokens = tokens.view(batch_size, num_views, self.num_patches, self.config.hidden_dim)
        invalid_mask = invalid_mask.view(batch_size, num_views, self.num_patches)
        return tokens, invalid_mask
