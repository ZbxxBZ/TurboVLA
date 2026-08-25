from __future__ import annotations

import torch
from torch import nn

from .configuration import DepthFusionConfig


class _DepthGateMixin:
    config: DepthFusionConfig
    depth_gate: nn.Parameter

    def _init_depth_gate(self, config: DepthFusionConfig) -> None:
        self._gate_override: float | None = None
        self._last_residual_ratio: torch.Tensor | None = None
        if config.gate_parameterization == "bounded_sigmoid":
            ratio = (config.gate_init - config.gate_min) / (config.gate_max - config.gate_min)
            raw_init = torch.logit(torch.tensor(ratio, dtype=torch.float32)).item()
        else:
            raw_init = float(config.gate_init)
        self.depth_gate = nn.Parameter(torch.full((config.hidden_dim,), raw_init))

    def set_gate_override(self, value: float | None) -> None:
        if value is not None:
            value = float(value)
            if self.config.gate_parameterization == "bounded_sigmoid" and not (
                self.config.gate_min <= value <= self.config.gate_max
            ):
                raise ValueError(
                    f"gate override {value} is outside "
                    f"[{self.config.gate_min}, {self.config.gate_max}]"
                )
        self._gate_override = value

    def effective_gate(self) -> torch.Tensor:
        if self._gate_override is not None:
            return torch.full_like(self.depth_gate, self._gate_override)
        if self.config.gate_parameterization == "bounded_sigmoid":
            width = self.config.gate_max - self.config.gate_min
            return self.config.gate_min + width * torch.sigmoid(self.depth_gate)
        return torch.tanh(self.depth_gate)

    def residual_ratio(self) -> torch.Tensor | None:
        """Return the latest per-view RMS(g * delta) / RMS(rgb) values."""
        return self._last_residual_ratio

    def _record_residual_ratio(
        self,
        rgb_tokens: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            reduce_dims = tuple(range(2, rgb_tokens.ndim))
            rgb_rms = rgb_tokens.detach().float().square().mean(dim=reduce_dims).sqrt()
            residual_rms = residual.detach().float().square().mean(dim=reduce_dims).sqrt()
            self._last_residual_ratio = (residual_rms / rgb_rms.clamp_min(1e-6)).flatten()


class GatedAlignedDepthFusion(_DepthGateMixin, nn.Module):
    """把每个深度 token 只注入同索引的 RGB token。"""

    def __init__(self, config: DepthFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.depth_norm = nn.LayerNorm(config.hidden_dim)
        self.local_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.delta_dropout = nn.Dropout(config.dropout)
        self._init_depth_gate(config)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        depth_invalid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rgb_tokens.ndim != 4 or depth_tokens.ndim != 4:
            raise ValueError("rgb_tokens and depth_tokens must be [B,V,N,D]")
        if rgb_tokens.shape != depth_tokens.shape:
            raise ValueError(
                f"RGB/depth token shapes must match, got {tuple(rgb_tokens.shape)} and {tuple(depth_tokens.shape)}"
            )

        batch_size, num_views, num_tokens, hidden_dim = rgb_tokens.shape
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"expected token hidden dim {self.config.hidden_dim}, got {hidden_dim}"
            )

        aligned_depth = depth_tokens.to(device=rgb_tokens.device, dtype=rgb_tokens.dtype)
        local_delta = self.local_projection(self.depth_norm(aligned_depth))
        local_delta = self.delta_dropout(local_delta)

        if depth_invalid_mask is not None:
            expected_mask_shape = (batch_size, num_views, num_tokens)
            if tuple(depth_invalid_mask.shape) != expected_mask_shape:
                raise ValueError(
                    f"depth_invalid_mask must be {expected_mask_shape}, got {tuple(depth_invalid_mask.shape)}"
                )
            invalid_mask = depth_invalid_mask.to(device=rgb_tokens.device, dtype=torch.bool)
            # 在线性层之后清零，避免 local_projection.bias 从无效 token 泄漏到 RGB。
            local_delta = local_delta.masked_fill(invalid_mask[..., None], 0.0)

        gate = self.effective_gate().to(device=local_delta.device, dtype=local_delta.dtype)
        residual = gate.view(1, 1, 1, -1) * local_delta
        self._record_residual_ratio(rgb_tokens, residual)
        return rgb_tokens + residual


# rgb_tokens:         [B, V, N, D]
# depth_tokens:       [B, V, N, D]
# depth_invalid_mask: [B, V, N]
class GatedDepthCrossAttention(_DepthGateMixin, nn.Module):
    """以 RGB token 为 Query、真实深度 token 为 Key/Value 的门控交叉注意力。"""

    def __init__(self, config: DepthFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.rgb_norm = nn.LayerNorm(config.hidden_dim)
        self.depth_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.nheads,
            dropout=config.dropout,
            batch_first=True,
        )
        if config.zero_init_output:
            nn.init.zeros_(self.cross_attention.out_proj.weight)
            if self.cross_attention.out_proj.bias is not None:
                nn.init.zeros_(self.cross_attention.out_proj.bias)
        self.delta_dropout = nn.Dropout(config.dropout)

        # 门控参数
        # 逐通道 gate 从 0 开始，首次加载旧 RGB checkpoint 时新分支不会改变原模型输出。
        self._init_depth_gate(config)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        depth_invalid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rgb_tokens.ndim != 4 or depth_tokens.ndim != 4:
            raise ValueError("rgb_tokens and depth_tokens must be [B,V,N,D]")
        if rgb_tokens.shape != depth_tokens.shape:
            raise ValueError(
                f"RGB/depth token shapes must match, got {tuple(rgb_tokens.shape)} and {tuple(depth_tokens.shape)}"
            )

        batch_size, num_views, num_tokens, hidden_dim = rgb_tokens.shape
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"expected token hidden dim {self.config.hidden_dim}, got {hidden_dim}"
            )
        flat_rgb = rgb_tokens.reshape(
            batch_size * num_views,
            num_tokens,
            hidden_dim
        )  # [B,V,N,D] → [B×V,N,D]

        # 融合模块自身统一设备和精度，避免单独调用时 depth/mask 仍在 CPU 导致注意力报错。
        flat_depth = depth_tokens.reshape(batch_size * num_views, num_tokens, hidden_dim).to(
            device=flat_rgb.device,
            dtype=flat_rgb.dtype,
        )

        if depth_invalid_mask is None:
            key_padding_mask = torch.zeros(
                batch_size * num_views,
                num_tokens,
                dtype=torch.bool,
                device=rgb_tokens.device,
            )
        else:
            expected_mask_shape = (batch_size, num_views, num_tokens)
            if tuple(depth_invalid_mask.shape) != expected_mask_shape:
                raise ValueError(
                    f"depth_invalid_mask must be {expected_mask_shape}, got {tuple(depth_invalid_mask.shape)}"
                )
            key_padding_mask = depth_invalid_mask.to(
                device=rgb_tokens.device,
                dtype=torch.bool,
            ).reshape(batch_size * num_views, num_tokens).clone()

        # PyTorch MHA 在一整行 K/V 都被 mask 时会产生 NaN；临时放开一个零 token，随后再把该行残差清零。
        all_invalid = key_padding_mask.all(dim=1)
        # 全程使用张量操作，避免 all_invalid.any() 触发每次推理的 GPU-CPU 同步。
        flat_depth = flat_depth.clone()
        flat_depth[:, 0] = torch.where(  # 如果这一行全部无效，就把第一个 token 置零，否则保留原来的第一个 token。
            all_invalid[:, None],
            torch.zeros_like(flat_depth[:, 0]),
            flat_depth[:, 0],
        )
        # 对于普通行，第一个 mask 保持不变；对于全无效行，把第一个 mask 临时改为 False
        key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_invalid

        normalized_depth = self.depth_norm(flat_depth)

        delta, _ = self.cross_attention(
            query=self.rgb_norm(flat_rgb),
            key=normalized_depth,
            value=normalized_depth,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # 把整个全无效视角的 delta 清零
        delta = self.delta_dropout(delta)
        delta = delta.masked_fill(all_invalid[:, None, None], 0.0)

        # tanh 限制深度修正幅度；gate=0 时结果逐元素严格等于原 RGB token。
        gate = self.effective_gate().to(device=delta.device, dtype=delta.dtype)
        residual = gate.view(1, 1, -1) * delta
        self._record_residual_ratio(
            flat_rgb.view(batch_size, num_views, num_tokens, hidden_dim),
            residual.view(batch_size, num_views, num_tokens, hidden_dim),
        )
        fused = flat_rgb + residual
        return fused.view(batch_size, num_views, num_tokens, hidden_dim) # [B×V,N,D] -> [B,V,N,D]
