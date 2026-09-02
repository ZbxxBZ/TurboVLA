from __future__ import annotations

import torch
from torch import nn

from .configuration import DepthFusionConfig


_RATIO_WINDOW_MAX = 64


class _DepthGateMixin:
    config: DepthFusionConfig
    depth_gate: nn.Parameter

    def _init_depth_gate(self, config: DepthFusionConfig) -> None:
        self._gate_override: float | None = None
        self._residual_ratio_values: list[torch.Tensor] = []
        self._residual_ratio_valid_values: list[torch.Tensor] = []
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
            if self.config.gate_parameterization == "tanh" and not -1.0 <= value <= 1.0:
                raise ValueError(f"gate override {value} is outside [-1.0, 1.0]")
        self._gate_override = value

    def _match_residual_scale(self, rgb: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Match residual RMS to RGB RMS before applying the gate."""
        if not self.config.residual_scale_match:
            return delta
        rgb_rms = rgb.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
        # Keep zero residuals zero, including all-invalid views.
        scale = torch.where(
            delta_rms > 1e-6,
            rgb_rms / delta_rms.clamp_min(1e-6),
            torch.zeros_like(rgb_rms),
        )
        return delta * scale.to(dtype=delta.dtype)

    def effective_gate(self) -> torch.Tensor:
        if self._gate_override is not None:
            return torch.full_like(self.depth_gate, self._gate_override)
        if self.config.gate_parameterization == "bounded_sigmoid":
            width = self.config.gate_max - self.config.gate_min
            return self.config.gate_min + width * torch.sigmoid(self.depth_gate)
        return torch.tanh(self.depth_gate)

    def residual_ratio(self) -> torch.Tensor | None:
        """Return the fixed-size residual-ratio window since the last reset."""
        if not self._residual_ratio_values:
            return None
        return torch.cat(self._residual_ratio_values, dim=0)

    def reset_residual_ratio(self) -> None:
        self._residual_ratio_values.clear()
        self._residual_ratio_valid_values.clear()

    def residual_ratio_valid(self) -> torch.Tensor | None:
        if not self._residual_ratio_valid_values:
            return None
        return torch.cat(self._residual_ratio_valid_values, dim=0)

    def _record_residual_ratio(
        self,
        rgb_tokens: torch.Tensor,
        residual: torch.Tensor,
        valid_view_mask: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            reduce_dims = tuple(range(2, rgb_tokens.ndim))
            rgb_rms = rgb_tokens.detach().float().square().mean(dim=reduce_dims).sqrt()
            residual_rms = residual.detach().float().square().mean(dim=reduce_dims).sqrt()
            ratio = (residual_rms / rgb_rms.clamp_min(1e-6)).flatten()
            self._residual_ratio_values.append(ratio)
            if valid_view_mask is None:
                self._residual_ratio_valid_values.append(ratio)
            else:
                valid_view_mask = valid_view_mask.to(device=ratio.device, dtype=torch.bool).flatten()
                if valid_view_mask.shape != ratio.shape:
                    raise ValueError(
                        f"valid_view_mask must have shape {tuple(ratio.shape)}, "
                        f"got {tuple(valid_view_mask.shape)}"
                    )
                self._residual_ratio_valid_values.append(ratio[valid_view_mask])
            if len(self._residual_ratio_values) > _RATIO_WINDOW_MAX:
                self._residual_ratio_values.pop(0)
                self._residual_ratio_valid_values.pop(0)


class GatedAlignedDepthFusion(_DepthGateMixin, nn.Module):
    """Inject each depth token into its aligned RGB token."""

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
                f"RGB/depth token shapes must match, got {tuple(rgb_tokens.shape)} "
                f"and {tuple(depth_tokens.shape)}"
            )
        batch_size, num_views, num_tokens, hidden_dim = rgb_tokens.shape
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(f"expected token hidden dim {self.config.hidden_dim}, got {hidden_dim}")

        aligned_depth = depth_tokens.to(device=rgb_tokens.device, dtype=rgb_tokens.dtype)
        local_delta = self.delta_dropout(self.local_projection(self.depth_norm(aligned_depth)))
        valid_views = None
        if depth_invalid_mask is not None:
            expected = (batch_size, num_views, num_tokens)
            if tuple(depth_invalid_mask.shape) != expected:
                raise ValueError(f"depth_invalid_mask must be {expected}, got {tuple(depth_invalid_mask.shape)}")
            invalid_mask = depth_invalid_mask.to(device=rgb_tokens.device, dtype=torch.bool)
            local_delta = local_delta.masked_fill(invalid_mask[..., None], 0.0)
            valid_views = ~invalid_mask.all(dim=-1)

        local_delta = self._match_residual_scale(rgb_tokens, local_delta)
        gate = self.effective_gate().to(device=local_delta.device, dtype=local_delta.dtype)
        residual = gate.view(1, 1, 1, -1) * local_delta
        self._record_residual_ratio(rgb_tokens, residual, valid_view_mask=valid_views)
        return rgb_tokens + residual


class GatedDepthCrossAttention(_DepthGateMixin, nn.Module):
    """Use RGB tokens as queries and depth tokens as keys/values."""

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
            # Do not combine zero output initialization with gate_init=0 or
            # residual_scale_match: both choices remove every gradient path.
            nn.init.zeros_(self.cross_attention.out_proj.weight)
            if self.cross_attention.out_proj.bias is not None:
                nn.init.zeros_(self.cross_attention.out_proj.bias)
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
                f"RGB/depth token shapes must match, got {tuple(rgb_tokens.shape)} "
                f"and {tuple(depth_tokens.shape)}"
            )
        batch_size, num_views, num_tokens, hidden_dim = rgb_tokens.shape
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(f"expected token hidden dim {self.config.hidden_dim}, got {hidden_dim}")

        if depth_invalid_mask is None:
            flat_rgb = rgb_tokens.reshape(batch_size * num_views, num_tokens, hidden_dim)
            flat_depth = depth_tokens.reshape(batch_size * num_views, num_tokens, hidden_dim).to(
                device=flat_rgb.device, dtype=flat_rgb.dtype
            )
            normalized_depth = self.depth_norm(flat_depth)
            delta, _ = self.cross_attention(
                query=self.rgb_norm(flat_rgb),
                key=normalized_depth,
                value=normalized_depth,
                need_weights=False,
            )
            delta = delta.reshape(batch_size, num_views, num_tokens, hidden_dim)
            valid_view_mask = None
        else:
            expected = (batch_size, num_views, num_tokens)
            if tuple(depth_invalid_mask.shape) != expected:
                raise ValueError(f"depth_invalid_mask must be {expected}, got {tuple(depth_invalid_mask.shape)}")
            view_mask = depth_invalid_mask.to(device=rgb_tokens.device, dtype=torch.bool)
            active_view = int(self.config.active_view_index)
            if not 0 <= active_view < num_views:
                raise ValueError(f"active_view_index {active_view} is outside num_views={num_views}")

            # The current encoder emits depth for a fixed cam_head view. This
            # static selection avoids dynamic index construction and CUDA sync.
            view_rgb = rgb_tokens[:, active_view]
            view_depth = depth_tokens[:, active_view].to(
                device=rgb_tokens.device, dtype=rgb_tokens.dtype
            )
            active_mask = view_mask[:, active_view]
            all_invalid = active_mask.all(dim=1)

            # PyTorch MHA returns NaN when every key is masked. Temporarily
            # unmask the first key for those rows, then discard their residual
            # below. Its value is irrelevant because output is cleared; this
            # avoids cloning the complete depth tensor/key mask.
            first_mask = active_mask[:, :1] & ~all_invalid[:, None]
            active_mask = torch.cat((first_mask, active_mask[:, 1:]), dim=1)
            normalized_depth = self.depth_norm(view_depth)
            active_delta, _ = self.cross_attention(
                query=self.rgb_norm(view_rgb),
                key=normalized_depth,
                value=normalized_depth,
                key_padding_mask=active_mask,
                need_weights=False,
            )
            active_delta = active_delta.masked_fill(all_invalid[:, None, None], 0.0)
            zero_delta = torch.zeros_like(active_delta)
            delta = torch.stack(
                [active_delta if view == active_view else zero_delta for view in range(num_views)],
                dim=1,
            )
            valid_view_mask = torch.zeros(
                batch_size, num_views, dtype=torch.bool, device=rgb_tokens.device
            )
            valid_view_mask[:, active_view] = ~all_invalid

        delta = self.delta_dropout(delta)
        delta = self._match_residual_scale(rgb_tokens, delta)
        gate = self.effective_gate().to(device=delta.device, dtype=delta.dtype)
        residual = gate.view(1, 1, 1, -1) * delta
        self._record_residual_ratio(rgb_tokens, residual, valid_view_mask=valid_view_mask)
        return rgb_tokens + residual
