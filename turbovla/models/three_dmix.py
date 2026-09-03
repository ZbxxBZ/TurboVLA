"""3D-MIX semantic-conditioned gated fusion for TurboVLA.

The module follows the Gated Fusion variant described in 3D-MIX.  It keeps
the original TurboVLA VL tokens untouched and appends fused VGGT geometry
tokens to the action-head memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ThreeDMixConfig:
    """Configuration for the optional 3D-MIX bridge module."""

    enabled: bool = False
    vggt_dim: int = 2048
    semantic_pool: str = "vl"
    output_scale_init: float = 0.0

    def __post_init__(self) -> None:
        if self.vggt_dim < 1:
            raise ValueError("three_dmix.vggt_dim must be positive")
        if self.semantic_pool not in {"vl", "text"}:
            raise ValueError("three_dmix.semantic_pool must be 'vl' or 'text'")


class ThreeDMix(nn.Module):
    """Semantic-conditioned gated fusion of VL and VGGT token features.

    Args:
        hidden_dim: TurboVLA condition width ``D``.
        vggt_dim: Width of the cached/raw VGGT features.
        semantic_pool: ``"vl"`` (paper-aligned default) pools visual and
            valid text tokens; ``"text"`` is an ablation option.
        output_scale_init: Initial residual scale applied to fused tokens.
            Zero starts close to the original TurboVLA while allowing the
            scale to learn during fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int,
        vggt_dim: int,
        semantic_pool: str = "vl",
        output_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.vggt_dim = int(vggt_dim)
        self.semantic_pool = str(semantic_pool)
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.vggt_dim < 1:
            raise ValueError("vggt_dim must be positive")
        if self.semantic_pool not in {"vl", "text"}:
            raise ValueError("semantic_pool must be 'vl' or 'text'")

        # Equation (1): F_geo = W_proj F_VGGT.
        self.vggt_projection = nn.Linear(self.vggt_dim, self.hidden_dim)
        # Equation (3): G = sigmoid(W_gate [S_broadcast; F_geo]).
        self.gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        # Equation (4): independently project semantic and geometry branches.
        self.semantic_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.geometry_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool ``tokens`` over sequence positions marked as valid."""
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,L,D], got {tuple(tokens.shape)}")
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError(
                "valid_mask must match the first two token dimensions, "
                f"got {tuple(valid_mask.shape)} for {tuple(tokens.shape)}"
            )
        weights = valid_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tokens * weights).sum(dim=1, keepdim=True) / denom

    def _semantic_summary(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens and text_tokens must be [B,L,D], "
                f"got {tuple(visual_tokens.shape)} and {tuple(text_tokens.shape)}"
            )
        if visual_tokens.shape[0] != text_tokens.shape[0] or visual_tokens.shape[2] != text_tokens.shape[2]:
            raise ValueError("visual_tokens and text_tokens must agree on batch and hidden dimensions")
        if text_key_padding_mask.shape != text_tokens.shape[:2]:
            raise ValueError(
                "text_key_padding_mask must match text_tokens, "
                f"got {tuple(text_key_padding_mask.shape)} and {tuple(text_tokens.shape)}"
            )
        padding_mask = text_key_padding_mask.to(device=visual_tokens.device).bool()

        if self.semantic_pool == "text":
            return self._masked_mean(text_tokens, ~padding_mask)

        vl_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_valid = torch.ones(
            visual_tokens.shape[:2], device=visual_tokens.device, dtype=torch.bool
        )
        valid_mask = torch.cat([visual_valid, ~padding_mask], dim=1)
        return self._masked_mean(vl_tokens, valid_mask)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        vggt_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused geometry tokens and their gates.

        Args:
            visual_tokens: Final TurboVLA VL-interaction visual tokens,
                ``[B, L_v, D]``.
            text_tokens: Final TurboVLA VL-interaction text tokens,
                ``[B, L_t, D]``.
            text_key_padding_mask: ``True`` at padded text positions,
                ``[B, L_t]``.
            vggt_features: Native-resolution or pre-pooled VGGT patch tokens,
                ``[B, N, C_vggt]``. A ``[B, V, N, C_vggt]`` input is also
                accepted and flattened across views.

        Returns:
            ``(fused_tokens, gates)`` with both tensors shaped ``[B, N, D]``.
        """
        if vggt_features is None:
            raise ValueError("ThreeDMix requires samples['vggt'] features")
        if vggt_features.ndim == 4:
            vggt_features = vggt_features.flatten(1, 2)
        if vggt_features.ndim != 3:
            raise ValueError(
                "vggt_features must be [B,N,C] or [B,V,N,C], "
                f"got {tuple(vggt_features.shape)}"
            )
        if vggt_features.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "vggt_features and VL tokens must have the same batch size, "
                f"got {vggt_features.shape[0]} and {visual_tokens.shape[0]}"
            )
        if vggt_features.shape[-1] != self.vggt_dim:
            raise ValueError(
                f"expected VGGT feature width {self.vggt_dim}, got {vggt_features.shape[-1]}"
            )

        vggt_features = vggt_features.to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        f_geo = self.vggt_projection(vggt_features)
        s_global = self._semantic_summary(
            visual_tokens,
            text_tokens,
            text_key_padding_mask,
        )
        s_broadcast = s_global.expand(-1, f_geo.shape[1], -1)

        gate_input = torch.cat([s_broadcast, f_geo], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        semantic_part = self.semantic_projection(s_broadcast)
        geometry_part = self.geometry_projection(f_geo)
        fused = gates * semantic_part + (1.0 - gates) * geometry_part
        fused = fused * self.output_scale.to(device=fused.device, dtype=fused.dtype)
        return fused, gates
