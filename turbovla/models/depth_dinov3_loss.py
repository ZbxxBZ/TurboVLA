"""Depth losses used by Raessan/depth_dinov3."""

from __future__ import annotations

import torch
from torch import nn


class GradientLoss(nn.Module):
    def __init__(self, valid_mask: bool = True, loss_weight: float = 1.0, max_depth: float | None = None) -> None:
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.eps = 0.001

    def _one_scale(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.valid_mask:
            mask = target > 0
            if self.max_depth is not None:
                mask = mask & (target <= self.max_depth)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
        count = mask.sum().clamp_min(1)
        difference = (prediction + self.eps).log() - (target + self.eps).log()
        difference = difference * mask

        vertical = (difference[:-2] - difference[2:]).abs()
        vertical = vertical * (mask[:-2] & mask[2:])
        horizontal = (difference[:, :-2] - difference[:, 2:]).abs()
        horizontal = horizontal * (mask[:, :-2] & mask[:, 2:])
        return (vertical.sum() + horizontal.sum()) / count

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = prediction.new_zeros(())
        for scale in range(4):
            stride = 1 if scale == 0 else 2 * scale
            loss = loss + self._one_scale(prediction[::stride, ::stride], target[::stride, ::stride])
        return self.loss_weight * loss


class SigLoss(nn.Module):
    def __init__(self, valid_mask: bool = True, loss_weight: float = 1.0, max_depth: float | None = None) -> None:
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.eps = 0.001

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.valid_mask:
            mask = target > 0
            if self.max_depth is not None:
                mask = mask & (target <= self.max_depth)
            prediction = prediction[mask]
            target = target[mask]
        difference = (prediction + self.eps).log() - (target + self.eps).log()
        value = difference.var(unbiased=False) + 0.15 * difference.mean().square()
        return self.loss_weight * value.sqrt()


class DepthLoss(nn.Module):
    """Raessan's masked L1 + scale-invariant + multiscale gradient objective."""

    def __init__(self, w_data: float = 1.0, w_sig: float = 1.0, w_grad: float = 5.0) -> None:
        super().__init__()
        self.w_data = float(w_data)
        self.sig = SigLoss(loss_weight=w_sig)
        self.grad = GradientLoss(loss_weight=w_grad)

    @staticmethod
    def _to_bhw(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.unsqueeze(0)
        if values.ndim == 4 and values.shape[1] == 1:
            return values.squeeze(1)
        if values.ndim != 3:
            raise ValueError(f"expected [H,W], [B,H,W], or [B,1,H,W], got {tuple(values.shape)}")
        return values

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self._to_bhw(prediction)
        target = self._to_bhw(target)
        mask = target > 0
        l1 = self.w_data * ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1)
        sig = self.sig(prediction, target)
        gradient = torch.stack(
            [self.grad(prediction[index], target[index]) for index in range(prediction.shape[0])]
        ).mean()
        return l1 + sig + gradient, l1, sig, gradient
