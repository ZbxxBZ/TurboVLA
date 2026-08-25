from __future__ import annotations

"""Frozen VGGT RGB-to-depth adapter for the TurboVLA depth branch.

VGGT is intentionally imported lazily.  RGB-only TurboVLA users and the
original DINOv3 depth backend therefore do not need the VGGT repository or its
larger dependencies installed.
"""

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import DepthEncoderConfig


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return value + torch.log(-torch.expm1(torch.tensor(-value))).item()


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"VGGT checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "vggt"):
        candidate = state.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    tensors = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("VGGT checkpoint does not contain tensor parameters")
    for prefix in ("module.", "model.", "vggt."):
        if tensors and all(key.startswith(prefix) for key in tensors):
            tensors = {key[len(prefix) :]: value for key, value in tensors.items()}
    return tensors


def _load_vggt(config: DepthEncoderConfig) -> nn.Module:
    repo_path = Path(config.vggt_repo_path)
    weights_path = Path(config.vggt_weights_path)
    if not (repo_path / "vggt" / "models" / "vggt.py").is_file():
        raise FileNotFoundError(
            "VGGT repository must contain vggt/models/vggt.py; "
            f"set depth.vggt_repo_path to the cloned repository, got {repo_path}"
        )
    if not weights_path.is_file():
        raise FileNotFoundError(f"VGGT weights not found: {weights_path}")

    repo_import_path = str(repo_path.resolve())
    if repo_import_path not in sys.path:
        sys.path.insert(0, repo_import_path)
    try:
        vggt_module = importlib.import_module("vggt.models.vggt")
        vggt_class = getattr(vggt_module, "VGGT")
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "could not import VGGT; install its dependencies and verify depth.vggt_repo_path"
        ) from error

    model = vggt_class(
        img_size=config.vggt_image_size,
        patch_size=config.vggt_patch_size,
        enable_camera=False,
        enable_point=False,
        enable_depth=True,
        enable_track=False,
    )
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_unexpected = ("camera_head.", "point_head.", "track_head.")
    unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected)]
    if missing or unexpected:
        raise RuntimeError(
            "VGGT checkpoint does not match the configured model: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model


class VGGTDepthEncoder(nn.Module):
    """Encode only cam_head RGB into geometry tokens using a frozen VGGT.

    The policy receives ImageNet-normalized RGB.  VGGT expects RGB in [0, 1],
    so this module reverses that normalization internally and applies VGGT's
    own normalization.  Only the head-view output is exposed as policy depth
    tokens; wrist views are masked in ``forward`` for interface compatibility.
    """

    def __init__(self, config: DepthEncoderConfig, *, vggt: nn.Module | None = None) -> None:
        super().__init__()
        if str(config.backend).lower() != "vggt":
            raise ValueError(f"VGGTDepthEncoder requires backend='vggt', got {config.backend!r}")
        self.config = config
        self.patch_grid = config.image_size // config.patch_size
        self.num_patches = self.patch_grid**2
        if self.patch_grid < 1 or config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        self.vggt = vggt if vggt is not None else _load_vggt(config)
        self.depth_patch_embedding = nn.Conv2d(
            3,
            config.feature_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.token_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.token_norm = nn.LayerNorm(config.hidden_dim)

        if config.learn_metric_calibration:
            self.metric_scale_raw = nn.Parameter(torch.tensor(_inverse_softplus(config.metric_scale_init)))
            self.metric_shift = nn.Parameter(torch.tensor(float(config.metric_shift_init)))
        else:
            self.register_buffer("metric_scale_raw", torch.tensor(_inverse_softplus(config.metric_scale_init)))
            self.register_buffer("metric_shift", torch.tensor(float(config.metric_shift_init)))

        if config.adapter_weights_path:
            self.load_adapter_checkpoint(config.adapter_weights_path)

        if config.freeze_backbone:
            self.vggt.requires_grad_(False)
        if config.frozen:
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> VGGTDepthEncoder:
        super().train(mode)
        if self.config.freeze_backbone:
            self.vggt.eval()
        return self

    @property
    def metric_scale(self) -> torch.Tensor:
        return F.softplus(self.metric_scale_raw)

    def adapter_state_dict(self) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        state = {
            "depth_patch_embedding": self.depth_patch_embedding.state_dict(),
            "token_projection": self.token_projection.state_dict(),
            "token_norm": self.token_norm.state_dict(),
            "metric_scale_raw": self.metric_scale_raw.detach().cpu(),
            "metric_shift": self.metric_shift.detach().cpu(),
        }
        return state

    def load_adapter_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"VGGT adapter checkpoint must be a mapping, got {type(checkpoint).__name__}")
        state = checkpoint.get("vggt_adapter", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("VGGT adapter checkpoint must contain a vggt_adapter mapping")
        for name, module in (
            ("depth_patch_embedding", self.depth_patch_embedding),
            ("token_projection", self.token_projection),
            ("token_norm", self.token_norm),
        ):
            module_state = state.get(name)
            if not isinstance(module_state, dict):
                raise ValueError(f"VGGT adapter checkpoint is missing {name}")
            module.load_state_dict(module_state, strict=True)
        for name, parameter in (("metric_scale_raw", self.metric_scale_raw), ("metric_shift", self.metric_shift)):
            value = state.get(name)
            if value is not None:
                parameter.data.copy_(torch.as_tensor(value, dtype=parameter.dtype))

    def _validate_head_rgb(self, head_rgb: torch.Tensor) -> torch.Tensor:
        if head_rgb.ndim != 4 or head_rgb.shape[1] != 3:
            raise ValueError(f"head_rgb must be [B,3,H,W], got {tuple(head_rgb.shape)}")
        if head_rgb.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(
                f"expected head RGB size {(self.config.image_size, self.config.image_size)}, "
                f"got {tuple(head_rgb.shape[-2:])}"
            )
        return head_rgb

    def select_head_rgb(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        if pixel_values.shape[1] != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} RGB views, got {pixel_values.shape[1]}")
        return self._validate_head_rgb(pixel_values[:, self.config.head_camera_index])

    def _vggt_input(self, head_rgb: torch.Tensor) -> torch.Tensor:
        if self.config.vggt_input_is_normalized:
            mean = IMAGENET_MEAN.to(device=head_rgb.device, dtype=head_rgb.dtype)
            std = IMAGENET_STD.to(device=head_rgb.device, dtype=head_rgb.dtype)
            head_rgb = head_rgb * std + mean
        head_rgb = head_rgb.clamp(0.0, 1.0)
        resized = F.interpolate(
            head_rgb,
            size=(self.config.vggt_image_size, self.config.vggt_image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized[:, None]

    def _run_vggt(self, head_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._vggt_input(head_rgb)
        context = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with context:
            predictions = self.vggt(images)
        if not isinstance(predictions, dict) or "depth" not in predictions or "depth_conf" not in predictions:
            raise RuntimeError("VGGT must return a dict containing depth and depth_conf")
        depth = predictions["depth"]
        confidence = predictions["depth_conf"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 5 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if confidence.ndim == 5 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if depth.ndim != 4 or confidence.ndim != 4:
            raise RuntimeError(
                f"unexpected VGGT output shapes: depth={tuple(depth.shape)}, confidence={tuple(confidence.shape)}"
            )
        return depth[:, 0:1].float(), confidence[:, 0:1].float()

    def _geometry_channels(self, depth: torch.Tensor, confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.metric_scale.to(dtype=depth.dtype, device=depth.device)
        shift = self.metric_shift.to(dtype=depth.dtype, device=depth.device)
        metric_depth = scale * depth + shift
        valid = (
            torch.isfinite(metric_depth)
            & (metric_depth >= self.config.min_depth_m)
            & torch.isfinite(confidence)
        )
        # Do not clamp the upper range before the adapter sees the value: a
        # relative-depth VGGT may need gradients through the learned scale.
        # The patch target and final prediction paths still enforce max_depth_m.
        metric_depth = metric_depth.clamp_min(self.config.min_depth_m)
        log_min = torch.log(torch.tensor(self.config.min_depth_m, device=depth.device, dtype=depth.dtype))
        log_max = torch.log(torch.tensor(self.config.max_depth_m, device=depth.device, dtype=depth.dtype))
        log_depth = (torch.log(metric_depth) - log_min) / (log_max - log_min)
        confidence = confidence.clamp_min(0.0)
        confidence = confidence / (1.0 + confidence)
        valid_float = valid.to(dtype=log_depth.dtype)
        channels = torch.cat((log_depth * valid_float, confidence * valid_float, valid_float), dim=1)
        channels = F.interpolate(
            channels,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return channels, valid_float

    def encode_head_features(self, head_rgb: torch.Tensor) -> torch.Tensor:
        """Return the trainable patch feature map before token projection."""
        head_rgb = self._validate_head_rgb(head_rgb)
        depth, confidence = self._run_vggt(head_rgb)
        channels, _ = self._geometry_channels(depth, confidence)
        return self.depth_patch_embedding(channels)

    def predict_head_depth(self, head_rgb: torch.Tensor) -> torch.Tensor:
        """Return VGGT's calibrated metric depth at the policy image resolution."""
        head_rgb = self._validate_head_rgb(head_rgb)
        depth, _ = self._run_vggt(head_rgb)
        scale = self.metric_scale.to(dtype=depth.dtype, device=depth.device)
        shift = self.metric_shift.to(dtype=depth.dtype, device=depth.device)
        depth = (scale * depth + shift).clamp(self.config.min_depth_m, self.config.max_depth_m)
        return F.interpolate(depth, size=(self.config.image_size, self.config.image_size), mode="bilinear", align_corners=False)

    def _encode_head_tokens_and_mask(self, head_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_rgb = self._validate_head_rgb(head_rgb)
        depth, confidence = self._run_vggt(head_rgb)
        channels, valid = self._geometry_channels(depth, confidence)
        features = self.depth_patch_embedding(channels)
        if features.shape[-2:] != (self.patch_grid, self.patch_grid):
            features = F.adaptive_avg_pool2d(features, (self.patch_grid, self.patch_grid))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_norm(self.token_projection(tokens))
        valid_fraction = F.adaptive_avg_pool2d(valid, (self.patch_grid, self.patch_grid)).flatten(1)
        invalid = valid_fraction < self.config.min_valid_fraction
        return tokens.masked_fill(invalid.unsqueeze(-1), 0.0), invalid

    def encode_head_tokens(self, head_rgb: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._encode_head_tokens_and_mask(head_rgb)
        return tokens

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_tokens, head_invalid = self._encode_head_tokens_and_mask(self.select_head_rgb(pixel_values))
        tokens_by_view = []
        masks_by_view = []
        for view_index in range(self.config.num_views):
            if view_index == self.config.head_camera_index:
                tokens_by_view.append(head_tokens)
                masks_by_view.append(head_invalid)
            else:
                tokens_by_view.append(torch.zeros_like(head_tokens))
                masks_by_view.append(torch.ones_like(head_invalid))
        return torch.stack(tokens_by_view, dim=1), torch.stack(masks_by_view, dim=1)
