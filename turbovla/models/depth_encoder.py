from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import DepthEncoderConfig
from .depth_dinov3 import DepthHeadLite


class DINOv3Backbone(nn.Module):
    """Raessan-compatible wrapper returning the final spatial DINOv3 feature map."""

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.dino = model
        self.num_layers = num_layers

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.dino.get_intermediate_layers(
            images,
            n=range(self.num_layers),
            reshape=True,
            norm=True,
        )
        return features[-1]


def _load_dinov3(config: DepthEncoderConfig) -> nn.Module:
    repo_path = Path(config.backbone_repo_path)
    weights_path = Path(config.backbone_weights_path)
    if not (repo_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 repository must contain hubconf.py: {repo_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 depth-backbone weights not found: {weights_path}")
    repo_import_path = str(repo_path.resolve())
    if repo_import_path not in sys.path:
        sys.path.insert(0, repo_import_path)
    backbones = importlib.import_module("dinov3.hub.backbones")
    try:
        builder = getattr(backbones, config.backbone_name)
    except AttributeError as error:
        raise ValueError(f"unknown DINOv3 backbone: {config.backbone_name}") from error
    model = builder(pretrained=True, weights=str(weights_path.resolve()))
    return DINOv3Backbone(model, config.backbone_num_layers)


def _depth_head_state(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a DepthHeadLite state dict from Raessan or local checkpoints."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"depth-head checkpoint must be a mapping, got {type(checkpoint).__name__}")

    state = checkpoint
    for key in ("depth_head", "model_state_dict", "state_dict", "model"):
        candidate = state.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break

    tensor_state = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensor_state:
        raise ValueError("depth-head checkpoint does not contain tensor parameters")

    prefixes = ("module.depth_head.", "model.depth_head.", "depth_head.", "module.")
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in tensor_state):
            return {key[len(prefix) :]: value for key, value in tensor_state.items()}
    return tensor_state


def _load_projection_checkpoint(
    path: str,
    token_projection: nn.Linear,
    token_norm: nn.LayerNorm,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"projection checkpoint must be a mapping, got {type(checkpoint).__name__}")
    projection_state = checkpoint.get("token_projection")
    norm_state = checkpoint.get("token_norm")
    if not isinstance(projection_state, dict) or not isinstance(norm_state, dict):
        raise ValueError("projection checkpoint must contain token_projection and token_norm")
    token_projection.load_state_dict(projection_state, strict=True)
    token_norm.load_state_dict(norm_state, strict=True)


class DINOv3DepthEncoder(nn.Module):
    """Encode only cam_head RGB into depth-supervised tokens for TurboVLA."""

    def __init__(
        self,
        config: DepthEncoderConfig,
        *,
        backbone: nn.Module | None = None,
        depth_head: DepthHeadLite | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_grid = config.image_size // config.patch_size
        self.num_patches = self.patch_grid**2

        self.backbone = backbone if backbone is not None else _load_dinov3(config)
        self.depth_head = (
            depth_head
            if depth_head is not None
            else DepthHeadLite(
                in_ch=config.backbone_hidden_dim,
                out_size=(config.image_size, config.image_size),
                common_ch=config.feature_dim,
                dropout=config.dropout,
            )
        )
        if depth_head is None and config.head_weights_path:
            checkpoint = torch.load(config.head_weights_path, map_location="cpu", weights_only=True)
            self.depth_head.load_state_dict(_depth_head_state(checkpoint), strict=True)

        self.token_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.token_norm = nn.LayerNorm(config.hidden_dim)
        if config.projection_weights_path:
            _load_projection_checkpoint(
                config.projection_weights_path,
                self.token_projection,
                self.token_norm,
            )

        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        if config.freeze_depth_head:
            self.depth_head.requires_grad_(False)
        if config.frozen:
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> DINOv3DepthEncoder:
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        if self.config.freeze_depth_head:
            self.depth_head.eval()
        return self

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

    def _backbone_features(self, head_rgb: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with context:
            return self.backbone(head_rgb)

    def encode_head_features(self, head_rgb: torch.Tensor) -> torch.Tensor:
        """Return the depth-supervised 2-D feature map before scalar depth prediction."""
        head_rgb = self._validate_head_rgb(head_rgb)
        backbone_features = self._backbone_features(head_rgb)
        context = torch.no_grad() if self.config.freeze_depth_head else nullcontext()
        with context:
            return self.depth_head.forward_features(backbone_features)

    def predict_head_depth(self, head_rgb: torch.Tensor) -> torch.Tensor:
        """Predict metric cam_head depth for the stage-one supervision objective."""
        head_rgb = self._validate_head_rgb(head_rgb)
        backbone_features = self._backbone_features(head_rgb)
        context = torch.no_grad() if self.config.freeze_depth_head else nullcontext()
        with context:
            return self.depth_head(backbone_features)

    def geometry_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_head_features(self.select_head_rgb(pixel_values))

    def predict_depth(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compatibility helper accepting the policy's full multi-view RGB tensor."""
        return self.predict_head_depth(self.select_head_rgb(pixel_values))

    def project_geometry_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project a fused geometry map to the exact tokens consumed by policy fusion."""
        if features.ndim != 4 or features.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"features must be [B,{self.config.feature_dim},H,W], got {tuple(features.shape)}"
            )
        features = F.adaptive_avg_pool2d(features, (self.patch_grid, self.patch_grid))
        tokens = features.flatten(2).transpose(1, 2)
        return self.token_norm(self.token_projection(tokens))

    def encode_head_tokens(self, head_rgb: torch.Tensor) -> torch.Tensor:
        """Encode cam_head RGB to the 196 projected tokens used in stage two."""
        return self.project_geometry_features(self.encode_head_features(head_rgb))

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_tokens = self.encode_head_tokens(self.select_head_rgb(pixel_values))

        tokens_by_view = []
        masks_by_view = []
        for view_index in range(self.config.num_views):
            if view_index == self.config.head_camera_index:
                tokens_by_view.append(head_tokens)
                masks_by_view.append(
                    torch.zeros(head_tokens.shape[:2], dtype=torch.bool, device=head_tokens.device)
                )
            else:
                tokens_by_view.append(torch.zeros_like(head_tokens))
                masks_by_view.append(
                    torch.ones(head_tokens.shape[:2], dtype=torch.bool, device=head_tokens.device)
                )
        return torch.stack(tokens_by_view, dim=1), torch.stack(masks_by_view, dim=1)


def build_depth_encoder(config: DepthEncoderConfig) -> nn.Module:
    """Build the configured depth backend while keeping old DINOv3 configs valid."""
    backend = str(config.backend).lower()
    if backend == "dinov3":
        return DINOv3DepthEncoder(config)
    if backend == "vggt":
        from .vggt_depth_encoder import VGGTDepthEncoder

        return VGGTDepthEncoder(config)
    raise ValueError(f"unsupported depth encoder backend: {config.backend!r}")
