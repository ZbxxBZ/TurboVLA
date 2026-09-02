from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .action_head import TurboVLAActionHead
from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import _get_clones
from .configuration import (
    ActionHeadConfig,
    DepthEncoderConfig,
    DepthFusionConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from .depth_encoder import DINOv3DepthEncoder
from .vggt_depth_encoder import VGGTDepthEncoder
from .depth_fusion import GatedAlignedDepthFusion, GatedDepthCrossAttention
from .text_encoder import TurboVLATextEncoder
from .vision_encoder import DINOv3VisionEncoder


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class VisionLanguageInteraction(nn.Module):
    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        text_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, config.nheads // 2),
            dim_feedforward=config.enhancer_inner_dim,
            dropout=config.text_dropout,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=config.hidden_dim,
            l_dim=config.hidden_dim,
            embed_dim=config.enhancer_inner_dim,
            num_heads=max(1, config.nheads // 2),
            dropout=config.fusion_dropout,
            drop_path=config.fusion_droppath,
            residual_style=config.residual_style,
            attention_backend=config.attention_backend,
        )
        self.text_layers = _get_clones(text_layer, config.num_layers)
        self.fusion_layers = _get_clones(fusion_layer, config.num_layers)
        self.padding_strategy = config.padding_strategy

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_fill = self.padding_strategy == "zero_fill"
        if zero_fill:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

        for fusion_layer, text_layer in zip(self.fusion_layers, self.text_layers):
            visual_tokens, text_tokens = fusion_layer(
                v=visual_tokens,
                l=text_tokens,
                attention_mask_v=None,
                attention_mask_l=text_key_padding_mask,
            )
            source_mask = None if text_self_attention_masks is None else ~text_self_attention_masks
            text_tokens = text_layer(
                src=text_tokens.transpose(0, 1),
                src_mask=source_mask,
                src_key_padding_mask=None if zero_fill else text_key_padding_mask,
                pos=None,
            ).transpose(0, 1)
            if zero_fill:
                text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        return visual_tokens, text_tokens


class TurboVLA(nn.Module):
    """Shared TurboVLA architecture for LIBERO and RoboTwin."""

    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.interaction.hidden_dim
        self.action_dim = int(config.action.action_dim)
        self.chunk_size = int(config.action.horizon)
        self.state_dim = int(config.action.state_dim)
        self.num_views = int(config.vision.num_views)

        self.text_encoder = TurboVLATextEncoder(config.text, hidden_dim=hidden_dim)
        self.vision_encoder = DINOv3VisionEncoder(config.vision)
        self.vision_projection = VisionProjection(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.vision_encoder.hidden_size // 2),
            dropout=config.vision.dropout,
        )

        self.depth_encoder: DINOv3DepthEncoder | VGGTDepthEncoder | None = None
        self.depth_fusion: GatedAlignedDepthFusion | GatedDepthCrossAttention | None = None
        if config.depth.enabled:
            if config.depth.patch_size != self.vision_encoder.patch_size:
                raise ValueError(
                    "depth.patch_size must match the DINOv3 patch size so RGB/depth tokens stay aligned"
                )
            if config.depth.image_size != config.vision.image_size:
                raise ValueError("depth.image_size must match vision.image_size")
            # 深度分支只在显式启用的配方中创建；RGB-only 模型的参数名和行为保持不变。
            depth_backend = str(config.depth.backend).lower()
            if depth_backend == "dinov3":
                self.depth_encoder = DINOv3DepthEncoder(config.depth)
            elif depth_backend == "vggt":
                self.depth_encoder = VGGTDepthEncoder(config.depth)
            else:
                raise ValueError(f"unsupported depth encoder backend: {config.depth.backend!r}")
            if config.depth_fusion.mode == "aligned":
                self.depth_fusion = GatedAlignedDepthFusion(config.depth_fusion)
            else:
                self.depth_fusion = GatedDepthCrossAttention(config.depth_fusion)

        if config.vision.position_embedding == "learned_patch":
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_views, self.vision_encoder.num_patches, hidden_dim)
            )
            self.patch_position_scale = nn.Parameter(
                torch.full((1, self.num_views, 1, 1), float(config.vision.position_scale_init))
            )
            nn.init.trunc_normal_(self.patch_position_embedding, std=config.vision.position_init_std)
        else:
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, hidden_dim))
            self.register_parameter("patch_position_embedding", None)
            self.register_parameter("patch_position_scale", None)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        self.vision_language_interaction = VisionLanguageInteraction(config.interaction)
        self.action_head = TurboVLAActionHead(
            config=config.action,
            hidden_dim=hidden_dim,
            nheads=config.interaction.nheads,
            dim_feedforward=config.interaction.dim_feedforward,
        )

    def _normalize_samples(self, samples: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, Mapping):
            if "dinov3" not in samples:
                raise ValueError("samples mapping must contain 'dinov3'")
            pixel_values = samples["dinov3"]
        else:
            pixel_values = samples
        if pixel_values.ndim == 6:
            pixel_values = pixel_values[:, -1]
        if pixel_values.ndim != 5:
            raise ValueError(f"samples must be [B,V,3,H,W] or [B,T,V,3,H,W], got {tuple(pixel_values.shape)}")
        if pixel_values.shape[1] != self.num_views:
            raise ValueError(f"expected {self.num_views} RGB views, got {pixel_values.shape[1]}")
        return pixel_values

    def _position_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.vision.position_embedding == "learned_patch":
            if tokens.shape[2] != self.patch_position_embedding.shape[2]:
                raise ValueError(
                    f"configured patch position length {self.patch_position_embedding.shape[2]} "
                    f"does not match encoded length {tokens.shape[2]}"
                )
            position = self.patch_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
            scale = self.patch_position_scale.to(device=tokens.device, dtype=tokens.dtype)
            view = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
            return tokens + scale * position + view
        view = self.view_embedding[:, :, None, :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view

    def encode_vision(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.vision_encoder(pixel_values)
        tokens = tokens.to(dtype=self.vision_projection.skip.weight.dtype)
        tokens = self.vision_projection(tokens)
        tokens = self._position_visual_tokens(tokens)
        return tokens.flatten(1, 2)

    def encode_depth(
        self,
        pixel_values: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.depth_encoder is None:
            raise RuntimeError("encode_depth requires the depth branch to be enabled")

        depth_tokens, depth_invalid_mask = self.depth_encoder(pixel_values.to(device))
        depth_tokens = depth_tokens.to(device=device, dtype=dtype)
        if self.config.depth_fusion.mode == "global":
            depth_tokens = self._position_visual_tokens(depth_tokens)
        return depth_tokens, depth_invalid_mask.to(device=device)

    def encode_condition(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions,
                device=device,
            )
            if text_tokens.shape[0] != pixel_values.shape[0]:
                raise ValueError("instruction batch size does not match image batch size")
            visual_tokens = self.encode_vision(pixel_values)
            visual_tokens, text_tokens = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
            )

            # The depth query must already carry task-language context.
            if self.config.depth.enabled:
                if self.depth_fusion is None:
                    raise RuntimeError("depth fusion is missing while the depth branch is enabled")
                depth_tokens, depth_invalid_mask = self.encode_depth(
                    pixel_values,
                    device=visual_tokens.device,
                    dtype=visual_tokens.dtype,
                )
                batch_size, num_depth_views, num_depth_tokens, hidden_dim = depth_tokens.shape
                expected_visual_tokens = num_depth_views * num_depth_tokens
                if visual_tokens.shape != (batch_size, expected_visual_tokens, hidden_dim):
                    raise ValueError(
                        "vision-language tokens cannot be aligned with depth tokens: "
                        f"got {tuple(visual_tokens.shape)} and {tuple(depth_tokens.shape)}"
                    )
                visual_tokens = visual_tokens.reshape(
                    batch_size,
                    num_depth_views,
                    num_depth_tokens,
                    hidden_dim,
                )
                visual_tokens = self.depth_fusion(
                    visual_tokens,
                    depth_tokens,
                    depth_invalid_mask,
                ).flatten(1, 2)
            return torch.cat([visual_tokens, text_tokens], dim=1)

    def forward(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode_condition(instructions, samples)
        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        return self.action_head(condition.to(dtype=action_dtype), state.to(dtype=action_dtype))

    # Transitional read-only names used only by legacy checkpoint initialization.
    @property
    def dinov3(self):
        return self.vision_encoder.backbone

    @property
    def text_proj(self):
        return self.text_encoder.text_projection

    @property
    def vision_proj(self):
        return self.vision_projection

    @property
    def feature_enhancer(self):
        return self.vision_language_interaction

    @property
    def state_proj(self):
        return self.action_head.state_projection

    @property
    def action_policy(self):
        return self.action_head.decoder


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_turbovla(args: TurboVLAConfig | Mapping[str, Any] | Any) -> TurboVLA:
    if isinstance(args, TurboVLAConfig):
        config = args
    elif isinstance(args, Mapping):
        config = TurboVLAConfig.from_mapping(args)
    else:
        config = TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=_arg(args, "bert_path", "bert-base-uncased"),
                max_length=int(_arg(args, "max_text_len", 256)),
                padding_length=_arg(args, "text_padding_length", None),
                padding_length_by_instruction=dict(_arg(args, "text_padding_length_by_instruction", {})),
                sub_sentence_present=bool(_arg(args, "sub_sentence_present", True)),
                frozen=bool(_arg(args, "freeze_text_encoder", True)),
                force_eval_when_frozen=True,
                zero_padded_tokens=bool(_arg(args, "zero_padded_text", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "text_attention_implementation", None),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=_arg(args, "dinov3_path", _arg(args, "LOCAL_DINOV3_PATH", "")),
                image_size=int(_arg(args, "image_size", _arg(args, "expected_image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                position_embedding=str(_arg(args, "position_embedding", "view")),
                encode_views_separately=bool(_arg(args, "encode_views_separately", True)),
                frozen=bool(_arg(args, "freeze_vision_encoder", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "vision_attention_implementation", None),
                compute_precision=str(_arg(args, "dinov3_precision", "bf16_autocast")),
                dropout=float(_arg(args, "vision_dropout", 0.1)),
            ),
            depth=DepthEncoderConfig(
                enabled=bool(_arg(args, "depth_enabled", False)),
                backend=str(_arg(args, "depth_backend", "dinov3")),
                stage1_mode=str(_arg(args, "depth_stage1_mode", "legacy_patch")),
                image_size=int(_arg(args, "depth_image_size", _arg(args, "image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                patch_size=int(_arg(args, "depth_patch_size", 16)),
                head_camera_index=int(_arg(args, "depth_head_camera_index", 0)),
                backbone_name=str(_arg(args, "depth_backbone_name", "dinov3_vits16plus")),
                backbone_repo_path=str(_arg(args, "depth_backbone_repo_path", "")),
                backbone_weights_path=str(_arg(args, "depth_backbone_weights_path", "")),
                backbone_num_layers=int(_arg(args, "depth_backbone_num_layers", 12)),
                backbone_hidden_dim=int(_arg(args, "depth_backbone_hidden_dim", 384)),
                head_weights_path=str(_arg(args, "depth_head_weights_path", "")),
                projection_weights_path=str(_arg(args, "depth_projection_weights_path", "")),
                adapter_weights_path=str(_arg(args, "depth_adapter_weights_path", "")),
                feature_dim=int(_arg(args, "depth_feature_dim", 160)),
                dpt_feature_dim=int(_arg(args, "depth_dpt_feature_dim", 256)),
                freeze_backbone=bool(_arg(args, "freeze_depth_backbone", True)),
                freeze_depth_head=bool(_arg(args, "freeze_depth_head", True)),
                frozen=bool(_arg(args, "freeze_depth_encoder", False)),
                dropout=float(_arg(args, "depth_dropout", 0.0)),
                vggt_repo_path=str(_arg(args, "vggt_repo_path", "")),
                vggt_weights_path=str(_arg(args, "vggt_weights_path", "")),
                vggt_image_size=int(_arg(args, "vggt_image_size", 518)),
                vggt_patch_size=int(_arg(args, "vggt_patch_size", 14)),
                vggt_input_is_normalized=bool(_arg(args, "vggt_input_is_normalized", True)),
                min_depth_m=float(_arg(args, "depth_min_m", 0.001)),
                max_depth_m=float(_arg(args, "depth_max_m", 5.0)),
                min_valid_fraction=float(_arg(args, "depth_min_valid_fraction", 0.5)),
                learn_metric_calibration=bool(_arg(args, "vggt_learn_metric_calibration", True)),
                metric_scale_init=float(_arg(args, "vggt_metric_scale_init", 1.0)),
                metric_shift_init=float(_arg(args, "vggt_metric_shift_init", 0.0)),
            ),
            depth_fusion=DepthFusionConfig(
                enabled=bool(_arg(args, "depth_enabled", False)),
                mode=str(_arg(args, "depth_fusion_mode", "global")),
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "depth_nheads", _arg(args, "nheads", 8))),
                dropout=float(_arg(args, "depth_fusion_dropout", 0.0)),
                gate_init=float(_arg(args, "depth_gate_init", 0.0)),
                gate_parameterization=str(_arg(args, "depth_gate_parameterization", "tanh")),
                gate_min=float(_arg(args, "depth_gate_min", 0.0)),
                gate_max=float(_arg(args, "depth_gate_max", 1.0)),
                residual_scale_match=bool(_arg(args, "depth_residual_scale_match", True)),
                active_view_index=int(_arg(args, "depth_head_camera_index", 0)),
                zero_init_output=bool(_arg(args, "depth_zero_init_output", False)),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "nheads", 8)),
                num_layers=int(_arg(args, "vla_feature_enhancer_layers", 6)),
                dim_feedforward=int(_arg(args, "dim_feedforward", 2048)),
                enhancer_inner_dim=int(_arg(args, "enhancer_inner_dim", 1024)),
                text_dropout=float(_arg(args, "text_dropout", 0.0)),
                fusion_dropout=float(_arg(args, "fusion_dropout", 0.0)),
                fusion_droppath=float(_arg(args, "fusion_droppath", 0.1)),
                padding_strategy=str(_arg(args, "padding_strategy", "key_padding_mask")),
                residual_style=str(_arg(args, "residual_style", "normalized")),
                attention_backend=str(_arg(args, "attention_backend", "manual")),
                compute_precision=str(_arg(args, "interaction_precision", "fp32")),
            ),
            action=ActionHeadConfig(
                action_dim=int(_arg(args, "action_dim", 7)),
                state_dim=int(_arg(args, "state_dim", 8)),
                horizon=int(_arg(args, "chunk_size", _arg(args, "action_horizon", 12))),
                num_state_tokens=int(_arg(args, "num_state_tokens", 2)),
                num_layers=int(_arg(args, "act_num_layers", 3)),
                mlp_hidden_dim=int(_arg(args, "act_mlp_hidden_dim", 512)),
                state_hidden_dim=int(_arg(args, "act_state_hidden_dim", 256)),
                dropout=float(_arg(args, "act_dropout", 0.1)),
            ),
        )
    return TurboVLA(config)
