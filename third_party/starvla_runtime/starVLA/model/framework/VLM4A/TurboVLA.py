from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from turbovla.models import TurboVLAConfig, build_turbovla
from turbovla.models.configuration import (
    ActionHeadConfig,
    DepthEncoderConfig,
    DepthFusionConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)


@dataclass
class TurboVLADefaultConfig:
    name: str = "TurboVLA"
    text: dict = field(
        default_factory=lambda: {
            "bert_path": "/path/to/bert-base-uncased",
            "max_text_len": 256,
            "sub_sentence_present": True,
            "local_files_only": True,
            "freeze_text_encoder": True,
            "attn_implementation": "flash_attention_2",
        }
    )
    vision: dict = field(
        default_factory=lambda: {
            "model_path": "/path/to/dinov3",
            "image_size": 224,
            "num_views": 3,
            "local_files_only": True,
            "load_pretrained_weights": True,
            "freeze_vision_encoder": False,
            "attn_implementation": "flash_attention_2",
            "position_init_std": 0.01,
            "position_scale_init": 0.01,
            "dropout": 0.1,
        }
    )
    depth: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "backend": "dinov3",
            "image_size": 224,
            "num_views": 3,
            "hidden_dim": 256,
            "patch_size": 16,
            "head_camera_index": 0,
            "backbone_name": "dinov3_vits16plus",
            "backbone_repo_path": "",
            "backbone_weights_path": "",
            "backbone_num_layers": 12,
            "backbone_hidden_dim": 384,
            "head_weights_path": "",
            "projection_weights_path": "",
            "adapter_weights_path": "",
            "feature_dim": 160,
            "dpt_feature_dim": 256,
            "stage1_mode": "legacy_patch",
            "freeze_backbone": True,
            "freeze_depth_head": True,
            "freeze_depth_encoder": False,
            "dropout": 0.0,
            "vggt_repo_path": "",
            "vggt_weights_path": "",
            "vggt_image_size": 518,
            "vggt_patch_size": 14,
            "vggt_input_is_normalized": True,
            "min_depth_m": 0.001,
            "max_depth_m": 5.0,
            "min_valid_fraction": 0.5,
            "learn_metric_calibration": True,
            "metric_scale_init": 1.0,
            "metric_shift_init": 0.0,
        }
    )
    depth_fusion: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "mode": "global",
            "hidden_dim": 256,
            "nheads": 8,
            "dropout": 0.0,
            "gate_init": 0.0,
            "gate_parameterization": "tanh",
            "gate_min": 0.0,
            "gate_max": 1.0,
            "residual_scale_match": True,
            "active_view_index": 0,
            "zero_init_output": False,
        }
    )
    interaction: dict = field(
        default_factory=lambda: {
            "hidden_dim": 256,
            "nheads": 8,
            "dim_feedforward": 2048,
            "enhancer_inner_dim": 1024,
            "num_layers": 6,
            "text_dropout": 0.0,
            "fusion_dropout": 0.0,
            "fusion_droppath": 0.1,
            "padding_strategy": "zero_fill",
            "residual_style": "pre_norm",
            "attention_backend": "sdpa",
            "compute_precision": "bf16_autocast",
        }
    )
    initialization: dict = field(
        default_factory=lambda: {
            "pretrained_ckpt": "",
            "load_pretrained": True,
            "load_full_model": False,
            "load_bert": True,
            "load_text_projection": True,
            "load_interaction": True,
        }
    )
    action: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "state_dim": 14,
            "horizon": 50,
            "num_layers": 3,
            "num_state_tokens": 2,
            "state_hidden_dim": 256,
            "mlp_hidden_dim": 512,
            "dropout": 0.1,
            "loss_type": "l1",
        }
    )


@FRAMEWORK_REGISTRY.register("TurboVLA")
class TurboVLAFramework(baseframework):
    """StarVLA batch and checkpoint adapter around the shared TurboVLA model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        del kwargs
        super().__init__()
        self.config = merge_framework_config(TurboVLADefaultConfig, config)
        fw = self.config.framework
        self.model = build_turbovla(self._core_config(fw))
        self.image_processor = AutoImageProcessor.from_pretrained(
            fw.vision.model_path,
            local_files_only=fw.vision.local_files_only,
        )
        self.image_size = int(fw.vision.image_size)
        self.num_views = int(fw.vision.num_views)
        if hasattr(self.image_processor, "size"):
            self.image_processor.size = {"height": self.image_size, "width": self.image_size}
        self.action_horizon = int(fw.action.horizon)
        self.loss_type = str(fw.action.loss_type).lower()
        if fw.initialization.load_pretrained:
            self._load_initialization(fw.initialization)

    @staticmethod
    def _core_config(fw) -> TurboVLAConfig:
        return TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=fw.text.bert_path,
                max_length=int(fw.text.max_text_len),
                padding_length=None,
                sub_sentence_present=bool(fw.text.sub_sentence_present),
                frozen=bool(fw.text.freeze_text_encoder),
                force_eval_when_frozen=True,
                zero_padded_tokens=True,
                local_files_only=bool(fw.text.local_files_only),
                attention_implementation=fw.text.get("attn_implementation"),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=fw.vision.model_path,
                image_size=int(fw.vision.image_size),
                num_views=int(fw.vision.num_views),
                position_embedding="learned_patch",
                encode_views_separately=False,
                frozen=bool(fw.vision.freeze_vision_encoder),
                local_files_only=bool(fw.vision.local_files_only),
                load_pretrained_weights=bool(fw.vision.load_pretrained_weights),
                attention_implementation=fw.vision.get("attn_implementation"),
                compute_precision="bf16_autocast",
                position_init_std=float(fw.vision.position_init_std),
                position_scale_init=float(fw.vision.position_scale_init),
                dropout=float(fw.vision.dropout),
            ),
            depth=DepthEncoderConfig(
                enabled=bool(fw.depth.enabled),
                backend=str(fw.depth.backend),
                image_size=int(fw.depth.image_size),
                num_views=int(fw.depth.num_views),
                hidden_dim=int(fw.depth.hidden_dim),
                patch_size=int(fw.depth.patch_size),
                head_camera_index=int(fw.depth.head_camera_index),
                backbone_name=str(fw.depth.backbone_name),
                backbone_repo_path=str(fw.depth.backbone_repo_path),
                backbone_weights_path=str(fw.depth.backbone_weights_path),
                backbone_num_layers=int(fw.depth.backbone_num_layers),
                backbone_hidden_dim=int(fw.depth.backbone_hidden_dim),
                head_weights_path=str(fw.depth.head_weights_path),
                projection_weights_path=str(fw.depth.projection_weights_path),
                adapter_weights_path=str(fw.depth.adapter_weights_path),
                feature_dim=int(fw.depth.feature_dim),
                dpt_feature_dim=int(fw.depth.get("dpt_feature_dim", 256)),
                stage1_mode=str(fw.depth.get("stage1_mode", "legacy_patch")),
                freeze_backbone=bool(fw.depth.freeze_backbone),
                freeze_depth_head=bool(fw.depth.freeze_depth_head),
                frozen=bool(fw.depth.freeze_depth_encoder),
                dropout=float(fw.depth.dropout),
                vggt_repo_path=str(fw.depth.vggt_repo_path),
                vggt_weights_path=str(fw.depth.vggt_weights_path),
                vggt_image_size=int(fw.depth.vggt_image_size),
                vggt_patch_size=int(fw.depth.vggt_patch_size),
                vggt_input_is_normalized=bool(fw.depth.vggt_input_is_normalized),
                min_depth_m=float(fw.depth.min_depth_m),
                max_depth_m=float(fw.depth.max_depth_m),
                min_valid_fraction=float(fw.depth.min_valid_fraction),
                learn_metric_calibration=bool(fw.depth.learn_metric_calibration),
                metric_scale_init=float(fw.depth.metric_scale_init),
                metric_shift_init=float(fw.depth.metric_shift_init),
            ),
            depth_fusion=DepthFusionConfig(
                enabled=bool(fw.depth_fusion.enabled),
                mode=str(fw.depth_fusion.mode),
                hidden_dim=int(fw.depth_fusion.hidden_dim),
                nheads=int(fw.depth_fusion.nheads),
                dropout=float(fw.depth_fusion.dropout),
                gate_init=float(fw.depth_fusion.gate_init),
                gate_parameterization=str(fw.depth_fusion.gate_parameterization),
                gate_min=float(fw.depth_fusion.gate_min),
                gate_max=float(fw.depth_fusion.gate_max),
                residual_scale_match=bool(fw.depth_fusion.get("residual_scale_match", True)),
                active_view_index=int(fw.depth_fusion.get("active_view_index", fw.depth.head_camera_index)),
                zero_init_output=bool(fw.depth_fusion.zero_init_output),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(fw.interaction.hidden_dim),
                nheads=int(fw.interaction.nheads),
                num_layers=int(fw.interaction.num_layers),
                dim_feedforward=int(fw.interaction.dim_feedforward),
                enhancer_inner_dim=int(fw.interaction.enhancer_inner_dim),
                text_dropout=float(fw.interaction.text_dropout),
                fusion_dropout=float(fw.interaction.fusion_dropout),
                fusion_droppath=float(fw.interaction.fusion_droppath),
                padding_strategy=str(fw.interaction.padding_strategy),
                residual_style=str(fw.interaction.residual_style),
                attention_backend=str(fw.interaction.attention_backend),
                compute_precision=str(fw.interaction.compute_precision),
            ),
            action=ActionHeadConfig(
                action_dim=int(fw.action.action_dim),
                state_dim=int(fw.action.state_dim),
                horizon=int(fw.action.horizon),
                num_state_tokens=int(fw.action.num_state_tokens),
                num_layers=int(fw.action.num_layers),
                mlp_hidden_dim=int(fw.action.mlp_hidden_dim),
                state_hidden_dim=int(fw.action.state_hidden_dim),
                dropout=float(fw.action.dropout),
            ),
        )

    @staticmethod
    def _checkpoint_state(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    return checkpoint[key]
        return checkpoint

    @staticmethod
    def _load_checkpoint_file(path: str):
        checkpoint_path = Path(path)
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(checkpoint_path), device="cpu")
        return torch.load(checkpoint_path, map_location="cpu")

    @staticmethod
    def _legacy_rgb_checkpoint_key(key: str) -> str | None:
        if key.startswith("dinov3.model."):
            return "vision_encoder.backbone." + key[len("dinov3.model.") :]
        if key.startswith("vision_proj."):
            suffix = key[len("vision_proj.") :]
            suffix = suffix.replace("norm_in.", "input_norm.").replace("norm_out.", "output_norm.")
            return "vision_projection." + suffix
        if key == "vision_pos_embed":
            return "patch_position_embedding"
        if key == "vision_pos_scale":
            return "patch_position_scale"
        if key == "view_embed":
            return "view_embedding"
        if key.startswith("text_encoder.bert."):
            return key
        if key.startswith("text_encoder.text_proj."):
            return "text_encoder.text_projection." + key[len("text_encoder.text_proj.") :]
        if key.startswith("feature_enhancer."):
            return "vision_language_interaction." + key[len("feature_enhancer.") :]
        if key.startswith("action_model.state_proj."):
            suffix = key[len("action_model.state_proj.") :]
            if suffix == "pos":
                suffix = "position"
            elif suffix.startswith("out_norm."):
                suffix = "output_norm." + suffix[len("out_norm.") :]
            return "action_head.state_projection." + suffix
        if key.startswith("action_model.action_policy.action_queries."):
            return "action_head.decoder.action_queries." + key[len("action_model.action_policy.action_queries.") :]
        if key.startswith("action_model.action_policy.decoder."):
            return "action_head.decoder.decoder." + key[len("action_model.action_policy.decoder.") :]
        if key.startswith("action_model.action_policy.action_head."):
            return "action_head.decoder.action_projection." + key[len("action_model.action_policy.action_head.") :]
        return None

    def _load_full_initialization(self, path: str) -> None:
        source = self._checkpoint_state(self._load_checkpoint_file(path))
        source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
        target = self.model.state_dict()
        loaded = {}
        unexpected = []
        shape_mismatches = []

        for source_key, value in source.items():
            target_key = source_key
            if target_key.startswith("model.") and target_key[len("model.") :] in target:
                target_key = target_key[len("model.") :]
            elif target_key not in target:
                target_key = self._legacy_rgb_checkpoint_key(source_key)
            if target_key is None or target_key not in target:
                unexpected.append(source_key)
                continue
            if tuple(target[target_key].shape) != tuple(value.shape):
                shape_mismatches.append(
                    (source_key, tuple(value.shape), target_key, tuple(target[target_key].shape))
                )
                continue
            loaded[target_key] = value

        allowed_missing_prefixes = ("depth_encoder.", "depth_fusion.")
        missing = sorted(
            key
            for key in target
            if key not in loaded and not key.startswith(allowed_missing_prefixes)
        )
        if unexpected or shape_mismatches or missing:
            raise RuntimeError(
                "full checkpoint is incompatible: "
                f"unexpected={unexpected[:10]}, shape_mismatches={shape_mismatches[:10]}, missing={missing[:10]}"
            )

        target.update(loaded)
        self.model.load_state_dict(target, strict=True)
        new_depth_tensors = sorted(key for key in target if key not in loaded)
        print(
            f"[TurboVLA] loaded {len(loaded)} full RGB tensors from {path}; "
            f"initialized {len(new_depth_tensors)} new depth tensors",
            flush=True,
        )

    def _load_initialization(self, init_cfg) -> None:
        path = str(init_cfg.pretrained_ckpt)
        if not path:
            raise ValueError("framework.initialization.pretrained_ckpt is required")
        if bool(init_cfg.get("load_full_model", False)):
            self._load_full_initialization(path)
            return
        source = self._checkpoint_state(self._load_checkpoint_file(path))
        source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
        mappings = []
        if init_cfg.load_bert:
            mappings.append(("bert.", "text_encoder.bert."))
        if init_cfg.load_text_projection:
            mappings.append(("feat_map.", "text_encoder.text_projection."))
        if init_cfg.load_interaction:
            mappings.extend(
                [
                    ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
                    ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
                ]
            )
        target = self.model.state_dict()
        loaded = {}
        for source_key, value in source.items():
            for source_prefix, target_prefix in mappings:
                if not source_key.startswith(source_prefix):
                    continue
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key in target and tuple(target[target_key].shape) == tuple(value.shape):
                    loaded[target_key] = value
                break
        if not loaded:
            raise RuntimeError(f"no compatible initialization tensors found in {path}")
        target.update(loaded)
        self.model.load_state_dict(target, strict=True)
        print(f"[TurboVLA] loaded {len(loaded)} initialization tensors from {path}", flush=True)

    @staticmethod
    def _as_view_list(
        images,
        num_views: int,
        *,
        require_exact: bool = False,
    ) -> list[Image.Image]:
        images = to_pil_preserve(images)
        views = [images] if isinstance(images, Image.Image) else list(images)
        if not views:
            raise ValueError("each example must contain at least one image")
        if require_exact and len(views) != num_views:
            raise ValueError(f"expected exactly {num_views} RGB views, got {len(views)}")
        if len(views) < num_views:
            views.extend([views[-1]] * (num_views - len(views)))
        return views[:num_views]

    def _model_inputs(self, examples: List[dict]):
        if not isinstance(examples, list):
            examples = [examples]
        device = next(self.parameters()).device
        views = [
            self._as_view_list(
                example["image"],
                self.num_views,
                require_exact=self.model.config.depth.enabled,
            )
            for example in examples
        ]
        flat_images = [image for example_views in views for image in example_views]
        pixel_values = self.image_processor(images=flat_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.view(len(examples), self.num_views, *pixel_values.shape[1:]).to(device)
        instructions = [str(example["lang"]) for example in examples]
        states = torch.as_tensor(
            np.asarray([example["state"] for example in examples]),
            device=device,
            dtype=torch.float32,
        )
        samples = {"dinov3": pixel_values}
        return instructions, samples, states

    def forward(self, examples: List[dict] = None, **kwargs):
        del kwargs
        instructions, samples, states = self._model_inputs(examples)
        predicted = self.model(instructions, samples, states)
        targets = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=predicted.device,
            dtype=predicted.dtype,
        )[:, -self.action_horizon :]
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, targets)
        elif self.loss_type in {"smooth_l1", "huber"}:
            loss = F.smooth_l1_loss(predicted, targets)
        else:
            loss = F.l1_loss(predicted, targets)
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        profile_latency = bool(kwargs.pop("profile_latency", False))
        start = time.perf_counter()
        instructions, samples, states = self._model_inputs(examples)
        predicted = self.model(instructions, samples, states)
        output = {"normalized_actions": predicted.detach().float().cpu().numpy()}
        if profile_latency:
            if predicted.device.type == "cuda":
                torch.cuda.synchronize(predicted.device)
            output["latency_ms"] = {"predict_action_total": (time.perf_counter() - start) * 1000.0}
        return output

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def text_encoder(self):
        return self.model.text_encoder

    @property
    def vision_encoder(self):
        return self.model.vision_encoder

    @property
    def vision_language_interaction(self):
        return self.model.vision_language_interaction

    @property
    def vision_projection(self):
        return self.model.vision_projection

    @property
    def depth_encoder(self):
        # 该属性供训练器按模块配置独立学习率。
        return self.model.depth_encoder

    @property
    def depth_fusion(self):
        return self.model.depth_fusion

    @property
    def action_head(self):
        return self.model.action_head
