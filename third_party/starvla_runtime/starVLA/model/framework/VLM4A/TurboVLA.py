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
            "image_size": 224,
            "num_views": 3,
            "hidden_dim": 256,
            "patch_size": 16,
            "input_unit": "millimeter",
            "depth_scale": 1000.0,
            "min_depth_m": 0.05,
            "max_depth_m": 5.0,
            "use_log_depth": True,
            "invalid_threshold": 0.5,
            "freeze_depth_encoder": False,
            "dropout": 0.0,
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
                attention_implementation=fw.vision.get("attn_implementation"),
                compute_precision="bf16_autocast",
                position_init_std=float(fw.vision.position_init_std),
                position_scale_init=float(fw.vision.position_scale_init),
                dropout=float(fw.vision.dropout),
            ),
            depth=DepthEncoderConfig(
                enabled=bool(fw.depth.enabled),
                image_size=int(fw.depth.image_size),
                num_views=int(fw.depth.num_views),
                hidden_dim=int(fw.depth.hidden_dim),
                patch_size=int(fw.depth.patch_size),
                input_unit=str(fw.depth.input_unit),
                depth_scale=float(fw.depth.depth_scale),
                min_depth_m=float(fw.depth.min_depth_m),
                max_depth_m=float(fw.depth.max_depth_m),
                use_log_depth=bool(fw.depth.use_log_depth),
                invalid_threshold=float(fw.depth.invalid_threshold),
                frozen=bool(fw.depth.freeze_depth_encoder),
                dropout=float(fw.depth.dropout),
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

    @staticmethod
    def _as_depth_view_list(depths, num_views: int) -> list[np.ndarray]:
        # 深度保持数值数组，禁止转成 PIL/RGB，避免毫米精度被压缩到 8 bit。
        if isinstance(depths, (list, tuple)):
            views = list(depths)
        else:
            if isinstance(depths, torch.Tensor):
                array = depths.detach().cpu().numpy()
            else:
                array = np.asarray(depths)
            if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1):
                views = [array]
            else:
                views = [array[index] for index in range(array.shape[0])]
        if not views:
            raise ValueError("each depth example must contain at least one view")
        # RGB-D 模式要求三路相机严格一一对应，缺失时不能复制最后一路来掩盖数据错误。
        if len(views) != num_views:
            raise ValueError(f"expected exactly {num_views} depth views, got {len(views)}")

        normalized_views = []
        for view in views[:num_views]:
            array = view.detach().cpu().numpy() if isinstance(view, torch.Tensor) else np.asarray(view)
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 2:
                raise ValueError(f"each depth view must be [H,W], got {array.shape}")
            normalized_views.append(array)
        return normalized_views

    def _prepare_depth_batch(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        depth_views = [self._as_depth_view_list(example["depth"], self.num_views) for example in examples]
        depth = torch.as_tensor(np.stack(depth_views), device=device, dtype=torch.float32).unsqueeze(2)
        if depth.shape[-2:] != (self.image_size, self.image_size):
            flat_depth = depth.flatten(0, 1)
            # 最近邻 resize 保留物体边界和 0 值无效区域，不制造虚假的中间深度。
            flat_depth = F.interpolate(flat_depth, size=(self.image_size, self.image_size), mode="nearest")
            depth = flat_depth.view(len(examples), self.num_views, 1, self.image_size, self.image_size)
        return depth

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
        if self.model.config.depth.enabled:
            missing_depth = [index for index, example in enumerate(examples) if "depth" not in example]
            if missing_depth:
                raise ValueError(f"depth-enabled checkpoint requires depth in examples {missing_depth}")
            samples["depth"] = self._prepare_depth_batch(examples, device)
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
