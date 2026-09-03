from __future__ import annotations

import sys
import time
from contextlib import nullcontext
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
    InteractionConfig,
    ThreeDMixConfig,
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
    three_dmix: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "vggt_dim": 2048,
            "semantic_pool": "vl",
            "output_scale_init": 0.0,
            "feature_key": "vggt",
            # Online VGGT is opt-in. The extractor is intentionally kept
            # outside the nn.Module tree so its frozen weights are not saved
            # into TurboVLA checkpoints or passed to the optimizer.
            "online": False,
            "vggt_model_path": "",
            "vggt_code_path": "/mnt/vggt",
            "vggt_input_size": 518,
        }
    )
    initialization: dict = field(
        default_factory=lambda: {
            "pretrained_ckpt": "",
            "load_pretrained": True,
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


class _OnlineVGGT:
    """Lazy, frozen VGGT feature extractor for online 3D-MIX training."""

    def __init__(self, model_path: str, code_path: str, input_size: int = 518) -> None:
        self.model_path = str(model_path)
        self.code_path = str(code_path)
        self.input_size = int(input_size)
        self.model = None
        self.device = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError(
                "Online VGGT is enabled but framework.three_dmix.vggt_model_path is empty"
            )
        if self.code_path and self.code_path not in sys.path:
            sys.path.insert(0, self.code_path)
        from vggt.models.vggt import VGGT

        model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        try:
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:  # Older torch versions do not expose weights_only.
            state = torch.load(self.model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"VGGT checkpoint is missing {len(missing)} tensors; refusing online extraction"
            )
        print(
            f"[TurboVLA] online VGGT loaded: {self.model_path} "
            f"(unexpected={len(unexpected)})",
            flush=True,
        )
        model.eval()
        self.model = model

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"online VGGT images must be [B,V,3,H,W], got {tuple(images.shape)}")
        self._ensure_model()
        if self.device != images.device:
            self.model.to(images.device)
            self.device = images.device
        precision_context = nullcontext()
        if images.device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            aggregated, patch_start = self.model.aggregator(images)
        # Keep every native patch token and flatten the view dimension:
        # [B,V,N_patch,2048] -> [B,V*N_patch,2048].
        return aggregated[-1][:, :, patch_start:, :].float().flatten(1, 2)


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
        self._online_vggt = None
        if bool(fw.three_dmix.get("online", False)):
            self._online_vggt = _OnlineVGGT(
                model_path=str(fw.three_dmix.get("vggt_model_path", "")),
                code_path=str(fw.three_dmix.get("vggt_code_path", "/mnt/vggt")),
                input_size=int(fw.three_dmix.get("vggt_input_size", 518)),
            )
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
            three_dmix=ThreeDMixConfig(
                enabled=bool(fw.three_dmix.enabled),
                vggt_dim=int(fw.three_dmix.vggt_dim),
                semantic_pool=str(fw.three_dmix.semantic_pool),
                output_scale_init=float(fw.three_dmix.output_scale_init),
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

    def _load_initialization(self, init_cfg) -> None:
        path = str(init_cfg.pretrained_ckpt)
        if not path:
            raise ValueError("framework.initialization.pretrained_ckpt is required")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            source = load_file(path, device="cpu")
        else:
            source = self._checkpoint_state(torch.load(path, map_location="cpu"))
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
            # Continue-training checkpoints produced by this TurboVLA model
            # already use the target names. Also accept a common `model.`
            # wrapper prefix before falling back to legacy StarVLA mappings.
            direct_candidates = [source_key]
            if source_key.startswith("model."):
                direct_candidates.append(source_key[len("model.") :])
            direct_key = next(
                (
                    candidate
                    for candidate in direct_candidates
                    if candidate in target and tuple(target[candidate].shape) == tuple(value.shape)
                ),
                None,
            )
            if direct_key is not None:
                loaded[direct_key] = value
                continue
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
    def _as_view_list(images, num_views: int) -> list[Image.Image]:
        images = to_pil_preserve(images)
        views = [images] if isinstance(images, Image.Image) else list(images)
        if not views:
            raise ValueError("each example must contain at least one image")
        if len(views) < num_views:
            views.extend([views[-1]] * (num_views - len(views)))
        return views[:num_views]

    @staticmethod
    def _vggt_image_tensor(image: Image.Image, input_size: int) -> torch.Tensor:
        """Convert one RGB PIL view to VGGT's [3,H,W] float input."""
        image = image.convert("RGB").resize((input_size, input_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _model_inputs(self, examples: List[dict]):
        if not isinstance(examples, list):
            examples = [examples]
        device = next(self.parameters()).device
        views = [self._as_view_list(example["image"], self.num_views) for example in examples]
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
        if bool(self.config.framework.three_dmix.enabled):
            if self._online_vggt is not None:
                vggt_images = torch.stack(
                    [
                        torch.stack(
                            [
                                self._vggt_image_tensor(view, self._online_vggt.input_size)
                                for view in self._as_view_list(example["image"], self.num_views)
                            ],
                            dim=0,
                        )
                        for example in examples
                    ],
                    dim=0,
                ).to(device)
                samples["vggt"] = self._online_vggt(vggt_images)
            else:
                feature_key = str(self.config.framework.three_dmix.get("feature_key", "vggt"))
                features = [
                    self._load_vggt_feature(example.get(feature_key), index)
                    for index, example in enumerate(examples)
                ]
                if any(feature is None for feature in features):
                    raise ValueError(
                        "ThreeDMix is enabled and online VGGT is disabled, but RoboTwin examples do not provide "
                        f"{feature_key!r}. Add cached VGGT tensors or enable framework.three_dmix.online."
                    )
                try:
                    samples["vggt"] = torch.stack(
                        [feature for feature in features if feature is not None], dim=0
                    ).to(device)
                except RuntimeError as exc:
                    shapes = [tuple(feature.shape) for feature in features if feature is not None]
                    raise ValueError(f"VGGT feature shapes must match across a batch, got {shapes}") from exc
        return instructions, samples, states

    @staticmethod
    def _load_vggt_feature(value, index: int) -> torch.Tensor | None:
        """Normalize an inline tensor/array or an offline feature path."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            # Preserve device for online features; cached tensors loaded from
            # disk are CPU tensors and follow the same path below.
            feature = value.detach()
        elif isinstance(value, np.ndarray):
            feature = torch.from_numpy(value)
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"VGGT feature for example {index} not found: {path}")
            if path.suffix.lower() in {".pt", ".pth", ".bin"}:
                try:
                    loaded = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, dict):
                    for key in ("features", "vggt", "patch_tokens"):
                        if key in loaded:
                            loaded = loaded[key]
                            break
                if not isinstance(loaded, torch.Tensor):
                    loaded = torch.as_tensor(loaded)
                feature = loaded
            elif path.suffix.lower() in {".npy", ".npz"}:
                loaded = np.load(path)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    key = "features" if "features" in loaded.files else loaded.files[0]
                    loaded = loaded[key]
                feature = torch.from_numpy(np.asarray(loaded))
            else:
                raise ValueError(f"Unsupported VGGT feature file type: {path.suffix}")
        else:
            feature = torch.as_tensor(value)
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        if feature.ndim == 3:
            feature = feature.flatten(0, 1)
        if feature.ndim != 2:
            raise ValueError(
                f"VGGT feature for example {index} must be [N,C] or [V,N,C], got {tuple(feature.shape)}"
            )
        return feature.float()

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
    def three_dmix(self):
        return self.model.three_dmix

    @property
    def action_head(self):
        return self.model.action_head
