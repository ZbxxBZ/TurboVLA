from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1


@dataclass
class DepthEncoderConfig:
    """真实深度编码器的配置。

    深度分支默认关闭，以便旧的 RGB-only 配置和 checkpoint 保持完全兼容。
    RoboTwin 的原始深度单位是毫米，数据管线会保留这个数值，编码器在这里统一换算成米。
    """

    enabled: bool = False
    image_size: int = 224
    num_views: int = 3
    hidden_dim: int = 256
    patch_size: int = 16
    input_unit: str = "millimeter"
    depth_scale: float = 1000.0
    min_depth_m: float = 0.05
    max_depth_m: float = 5.0
    use_log_depth: bool = True
    invalid_threshold: float = 0.5
    frozen: bool = False
    dropout: float = 0.0


@dataclass
class DepthFusionConfig:
    """RGB token 和深度 token 的门控融合配置。"""

    enabled: bool = False
    hidden_dim: int = 256
    nheads: int = 8
    dropout: float = 0.0
    gate_init: float = 0.0
    gate_parameterization: str = "tanh"
    gate_min: float = 0.0
    gate_max: float = 1.0
    mode: str = "global"


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    depth: DepthEncoderConfig = field(default_factory=DepthEncoderConfig)
    depth_fusion: DepthFusionConfig = field(default_factory=DepthFusionConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.depth.enabled != self.depth_fusion.enabled:
            raise ValueError("depth.enabled and depth_fusion.enabled must be enabled together")
        # 深度关闭时不约束视角数，保证所有旧的 LIBERO/RoboTwin 配置都能原样构建。
        if self.depth.enabled and self.depth.num_views != self.vision.num_views:
            raise ValueError("depth.num_views must match vision.num_views")
        if self.depth.enabled and self.depth.hidden_dim != self.interaction.hidden_dim:
            raise ValueError("depth.hidden_dim must match interaction.hidden_dim")
        if self.depth_fusion.enabled and self.depth_fusion.hidden_dim != self.interaction.hidden_dim:
            raise ValueError("depth_fusion.hidden_dim must match interaction.hidden_dim")
        if self.depth_fusion.mode not in {"global", "aligned"}:
            raise ValueError("depth_fusion.mode must be 'global' or 'aligned'")
        if self.depth_fusion.gate_parameterization not in {"tanh", "bounded_sigmoid"}:
            raise ValueError(
                "depth_fusion.gate_parameterization must be 'tanh' or 'bounded_sigmoid'"
            )
        if self.depth_fusion.gate_parameterization == "bounded_sigmoid":
            if not 0.0 <= self.depth_fusion.gate_min < self.depth_fusion.gate_max <= 1.0:
                raise ValueError("bounded depth gate requires 0 <= gate_min < gate_max <= 1")
            if not self.depth_fusion.gate_min < self.depth_fusion.gate_init < self.depth_fusion.gate_max:
                raise ValueError("bounded depth gate requires gate_min < gate_init < gate_max")
        if self.depth.patch_size < 1 or self.depth.image_size < 1:
            raise ValueError("depth.image_size and depth.patch_size must be positive")
        if self.depth.image_size % self.depth.patch_size != 0:
            raise ValueError("depth.image_size must be divisible by depth.patch_size")
        if self.depth.input_unit not in {"millimeter", "meter"}:
            raise ValueError("depth.input_unit must be 'millimeter' or 'meter'")
        if self.depth.depth_scale <= 0:
            raise ValueError("depth.depth_scale must be positive")
        if self.depth.min_depth_m <= 0 or self.depth.max_depth_m <= self.depth.min_depth_m:
            raise ValueError("depth min/max range is invalid")
        # 阈值为 0 时所有 patch 都会满足 invalid_fraction >= 0，因此直接禁止该退化配置。
        if not 0.0 < self.depth.invalid_threshold <= 1.0:
            raise ValueError("depth.invalid_threshold must be in (0, 1]")
        if self.depth_fusion.mode == "global" and self.depth_fusion.nheads < 1:
            raise ValueError("depth_fusion.nheads must be positive")
        if (
            self.depth_fusion.mode == "global"
            and self.depth_fusion.hidden_dim % self.depth_fusion.nheads != 0
        ):
            raise ValueError("depth_fusion.hidden_dim must be divisible by depth_fusion.nheads")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            depth=DepthEncoderConfig(**dict(data.get("depth", {}))),
            depth_fusion=DepthFusionConfig(**dict(data.get("depth_fusion", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
        )
