"""Two-step GPU smoke test for post-language global depth fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
STARVLA_ROOT = REPO_ROOT / "third_party" / "starvla_runtime"
for import_root in (REPO_ROOT, STARVLA_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import (  # noqa: E402
    TrainerUtils,
    build_param_lr_groups,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "experiments" / "robotwin" / "configs" / "clean50_depth.yaml"),
    )
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--dinov3-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def _grad_l1(module: torch.nn.Module) -> float:
    return sum(
        parameter.grad.detach().abs().float().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _parameter_grad_l1(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return parameter.grad.detach().abs().float().sum().item()


def _parameter_hashes(model: torch.nn.Module, *, trainable: bool) -> dict[str, str]:
    hashes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != trainable:
            continue
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    os.environ["BERT_MODEL_PATH"] = str(Path(args.bert_path).resolve())
    os.environ["DINOV3_MODEL_PATH"] = str(Path(args.dinov3_path).resolve())
    os.environ["TURBOVLA_INIT_CKPT"] = str(Path(args.checkpoint).resolve())
    os.environ.setdefault("ROBOTWIN_DATA_ROOT", str(REPO_ROOT / "unused-smoke-data"))

    cfg = OmegaConf.load(args.config)
    initialization_start = time.perf_counter()
    model = build_framework(cfg)
    initialization_seconds = time.perf_counter() - initialization_start

    # The production trainer builds the optimizer before applying its final freeze policy.
    param_groups = build_param_lr_groups(model, cfg)
    group_names = [group["name"] for group in param_groups]
    expected_groups = [
        "depth_encoder",
        "depth_fusion",
        "vision_language_interaction",
        "action_head",
        "base",
    ]
    if group_names != expected_groups:
        raise AssertionError(f"unexpected optimizer groups: {group_names}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )
    TrainerUtils.freeze_backbones(
        model,
        freeze_modules=cfg.trainer.freeze_modules,
        train_modules=cfg.trainer.train_modules,
    )

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_prefixes = (
        "model.vision_encoder.backbone.",
        "model.text_encoder.bert.",
    )
    unexpected_trainable = [name for name in trainable_names if name.startswith(frozen_prefixes)]
    if unexpected_trainable:
        raise AssertionError(f"frozen backbone parameters are trainable: {unexpected_trainable}")
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    unexpected_frozen = [name for name in frozen_names if not name.startswith(frozen_prefixes)]
    if unexpected_frozen:
        raise AssertionError(f"non-backbone parameters are frozen: {unexpected_frozen}")
    required_trainable_prefixes = (
        "model.text_encoder.text_projection.",
        "model.vision_projection.",
        "model.depth_encoder.",
        "model.depth_fusion.",
        "model.vision_language_interaction.",
        "model.action_head.state_projection.",
        "model.action_head.decoder.",
    )
    for prefix in required_trainable_prefixes:
        if not any(name.startswith(prefix) for name in trainable_names):
            raise AssertionError(f"expected trainable module is missing: {prefix}")
    if not any(name.startswith("model.depth_encoder.") for name in trainable_names):
        raise AssertionError("MetricDepthEncoder has no trainable parameters")
    if not any(name.startswith("model.depth_fusion.") for name in trainable_names):
        raise AssertionError("GatedDepthCrossAttention has no trainable parameters")

    device = torch.device(args.device)
    model.to(device)
    model.train()
    frozen_before = _parameter_hashes(model, trainable=False)
    trainable_before = _parameter_hashes(model, trainable=True)

    generator = torch.Generator(device=device).manual_seed(42)
    pixel_values = torch.randn(1, 3, 3, 224, 224, generator=generator, device=device)
    depth_values = torch.randint(
        200,
        1801,
        (1, 3, 1, 224, 224),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).float()
    depth_values[:, 0, :, :16, :16] = 0.0
    state = torch.zeros(1, 14, device=device)
    target = torch.zeros(1, 50, 14, device=device)
    instructions = ["pick up the object"]

    step_metrics = []
    for step in range(1, args.steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        predicted = model.model(
            instructions,
            {"dinov3": pixel_values, "depth": depth_values},
            state,
        )
        loss = F.l1_loss(predicted.float(), target)
        if not torch.isfinite(loss):
            raise AssertionError(f"step {step} produced a non-finite loss: {loss.item()}")
        loss.backward()

        gate_grad = _parameter_grad_l1(model.depth_fusion.depth_gate)
        encoder_grad = _grad_l1(model.depth_encoder)
        attention_grad = _grad_l1(model.depth_fusion.cross_attention)
        if gate_grad <= 0.0:
            raise AssertionError(f"step {step} did not produce a depth gate gradient")
        if encoder_grad <= 0.0 or attention_grad <= 0.0:
            raise AssertionError(
                "nonzero bounded gate must pass gradients into the depth branch from step 1: "
                f"step={step}, encoder={encoder_grad}, attention={attention_grad}"
            )

        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_metrics.append(
            {
                "step": step,
                "seconds": time.perf_counter() - step_start,
                "loss": loss.detach().float().item(),
                "gate_grad_l1": gate_grad,
                "encoder_grad_l1": encoder_grad,
                "cross_attention_grad_l1": attention_grad,
                "effective_gate_abs_mean": (
                    model.depth_fusion.effective_gate().detach().float().abs().mean().item()
                ),
            }
        )

    frozen_after = _parameter_hashes(model, trainable=False)
    if frozen_after != frozen_before:
        changed = sorted(name for name in frozen_before if frozen_before[name] != frozen_after.get(name))
        raise AssertionError(f"frozen parameters changed: {changed[:20]}")
    trainable_after = _parameter_hashes(model, trainable=True)
    changed_trainable = sorted(
        name for name in trainable_before if trainable_before[name] != trainable_after.get(name)
    )
    if not changed_trainable:
        raise AssertionError("optimizer did not change any trainable parameter")

    result = {
        "status": "passed",
        "device": str(device),
        "optimizer_groups": group_names,
        "trainable_tensors": len(trainable_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initialization_seconds": initialization_seconds,
        "frozen_tensors_verified": len(frozen_before),
        "changed_trainable_tensors": changed_trainable,
        "steps": step_metrics,
    }
    if device.type == "cuda":
        result["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
