#!/usr/bin/env python3
"""Stage 1: train either native VGGT DPT depth or the legacy token adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from scripts.robotwin.train_depth_dinov3 import (
    RoboTwinHeadRGBD,
    episode_paths,
    split_episodes,
    task_name,
)
from turbovla.models.configuration import DepthEncoderConfig
from turbovla.models.vggt_depth_encoder import VGGTDepthEncoder


class LinearDepthProbe(nn.Module):
    """Official DINOv3 linear-probe computation without importing DINOv3."""

    def __init__(self, hidden_dim: int, patch_grid: int, num_bins: int, min_depth: float, max_depth: float) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.hidden_dim = hidden_dim
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.head = nn.Conv2d(hidden_dim, num_bins, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = self.patch_grid**2
        if tokens.ndim != 3 or tokens.shape[1:] != (expected, self.hidden_dim):
            raise ValueError(f"expected [B,{expected},{self.hidden_dim}] tokens, got {tuple(tokens.shape)}")
        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.hidden_dim, self.patch_grid, self.patch_grid
        )
        logits = self.head(feature_map)
        weights = logits.relu() + 0.1
        weights = weights / weights.sum(dim=1, keepdim=True)
        bins = torch.linspace(self.min_depth, self.max_depth, logits.shape[1], device=logits.device)
        return torch.einsum("bkhw,k->bhw", weights, bins).unsqueeze(1)


class MaskedSigLoss(nn.Module):
    """Scale-invariant depth loss that is finite for an empty valid batch."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target) & (target > 0)
        if not mask.any():
            return prediction.sum() * 0.0
        difference = (prediction[mask] + 1e-3).log() - (target[mask] + 1e-3).log()
        return (difference.var(unbiased=False) + 0.15 * difference.mean().square()).sqrt()


class MaskedDenseDepthLoss(nn.Module):
    """Pixel-level metric/log depth loss used by the native VGGT DPT path."""

    def __init__(self, gradient_weight: float = 0.05) -> None:
        super().__init__()
        self.gradient_weight = float(gradient_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(target) & (target > 0) & torch.isfinite(prediction)
        if not valid.any():
            return prediction.sum() * 0.0
        pred = prediction.clamp_min(1e-3)
        gt = target.clamp_min(1e-3)
        log_error = pred.log() - gt.log()
        masked = log_error[valid]
        sigloss = (masked.var(unbiased=False) + 0.15 * masked.mean().square()).sqrt()

        # Match the public DINOv3/VGGT dense-depth heads by preserving local
        # depth boundaries in addition to the per-pixel metric value.
        gradient_terms: list[torch.Tensor] = []
        for dim in (-2, -1):
            error_delta = log_error.diff(dim=dim).abs()
            valid_delta = valid.narrow(dim, 1, valid.shape[dim] - 1) & valid.narrow(
                dim, 0, valid.shape[dim] - 1
            )
            if valid_delta.any():
                gradient_terms.append(error_delta[valid_delta].mean())
        gradient = torch.stack(gradient_terms).mean() if gradient_terms else sigloss.detach() * 0.0
        return sigloss + self.gradient_weight * gradient


def patch_depth_targets(
    depth: torch.Tensor,
    *,
    patch_grid: int,
    min_depth: float,
    max_depth: float,
    min_valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    values = torch.where(valid, depth, torch.zeros_like(depth))
    pooled_values = F.adaptive_avg_pool2d(values, (patch_grid, patch_grid))
    valid_fraction = F.adaptive_avg_pool2d(valid.float(), (patch_grid, patch_grid))
    target = pooled_values / valid_fraction.clamp_min(1e-6)
    valid_patches = valid_fraction >= min_valid_fraction
    return torch.where(valid_patches, target, torch.zeros_like(target)), valid_patches


def config_from_args(args: argparse.Namespace) -> DepthEncoderConfig:
    return DepthEncoderConfig(
        enabled=True,
        backend="vggt",
        stage1_mode=args.stage1_mode,
        image_size=args.image_size,
        num_views=3,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
        head_camera_index=0,
        feature_dim=args.feature_dim,
        dpt_feature_dim=args.dpt_feature_dim,
        freeze_backbone=True,
        freeze_depth_head=args.stage1_mode != "dpt_dense",
        frozen=False,
        dropout=0.0,
        vggt_repo_path=str(args.vggt_repo),
        vggt_weights_path=str(args.vggt_weights),
        vggt_image_size=args.vggt_image_size,
        vggt_patch_size=args.vggt_patch_size,
        vggt_input_is_normalized=True,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_valid_fraction=args.min_valid_fraction,
        learn_metric_calibration=True,
        metric_scale_init=args.metric_scale_init,
        metric_shift_init=args.metric_shift_init,
    )


def trainable_parameters(model: VGGTDepthEncoder, probe: nn.Module) -> list[nn.Parameter]:
    parameters = list(
        chain(
            model.depth_patch_embedding.parameters(),
            model.token_projection.parameters(),
            model.token_norm.parameters(),
            [model.metric_scale_raw, model.metric_shift],
        )
    )
    if model.stage1_mode == "dpt_dense" and not model.config.freeze_depth_head:
        parameters.extend(parameter for parameter in model.vggt.depth_head.parameters() if parameter.requires_grad)
    else:
        parameters.extend(probe.parameters())
    return parameters


@torch.inference_mode()
def validate(
    model: VGGTDepthEncoder,
    probe: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    probe.eval()
    loss_sum = 0.0
    abs_sum = 0.0
    rel_sum = 0.0
    sq_sum = 0.0
    count = 0
    batches = 0
    for rgb, depth in loader:
        rgb, depth = rgb.to(device), depth.to(device)
        if model.stage1_mode == "dpt_dense":
            target = depth
            valid_mask = (
                torch.isfinite(target)
                & (target >= args.min_depth_m)
                & (target <= args.max_depth_m)
            )
        else:
            target, valid_mask = patch_depth_targets(
                depth,
                patch_grid=model.patch_grid,
                min_depth=args.min_depth_m,
                max_depth=args.max_depth_m,
                min_valid_fraction=args.min_valid_fraction,
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = (
                model.predict_head_depth(rgb)
                if model.stage1_mode == "dpt_dense"
                else probe(model.encode_head_tokens(rgb))
            )
        prediction, target = prediction.float(), target.float()
        loss_sum += criterion(prediction.float(), target.float()).item()
        errors = (prediction - target).abs()[valid_mask]
        targets = target[valid_mask]
        count += targets.numel()
        abs_sum += errors.sum().item()
        rel_sum += (errors / targets.clamp_min(1e-3)).sum().item()
        sq_sum += errors.square().sum().item()
        batches += 1
        if args.max_validation_batches and batches >= args.max_validation_batches:
            break
    count = max(count, 1)
    return {
        "loss": loss_sum / max(batches, 1),
        "mae_m": abs_sum / count,
        "abs_rel": rel_sum / count,
        "rmse_m": math.sqrt(sq_sum / count),
        "valid_elements": count,
        "supervision": "pixels" if model.stage1_mode == "dpt_dense" else "patches",
    }


def save_checkpoint(
    path: Path,
    model: VGGTDepthEncoder,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "vggt_depth_token_adapter",
            "epoch": epoch,
            "validation": validation,
            "depth_encoder_config": asdict(model.config),
            "stage1_mode": args.stage1_mode,
            "vggt_adapter": model.adapter_state_dict(),
            **(
                {"temporary_linear_probe": probe.state_dict()}
                if args.stage1_mode == "legacy_patch"
                else {}
            ),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = episode_paths(args.dataset_root)
    train_paths, validation_paths = split_episodes(paths, args.validation_fraction, args.seed)
    train_dataset = RoboTwinHeadRGBD(
        train_paths,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        augment=True,
    )
    validation_dataset = RoboTwinHeadRGBD(
        validation_paths,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        augment=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = VGGTDepthEncoder(config_from_args(args)).to(device)
    probe: nn.Module = nn.Identity()
    if args.stage1_mode == "legacy_patch":
        probe = LinearDepthProbe(
            args.hidden_dim,
            model.patch_grid,
            args.num_depth_bins,
            args.min_depth_m,
            args.max_depth_m,
        ).to(device)
    criterion: nn.Module = (
        MaskedDenseDepthLoss(args.dense_gradient_weight).to(device)
        if args.stage1_mode == "dpt_dense"
        else MaskedSigLoss().to(device)
    )
    parameters = trainable_parameters(model, probe)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "train_episodes": len(train_paths),
                "validation_episodes": len(validation_paths),
                "train_frames": len(train_dataset),
                "validation_frames": len(validation_dataset),
                "tasks": len({task_name(path) for path in paths}),
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "stage1_mode": args.stage1_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        probe.train()
        started = time.time()
        running = 0.0
        for step, (rgb, depth) in enumerate(train_loader, start=1):
            rgb, depth = rgb.to(device), depth.to(device)
            if model.stage1_mode == "dpt_dense":
                target = depth
            else:
                target, _ = patch_depth_targets(
                    depth,
                    patch_grid=model.patch_grid,
                    min_depth=args.min_depth_m,
                    max_depth=args.max_depth_m,
                    min_valid_fraction=args.min_valid_fraction,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = (
                    model.predict_head_depth(rgb)
                    if model.stage1_mode == "dpt_dense"
                    else probe(model.encode_head_tokens(rgb))
                )
                loss = criterion(prediction.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running / args.log_every:.6f}", flush=True)
                running = 0.0

        validation = validate(model, probe, validation_loader, criterion, args, device)
        validation["elapsed_seconds"] = time.time() - started
        print(f"epoch={epoch} validation={json.dumps(validation, sort_keys=True)}", flush=True)
        save_checkpoint(args.output_dir / "last.pt", model, probe, optimizer, epoch, validation, args)
        save_checkpoint(args.output_dir / f"epoch_{epoch:02d}.pt", model, probe, optimizer, epoch, validation, args)
        if validation["abs_rel"] < best:
            best = validation["abs_rel"]
            save_checkpoint(args.output_dir / "best.pt", model, probe, optimizer, epoch, validation, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=160)
    parser.add_argument("--dpt-feature-dim", type=int, default=256)
    parser.add_argument("--stage1-mode", choices=("legacy_patch", "dpt_dense"), default="dpt_dense")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--vggt-image-size", type=int, default=518)
    parser.add_argument("--vggt-patch-size", type=int, default=14)
    parser.add_argument("--num-depth-bins", type=int, default=256)
    parser.add_argument("--min-depth-m", type=float, default=0.001)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--metric-scale-init", type=float, default=1.0)
    parser.add_argument("--metric-shift-init", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigloss-warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=35.0)
    parser.add_argument("--dense-gradient-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.image_size % args.patch_size:
        parser.error("image-size must be divisible by patch-size")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be in (0, 1)")
    return args


if __name__ == "__main__":
    run(parse_args())
