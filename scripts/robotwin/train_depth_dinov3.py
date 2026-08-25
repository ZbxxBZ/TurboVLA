#!/usr/bin/env python3
"""Fine-tune the lightweight DINOv3 depth head on RoboTwin cam_head RGB-D."""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from turbovla.models.configuration import DepthEncoderConfig
from turbovla.models.depth_dinov3_loss import DepthLoss
from turbovla.models.depth_encoder import DINOv3DepthEncoder


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def episode_paths(dataset_root: Path) -> list[Path]:
    paths = sorted(dataset_root.glob("**/*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes found below {dataset_root}")
    return paths


def task_name(path: Path) -> str:
    parts = path.parts
    try:
        return parts[parts.index("demo_clean_depth_turbovla") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"cannot infer task name from {path}") from error


def split_episodes(paths: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    by_task: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_task[task_name(path)].append(path)
    train_paths: list[Path] = []
    validation_paths: list[Path] = []
    generator = random.Random(seed)
    for task in sorted(by_task):
        task_paths = sorted(by_task[task])
        generator.shuffle(task_paths)
        validation_count = max(1, round(len(task_paths) * validation_fraction))
        validation_paths.extend(task_paths[:validation_count])
        train_paths.extend(task_paths[validation_count:])
    return sorted(train_paths), sorted(validation_paths)


class RoboTwinHeadRGBD(Dataset):
    def __init__(
        self,
        paths: list[Path],
        *,
        image_size: int,
        frame_stride: int,
        augment: bool,
        open_files: int = 16,
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.open_files = open_files
        self.handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self.frames: list[tuple[Path, int]] = []
        for path in paths:
            with h5py.File(path, "r") as handle:
                count = len(handle["vision/cam_head/colors"])
                if len(handle["vision/cam_head/depths"]) != count:
                    raise ValueError(f"RGB/depth length mismatch in {path}")
            self.frames.extend((path, index) for index in range(0, count, frame_stride))

    def __len__(self) -> int:
        return len(self.frames)

    def _handle(self, path: Path) -> h5py.File:
        handle = self.handles.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r")
        self.handles[path] = handle
        while len(self.handles) > self.open_files:
            _, stale = self.handles.popitem(last=False)
            stale.close()
        return handle

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, frame = self.frames[item]
        handle = self._handle(path)
        encoded = bytes(handle["vision/cam_head/colors"][frame])
        image = Image.open(io.BytesIO(encoded)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb_array = np.asarray(image, dtype=np.uint8).copy()
        rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0

        depth_array = np.asarray(handle["vision/cam_head/depths"][frame], dtype=np.float32).copy()
        depth = torch.from_numpy(depth_array)[None] / 1000.0
        depth = F.interpolate(depth[None], (self.image_size, self.image_size), mode="nearest")[0]
        valid = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))

        if self.augment:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            rgb = (rgb * brightness).clamp(0, 1)
            mean = rgb.mean(dim=(-2, -1), keepdim=True)
            rgb = ((rgb - mean) * contrast + mean).clamp(0, 1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return rgb, depth


def model_config(args: argparse.Namespace, freeze_depth_head: bool) -> DepthEncoderConfig:
    return DepthEncoderConfig(
        enabled=True,
        image_size=args.image_size,
        num_views=3,
        hidden_dim=256,
        patch_size=16,
        head_camera_index=0,
        backbone_name="dinov3_vits16plus",
        backbone_repo_path=str(args.dinov3_repo),
        backbone_weights_path=str(args.dinov3_weights),
        backbone_num_layers=12,
        backbone_hidden_dim=384,
        head_weights_path=str(args.head_weights),
        feature_dim=160,
        freeze_backbone=True,
        freeze_depth_head=freeze_depth_head,
        frozen=False,
        dropout=args.head_dropout,
    )


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, int]:
    mask = torch.isfinite(target) & (target > 0)
    prediction = prediction[mask]
    target = target[mask]
    count = target.numel()
    absolute = (prediction - target).abs()
    return absolute.sum().item(), (absolute / target.clamp_min(1e-3)).sum().item(), absolute.square().sum().item(), count


@torch.inference_mode()
def validate(
    model: DINOv3DepthEncoder,
    loader: DataLoader,
    criterion: DepthLoss,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "l1_sum": 0.0, "abs_rel_sum": 0.0, "squared_sum": 0.0, "pixels": 0}
    batches = 0
    for rgb, depth in loader:
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model.predict_head_depth(rgb)
        loss = criterion(prediction.float(), depth)[0]
        l1_sum, abs_rel_sum, squared_sum, pixels = metrics(prediction.float(), depth)
        totals["loss"] += loss.item()
        totals["l1_sum"] += l1_sum
        totals["abs_rel_sum"] += abs_rel_sum
        totals["squared_sum"] += squared_sum
        totals["pixels"] += pixels
        batches += 1
        if max_batches and batches >= max_batches:
            break
    pixels = max(int(totals["pixels"]), 1)
    return {
        "loss": totals["loss"] / max(batches, 1),
        "mae_m": totals["l1_sum"] / pixels,
        "abs_rel": totals["abs_rel_sum"] / pixels,
        "rmse_m": math.sqrt(totals["squared_sum"] / pixels),
    }


def save_checkpoint(
    path: Path,
    model: DINOv3DepthEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, float],
    best_abs_rel: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "depth_head": model.depth_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation": validation,
            "best_abs_rel": best_abs_rel,
            "config": asdict(model.config),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
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

    model = DINOv3DepthEncoder(model_config(args, freeze_depth_head=False)).to(device)
    model.backbone.eval()
    optimizer = torch.optim.AdamW(model.depth_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = DepthLoss(w_data=1.0, w_sig=1.0, w_grad=5.0).to(device)

    start_epoch = 1
    best = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.depth_head.load_state_dict(checkpoint["depth_head"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(
            checkpoint.get(
                "best_abs_rel",
                checkpoint.get("validation", {}).get("abs_rel", float("inf")),
            )
        )
        existing_best_path = args.output_dir / "best.pt"
        if existing_best_path.is_file():
            existing_best = torch.load(existing_best_path, map_location="cpu", weights_only=True)
            best = min(best, float(existing_best.get("validation", {}).get("abs_rel", best)))
        print(
            f"resumed={args.resume} completed_epoch={start_epoch - 1} "
            f"target_epoch={args.epochs} learning_rate={optimizer.param_groups[0]['lr']:.8g} "
            f"best_abs_rel={best:.8f}",
            flush=True,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"resume checkpoint already completed epoch {start_epoch - 1}, "
            f"but target --epochs is {args.epochs}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "train_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "tasks": len({task_name(path) for path in paths}),
    }
    (args.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)

    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for step, (rgb, depth) in enumerate(train_loader, start=1):
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model.predict_head_depth(rgb)
            losses = criterion(prediction.float(), depth)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), args.max_grad_norm)
            optimizer.step()
            running += losses[0].item()
            global_step += 1
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} global_step={global_step} "
                    f"loss={running / args.log_every:.6f} l1={losses[1].item():.6f} "
                    f"sig={losses[2].item():.6f} grad={losses[3].item():.6f}",
                    flush=True,
                )
                running = 0.0
            if args.max_steps and global_step >= args.max_steps:
                break

        validation = validate(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=args.max_validation_batches,
        )
        validation["elapsed_seconds"] = time.time() - started
        print(f"epoch={epoch} validation={json.dumps(validation, sort_keys=True)}", flush=True)
        is_best = validation["abs_rel"] < best
        if is_best:
            best = validation["abs_rel"]
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, validation, best, args)
        save_checkpoint(
            args.output_dir / f"epoch_{epoch:02d}.pt",
            model,
            optimizer,
            epoch,
            validation,
            best,
            args,
        )
        if is_best:
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, validation, best, args)
        if args.max_steps and global_step >= args.max_steps:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--dinov3-weights", type=Path, required=True)
    parser.add_argument("--head-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    if args.frame_stride < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("frame stride, batch size, and epochs must be positive")
    if args.max_steps < 0 or args.max_validation_batches < 0:
        parser.error("step and validation batch limits cannot be negative")
    if not 0 <= args.head_dropout < 1:
        parser.error("--head-dropout must be in [0, 1)")
    return args


if __name__ == "__main__":
    train(parse_args())
