#!/usr/bin/env python3
"""Visualize Stage 1 VGGT-DPT depth predictions on random RoboTwin frames.

The script loads the Stage 1 adapter checkpoint, keeps the VGGT aggregator and
DPT head frozen, samples a small number of cam_head RGB-D frames, and writes a
four-panel PNG for each frame: RGB, predicted depth, ground-truth depth, and
absolute error.  It is intentionally inference-only and does not modify any
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.robotwin.train_depth_dinov3 import RoboTwinHeadRGBD, episode_paths
from turbovla.models.configuration import DepthEncoderConfig
from turbovla.models.vggt_depth_encoder import VGGTDepthEncoder


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_config(checkpoint: dict[str, object], args: argparse.Namespace) -> DepthEncoderConfig:
    raw_config = checkpoint.get("depth_encoder_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Stage 1 checkpoint is missing depth_encoder_config")
    valid_fields = set(DepthEncoderConfig.__dataclass_fields__)
    config_values = {key: value for key, value in raw_config.items() if key in valid_fields}
    checkpoint_mode = checkpoint.get("stage1_mode") or config_values.get("stage1_mode")
    config_values.update(
        {
            "enabled": True,
            "backend": "vggt",
            "stage1_mode": str(checkpoint_mode or ""),
            "image_size": args.image_size,
            "vggt_repo_path": str(args.vggt_repo),
            "vggt_weights_path": str(args.vggt_weights),
            "adapter_weights_path": str(args.checkpoint),
            "freeze_backbone": True,
            "freeze_depth_head": True,
            "frozen": True,
        }
    )
    if config_values["stage1_mode"] != "dpt_dense":
        raise ValueError(
            "This visualizer is for the dense DPT Stage 1 checkpoint; "
            f"received stage1_mode={config_values['stage1_mode']!r}"
        )
    return DepthEncoderConfig(**config_values)


def _denormalize_rgb(rgb: torch.Tensor) -> np.ndarray:
    array = rgb.detach().cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(array.transpose(1, 2, 0), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _finite_depth(depth: torch.Tensor) -> np.ndarray:
    array = depth.detach().float().cpu().numpy()
    return np.where(np.isfinite(array) & (array > 0.0), array, 0.0).astype(np.float32)


def _depth_to_rgb(depth: np.ndarray, display_max: float) -> Image.Image:
    """Render a depth array without requiring a plotting package."""
    scale = np.clip(depth / max(display_max, 1e-6), 0.0, 1.0)
    # A compact blue-to-cyan-to-yellow-to-red ramp, similar to common depth maps.
    anchors = np.asarray(
        (
            (0.02, 0.02, 0.18),
            (0.00, 0.55, 0.85),
            (0.10, 0.85, 0.35),
            (0.95, 0.85, 0.05),
            (0.85, 0.05, 0.02),
        ),
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(anchors), dtype=np.float32)
    rgb = np.empty((*depth.shape, 3), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.interp(scale, positions, anchors[:, channel])
    rgb[depth <= 0.0] = 0.0
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")


def _error_to_rgb(error: np.ndarray, valid: np.ndarray, display_max: float) -> Image.Image:
    scale = np.clip(error / max(display_max, 1e-6), 0.0, 1.0)
    # Black means invalid/no error; valid errors go from dark blue to yellow/red.
    rgb = np.stack((scale, scale * scale, 1.0 - scale), axis=-1)
    rgb[~valid] = 0.0
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")


def _captioned(image: Image.Image, title: str, *, height: int = 250) -> Image.Image:
    panel = Image.new("RGB", (image.width, height), "white")
    panel.paste(image.resize((image.width, image.width), Image.Resampling.NEAREST), (0, 24))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - Pillow always provides the default font in practice.
        font = None
    draw.text((6, 6), title, fill="black", font=font)
    return panel


def _make_panel(
    rgb: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    error: np.ndarray,
    valid: np.ndarray,
    *,
    display_max: float,
    error_max: float,
) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    panels = (
        _captioned(image, "cam_head RGB"),
        _captioned(_depth_to_rgb(prediction, display_max), "predicted depth (m)"),
        _captioned(_depth_to_rgb(target, display_max), "ground truth depth (m)"),
        _captioned(_error_to_rgb(error, valid, error_max), "absolute error (m)"),
    )
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    return canvas


def _sample_indices(dataset: RoboTwinHeadRGBD, count: int, seed: int) -> list[int]:
    if count < 1:
        raise ValueError("num-frames must be positive")
    if not len(dataset):
        raise ValueError("the RoboTwin dataset contains no frames")
    generator = random.Random(seed)
    return generator.sample(range(len(dataset)), min(count, len(dataset)))


def run(args: argparse.Namespace) -> None:
    if args.frame_stride < 1:
        raise ValueError("frame-stride must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Stage 1 checkpoint must be a mapping")

    paths = episode_paths(args.dataset_root)
    dataset = RoboTwinHeadRGBD(
        paths,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        augment=False,
    )
    device = torch.device(args.device)
    config = _load_config(checkpoint, args)
    model = VGGTDepthEncoder(config).to(device).eval()
    # This is an inference visualization; make the freeze contract explicit.
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("visualizer expected the VGGT/DPT model to be fully frozen")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    panels: list[Image.Image] = []
    with torch.inference_mode():
        for output_index, dataset_index in enumerate(_sample_indices(dataset, args.num_frames, args.seed), start=1):
            rgb, target = dataset[dataset_index]
            prediction = model.predict_head_depth(rgb.unsqueeze(0).to(device))[0, 0].cpu()
            target_array = _finite_depth(target[0])
            prediction_array = _finite_depth(prediction)
            valid = np.isfinite(target_array) & (target_array > 0.0)
            error = np.abs(prediction_array - target_array)
            valid_error = error[valid]
            valid_target = target_array[valid]
            if not len(valid_error):
                raise ValueError(f"sample {dataset_index} contains no valid depth pixels")
            display_max = float(
                max(
                    np.percentile(valid_target, 99.0),
                    np.percentile(prediction_array[prediction_array > 0.0], 99.0)
                    if (prediction_array > 0.0).any()
                    else 0.0,
                    1e-3,
                )
            )
            error_max = max(float(np.percentile(valid_error, 99.0)), 1e-3)
            source_path, source_frame = dataset.frames[dataset_index]
            record = {
                "index": output_index,
                "dataset_index": dataset_index,
                "source": str(source_path),
                "frame": int(source_frame),
                "valid_pixels": int(valid.sum()),
                "display_max_m": display_max,
                "error_display_max_m": error_max,
                "mae_m": float(valid_error.mean()),
                "abs_rel": float((valid_error / np.maximum(valid_target, 1e-3)).mean()),
                "rmse_m": float(np.sqrt(np.square(valid_error).mean())),
            }
            records.append(record)
            panel = _make_panel(
                _denormalize_rgb(rgb),
                prediction_array,
                target_array,
                error,
                valid,
                display_max=display_max,
                error_max=error_max,
            )
            panel.save(args.output_dir / f"frame_{output_index:02d}.png")
            panels.append(panel)

    contact_sheet = Image.new(
        "RGB",
        (max(panel.width for panel in panels), sum(panel.height for panel in panels)),
        "white",
    )
    offset = 0
    for panel in panels:
        contact_sheet.paste(panel, (0, offset))
        offset += panel.height
    contact_sheet.save(args.output_dir / "contact_sheet.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "stage1_mode": config.stage1_mode,
        "seed": args.seed,
        "num_frames": len(records),
        "image_size": args.image_size,
        "model_frozen": True,
        "contact_sheet": str(args.output_dir / "contact_sheet.png"),
        "config": asdict(config),
        "frames": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run(_parse_args())
