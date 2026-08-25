"""Export camera and depth review images from one RoboTwin HDF5 episode."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw


def decode_jpeg(value: object) -> Image.Image:
    if isinstance(value, np.ndarray):
        encoded = value.tobytes()
    elif isinstance(value, np.bytes_):
        encoded = bytes(value)
    else:
        encoded = value
    if not isinstance(encoded, bytes):
        raise TypeError(f"expected encoded bytes, got {type(encoded).__name__}")
    return Image.open(io.BytesIO(encoded.rstrip(b"\0"))).convert("RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 24), fill=(0, 0, 0))
    draw.text((7, 6), label, fill=(255, 255, 255))
    return result


def colorize_depth(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("depth frame has no positive finite pixels")
    lower, upper = np.percentile(depth[valid], (1, 99))
    scaled = np.clip((depth - lower) / max(upper - lower, 1e-6), 0, 1)
    # Invert so nearer surfaces use warmer colors.
    colored_bgr = cv2.applyColorMap(
        np.asarray((1 - scaled) * 255, dtype=np.uint8), cv2.COLORMAP_TURBO
    )
    colored_bgr[~valid] = 0
    return Image.fromarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB))


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (left.width * 2, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def export_review(episode: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with h5py.File(episode, "r") as handle:
        head = handle["vision/cam_head/colors"]
        third = handle["vision/cam_third_view/colors"]
        depths = handle["vision/cam_head/depths"]
        if not (len(head) == len(third) == len(depths)):
            raise ValueError("head, third-view, and depth frame counts differ")

        indices = sorted({0, len(head) // 2, len(head) - 1})
        comparisons: list[Image.Image] = []
        for index in indices:
            head_rgb = decode_jpeg(head[index])
            third_rgb = decode_jpeg(third[index])
            depth_rgb = colorize_depth(np.asarray(depths[index]))
            depth_rgb = depth_rgb.resize(head_rgb.size, Image.Resampling.NEAREST)

            comparison = side_by_side(
                add_label(head_rgb, f"cam_head | frame {index}"),
                add_label(third_rgb, f"cam_third_view | frame {index}"),
            )
            overlay = Image.blend(head_rgb, depth_rgb, alpha=0.38)
            overlay = side_by_side(
                add_label(depth_rgb, f"cam_head depth | frame {index}"),
                add_label(overlay, f"cam_head RGB + depth | frame {index}"),
            )

            for name, image in (
                (f"frame_{index:03d}_head_vs_third.jpg", comparison),
                (f"frame_{index:03d}_head_depth_overlay.jpg", overlay),
            ):
                path = output_dir / name
                image.save(path, quality=95)
                written.append(path)
            comparisons.append(comparison)

        sheet = Image.new(
            "RGB", (comparisons[0].width, comparisons[0].height * len(comparisons))
        )
        for row, comparison in enumerate(comparisons):
            sheet.paste(comparison, (0, row * comparison.height))
        sheet_path = output_dir / "head_vs_third_contact_sheet.jpg"
        sheet.save(sheet_path, quality=95)
        written.append(sheet_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in export_review(args.episode, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
