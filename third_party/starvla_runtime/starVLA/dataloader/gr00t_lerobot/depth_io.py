from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _decode_depth_image(source) -> np.ndarray:
    # 在上下文中完成解码并复制数组，防止大量样本读取时遗留打开的文件或惰性图像句柄。
    with Image.open(source) as image:
        return np.asarray(image).copy()


def decode_depth_entry(entry, dataset_path: Path) -> np.ndarray:
    """解码 parquet 中的无损深度条目，不依赖 RGB 视频/PyAV 后端。"""
    if isinstance(entry, torch.Tensor):
        array = entry.detach().cpu().numpy()
    elif isinstance(entry, np.ndarray):
        array = entry
    elif isinstance(entry, Image.Image):
        array = np.asarray(entry).copy()
    elif isinstance(entry, (bytes, bytearray, memoryview)):
        array = _decode_depth_image(io.BytesIO(bytes(entry)))
    elif isinstance(entry, (str, Path)):
        path = Path(entry)
        if not path.is_absolute():
            path = dataset_path / path
        array = _decode_depth_image(path)
    elif isinstance(entry, dict):
        if entry.get("bytes") is not None:
            array = _decode_depth_image(io.BytesIO(bytes(entry["bytes"])))
        elif entry.get("path") is not None:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = dataset_path / path
            array = _decode_depth_image(path)
        elif entry.get("array") is not None:
            array = np.asarray(entry["array"])
        else:
            raise TypeError("Depth dictionary must contain 'bytes', 'path', or 'array'")
    else:
        raise TypeError(f"Unsupported depth entry type: {type(entry)}")

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Decoded depth must be [H,W], got {array.shape}")
    return array
