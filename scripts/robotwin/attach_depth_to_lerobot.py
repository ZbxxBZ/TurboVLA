"""把 RoboTwin 原始 HDF5 的真实深度附加到现有 LeRobot clean50 数据集副本。"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _depth_to_png_bytes(depth: np.ndarray) -> bytes:
    # 统一保存为无损 uint16 毫米 PNG，0 继续表示无效/透明像素。
    depth_u16 = np.rint(depth).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(depth_u16).save(buffer, format="PNG")
    return buffer.getvalue()


def _find_episode_file(raw_data_dir: Path, episode_index: int) -> Path:
    candidates = (
        raw_data_dir / f"episode{episode_index}.hdf5",
        raw_data_dir / f"episode_{episode_index}.hdf5",
        raw_data_dir / f"episode{episode_index:06d}.hdf5",
        raw_data_dir / f"episode_{episode_index:06d}.hdf5",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No raw HDF5 found for episode {episode_index} under {raw_data_dir}"
    )


def _load_episode_data(raw_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    depths: dict[str, np.ndarray] = {}
    with h5py.File(raw_path, "r") as handle:
        for output_camera, raw_camera in CAMERA_MAP.items():
            hdf5_key = f"/observation/{raw_camera}/depth"
            if hdf5_key not in handle:
                raise KeyError(
                    f"{raw_path} does not contain {hdf5_key}; collect with data_type.depth=true"
                )
            values = np.asarray(handle[hdf5_key])
            if values.ndim != 3:
                raise ValueError(f"Expected [T,H,W] depth at {hdf5_key}, got {values.shape}")
            depths[output_camera] = values

        state_key = "/joint_action/vector"
        if state_key not in handle:
            raise KeyError(
                f"{raw_path} does not contain {state_key}; cannot verify RGB/depth trajectory alignment"
            )
        states = np.asarray(handle[state_key])
        if states.ndim != 2:
            raise ValueError(f"Expected [T,D] robot state at {state_key}, got {states.shape}")

    lengths = {camera: len(values) for camera, values in depths.items()}
    lengths["joint_action/vector"] = len(states)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Raw RGB-D episode streams have inconsistent lengths in {raw_path}: {lengths}")
    return depths, states


def _validate_episode_alignment(
    frame_table: pd.DataFrame,
    row_positions: np.ndarray,
    raw_states: np.ndarray,
    parquet_path: Path,
    raw_path: Path,
) -> np.ndarray:
    frame_numbers = frame_table["frame_index"].iloc[row_positions].to_numpy()
    try:
        frame_indices = frame_numbers.astype(np.int64)
        numeric_frames = frame_numbers.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid frame_index values in {parquet_path}") from error
    if not np.isfinite(numeric_frames).all() or not np.array_equal(numeric_frames, frame_indices):
        raise ValueError(f"frame_index must contain finite integers in {parquet_path}")
    if (frame_indices < 0).any() or (frame_indices >= len(raw_states)).any():
        raise IndexError(
            f"Frame indices from {parquet_path} exceed raw episode {raw_path} length {len(raw_states)}"
        )

    try:
        source_states = np.stack(
            [np.asarray(value) for value in frame_table["observation.state"].iloc[row_positions]]
        ).astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot decode observation.state in {parquet_path}") from error
    expected_states = np.asarray(raw_states[frame_indices], dtype=np.float64)
    if source_states.shape != expected_states.shape:
        raise ValueError(
            f"State shape mismatch between {parquet_path} {source_states.shape} and {raw_path} {expected_states.shape}"
        )
    if not np.allclose(source_states, expected_states, rtol=1e-5, atol=1e-5):
        absolute_error = np.abs(source_states - expected_states)
        bad_row = int(np.unravel_index(np.argmax(absolute_error), absolute_error.shape)[0])
        frame_index = int(frame_indices[bad_row])
        raise ValueError(
            "RGB/depth trajectory mismatch: observation.state does not match the raw HDF5 "
            f"at episode frame {frame_index} (max_abs_error={absolute_error.max():.6g}). "
            "Use RGB and depth generated from the same RoboTwin episode/seed."
        )
    return frame_indices


def _augment_parquet(parquet_path: Path, raw_data_dir: Path) -> tuple[int, int, int]:
    frame_table = pd.read_parquet(parquet_path)
    required_columns = {"episode_index", "frame_index", "observation.state"}
    missing = required_columns.difference(frame_table.columns)
    if missing:
        raise KeyError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    encoded_columns = {
        f"observation.depths.{camera}": np.empty(len(frame_table), dtype=object)
        for camera in CAMERA_MAP
    }
    resolution: tuple[int, int] | None = None

    episode_values = frame_table["episode_index"].to_numpy()
    for episode_index in np.unique(episode_values):
        raw_path = _find_episode_file(raw_data_dir, int(episode_index))
        episode_depths, raw_states = _load_episode_data(raw_path)
        row_positions = np.flatnonzero(episode_values == episode_index)
        # 帧数相同不足以证明 RGB-D 对齐；逐帧关节状态必须来自同一条原始轨迹。
        frame_indices = _validate_episode_alignment(
            frame_table,
            row_positions,
            raw_states,
            parquet_path,
            raw_path,
        )

        for row_position, frame_index in zip(row_positions, frame_indices):
            for camera, depth_sequence in episode_depths.items():
                if frame_index >= len(depth_sequence):
                    raise IndexError(
                        f"Frame {frame_index} from {parquet_path} exceeds {raw_path} length {len(depth_sequence)}"
                    )
                depth_frame = depth_sequence[frame_index]
                current_resolution = (int(depth_frame.shape[0]), int(depth_frame.shape[1]))
                if resolution is None:
                    resolution = current_resolution
                elif resolution != current_resolution:
                    raise ValueError(f"Depth resolutions are inconsistent: {resolution} vs {current_resolution}")
                encoded_columns[f"observation.depths.{camera}"][row_position] = _depth_to_png_bytes(depth_frame)

    for column_name, values in encoded_columns.items():
        frame_table[column_name] = values

    # 写回的是输出数据集副本；zstd 对 16-bit PNG 字节列仍能提供少量额外压缩。
    frame_table.to_parquet(parquet_path, index=False, compression="zstd")
    if resolution is None:
        raise ValueError(f"No frames found in {parquet_path}")
    return resolution[0], resolution[1], len(frame_table)


def _update_metadata(dataset_dir: Path, height: int, width: int) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    modality_path = dataset_dir / "meta" / "modality.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    with modality_path.open("r", encoding="utf-8") as handle:
        modality = json.load(handle)

    for camera in CAMERA_MAP:
        feature_key = f"observation.depths.{camera}"
        info["features"][feature_key] = {
            "dtype": "image",
            "shape": [1, height, width],
            "names": ["channel", "height", "width"],
            "depth_dtype": "uint16",
            "depth_unit": "millimeter",
        }
    modality["depth"] = {
        camera: {"original_key": f"observation.depths.{camera}"}
        for camera in CAMERA_MAP
    }

    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    with modality_path.open("w", encoding="utf-8") as handle:
        json.dump(modality, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def _augment_task(source_task: Path, raw_task: Path, output_task: Path) -> None:
    if output_task.exists():
        raise FileExistsError(f"Output task already exists: {output_task}")
    shutil.copytree(source_task, output_task)

    parquet_paths = sorted(output_task.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {output_task / 'data'}")

    common_resolution: tuple[int, int] | None = None
    total_frames = 0
    for parquet_path in parquet_paths:
        height, width, frame_count = _augment_parquet(parquet_path, raw_task)
        if common_resolution is None:
            common_resolution = (height, width)
        elif common_resolution != (height, width):
            raise ValueError(
                f"Task {source_task.name} has multiple depth resolutions: {common_resolution} and {(height, width)}"
            )
        total_frames += frame_count

    assert common_resolution is not None
    _update_metadata(output_task, *common_resolution)
    print(f"[OK] {source_task.name}: {total_frames} frames, depth={common_resolution}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="原 RGB LeRobot 数据集根目录")
    parser.add_argument("--raw-root", type=Path, required=True, help="RoboTwin 原始采集数据根目录")
    parser.add_argument("--output-root", type=Path, required=True, help="新的 RGB-D LeRobot 输出根目录")
    parser.add_argument("--split", default="Clean", help="LeRobot split 目录名")
    parser.add_argument("--raw-config", default="demo_clean_depth", help="原始数据使用的 RoboTwin 配置名")
    parser.add_argument("--tasks", nargs="*", help="只转换指定任务；默认转换 split 下全部任务")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output-root must be outside source-root so the RGB source cannot be overwritten")

    source_split = source_root / args.split
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        output_split = staging_root / args.split
        output_split.mkdir(parents=True)

        tasks = args.tasks or sorted(path.name for path in source_split.iterdir() if path.is_dir())
        for task_name in tasks:
            source_task = source_split / task_name
            raw_task = raw_root / task_name / args.raw_config / "data"
            if not source_task.is_dir():
                raise FileNotFoundError(f"Source task does not exist: {source_task}")
            if not raw_task.is_dir():
                raise FileNotFoundError(f"Raw task data does not exist: {raw_task}")
            _augment_task(source_task, raw_task, output_split / task_name)

        # 全部任务成功后再原子改名，失败时不会留下看似完整的 RGB-D 数据根目录。
        staging_root.replace(output_root)
    except Exception:
        # staging_root 由本进程在 output_root 同级创建，清理范围只包含本次未完成输出。
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"RGB-D dataset written to {output_root}")


if __name__ == "__main__":
    main()
