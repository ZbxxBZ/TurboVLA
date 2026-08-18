"""Convert synchronized RoboTwin XPolicyLab HDF5 episodes to LeRobot v2.1.

The converter reads RGB, metric depth, robot state, actions, and language from
the same HDF5 episode. RGB and depth are embedded losslessly in parquet image
columns so their frame correspondence cannot drift during conversion.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image


CAMERA_MAP = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
EPISODE_RE = re.compile(r"episode_?(\d+)\.hdf5$")


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_tasks(task_file: Path | None, positional_tasks: list[str]) -> list[str]:
    tasks = list(positional_tasks)
    if task_file is not None:
        for raw_line in task_file.read_text(encoding="utf-8").splitlines():
            task = raw_line.split("#", 1)[0].strip()
            if task:
                tasks.append(task)
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        raise ValueError("provide task names or --task-file")
    return tasks


def _episode_paths(data_dir: Path, max_episodes: int | None) -> list[Path]:
    indexed_paths: list[tuple[int, Path]] = []
    for path in data_dir.glob("episode*.hdf5"):
        match = EPISODE_RE.fullmatch(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort()
    if max_episodes is not None:
        indexed_paths = indexed_paths[:max_episodes]
    if not indexed_paths:
        raise FileNotFoundError(f"no HDF5 episodes found under {data_dir}")
    expected = list(range(len(indexed_paths)))
    actual = [index for index, _ in indexed_paths]
    if actual != expected:
        raise ValueError(f"episode indices under {data_dir} are not contiguous: {actual}")
    return [path for _, path in indexed_paths]


def _decode_json_string(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _choose_instruction(handle: h5py.File, episode_index: int, task: str) -> str:
    if "/instruction" in handle:
        instruction = _decode_json_string(handle["/instruction"][()])
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
    if "/instructions" in handle:
        instructions = _decode_json_string(handle["/instructions"][()])
        if isinstance(instructions, str):
            instructions = [instructions]
        if isinstance(instructions, list):
            instructions = [str(value).strip() for value in instructions if str(value).strip()]
            if instructions:
                return instructions[episode_index % len(instructions)]
    return task.replace("_", " ")


def _encoded_rgb(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError(f"RGB entry must contain encoded bytes, got {type(value)}")
    value = value.rstrip(b"\0")
    if len(value) < 32:
        raise ValueError("RGB entry is empty or truncated")
    return value


def _image_entry(value: bytes) -> dict[str, Any]:
    return {"bytes": value, "path": None}


def _depth_png(depth: np.ndarray) -> bytes:
    values = np.asarray(depth)
    if not np.isfinite(values).all():
        raise ValueError("depth frame contains NaN or Inf")
    if (values < 0).any():
        raise ValueError("depth frame contains negative values")
    depth_u16 = np.rint(values).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(depth_u16).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _joint_vector(handle: h5py.File, group_name: str, frames: int) -> np.ndarray:
    arrays = []
    for field in JOINT_FIELDS:
        key = f"/{group_name}/{field}"
        if key not in handle:
            raise KeyError(f"missing {key}")
        array = np.asarray(handle[key], dtype=np.float32)
        if array.ndim != 2 or len(array) != frames:
            raise ValueError(f"{key} must be [T,D] with T={frames}, got {array.shape}")
        arrays.append(array)
    vector = np.concatenate(arrays, axis=1)
    if vector.shape != (frames, 14):
        raise ValueError(f"/{group_name} fields must concatenate to [T,14], got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"/{group_name} contains NaN or Inf")
    return vector


def _frequency(handle: h5py.File) -> int:
    key = "/additional_info/frequency"
    if key not in handle:
        raise KeyError(f"missing {key}")
    frequency = int(np.asarray(handle[key]).item())
    if frequency <= 0:
        raise ValueError(f"invalid episode frequency: {frequency}")
    return frequency


def _episode_frame_table(
    path: Path,
    *,
    task: str,
    episode_index: int,
    task_index: int,
    global_start_index: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, tuple[int, int]]:
    with h5py.File(path, "r") as handle:
        first_rgb_key = f"/vision/{CAMERA_MAP['cam_high']}/colors"
        if first_rgb_key not in handle:
            raise KeyError(f"missing {first_rgb_key}")
        frames = len(handle[first_rgb_key])
        if frames <= 0:
            raise ValueError(f"episode has no frames: {path}")
        frequency = _frequency(handle)
        instruction = _choose_instruction(handle, episode_index, task)
        state = _joint_vector(handle, "state", frames)
        action = _joint_vector(handle, "action", frames)

        columns: dict[str, Any] = {
            "observation.state": list(state),
            "action": list(action),
            "timestamp": np.arange(frames, dtype=np.float32) / float(frequency),
            "frame_index": np.arange(frames, dtype=np.int64),
            "episode_index": np.full(frames, episode_index, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + frames, dtype=np.int64),
            "task_index": np.full(frames, task_index, dtype=np.int64),
        }

        spatial_shape: tuple[int, int] | None = None
        for output_camera, stored_camera in CAMERA_MAP.items():
            rgb_key = f"/vision/{stored_camera}/colors"
            depth_key = f"/vision/{stored_camera}/depths"
            if rgb_key not in handle or depth_key not in handle:
                raise KeyError(f"missing synchronized RGB-D pair {rgb_key}, {depth_key}")
            rgb = handle[rgb_key]
            depth = handle[depth_key]
            if len(rgb) != frames or len(depth) != frames or depth.ndim != 3:
                raise ValueError(
                    f"stream length/shape mismatch in {path}: RGB={rgb.shape}, depth={depth.shape}"
                )
            current_shape = (int(depth.shape[1]), int(depth.shape[2]))
            if spatial_shape is None:
                spatial_shape = current_shape
            elif current_shape != spatial_shape:
                raise ValueError(f"camera resolutions differ in {path}")
            columns[f"observation.images.{output_camera}"] = [
                _image_entry(_encoded_rgb(rgb[index])) for index in range(frames)
            ]
            columns[f"observation.depths.{output_camera}"] = [
                _image_entry(_depth_png(depth[index])) for index in range(frames)
            ]

    assert spatial_shape is not None
    table = pd.DataFrame(columns)
    table.attrs["instruction"] = instruction
    return table, state, action, frequency, spatial_shape


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _modality_metadata() -> dict[str, Any]:
    return {
        "action": {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in CAMERA_MAP
        },
        "depth": {
            camera: {"original_key": f"observation.depths.{camera}"}
            for camera in CAMERA_MAP
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }


def _info_metadata(
    *, episodes: int, frames: int, tasks: int, frequency: int, shape: tuple[int, int]
) -> dict[str, Any]:
    height, width = shape
    joint_names = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [joint_names],
        },
        "action": {"dtype": "float32", "shape": [14], "names": [joint_names]},
    }
    for camera in CAMERA_MAP:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": [3, height, width],
            "names": ["channels", "height", "width"],
        }
        features[f"observation.depths.{camera}"] = {
            "dtype": "image",
            "shape": [1, height, width],
            "names": ["channel", "height", "width"],
            "depth_dtype": "uint16",
            "depth_unit": "millimeter",
        }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": frequency,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def convert_task(
    source_task: Path,
    output_task: Path,
    *,
    task: str,
    max_episodes: int | None = None,
) -> dict[str, int]:
    data_dir = source_task / "data"
    episode_paths = _episode_paths(data_dir, max_episodes)
    parquet_dir = output_task / "data" / "chunk-000"
    meta_dir = output_task / "meta"
    parquet_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    task_indices: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    total_frames = 0
    common_frequency: int | None = None
    common_shape: tuple[int, int] | None = None

    for episode_index, source_path in enumerate(episode_paths):
        with h5py.File(source_path, "r") as handle:
            instruction = _choose_instruction(handle, episode_index, task)
        if instruction not in task_indices:
            task_indices[instruction] = len(task_indices)
            task_rows.append({"task_index": task_indices[instruction], "task": instruction})
        task_index = task_indices[instruction]
        table, state, action, frequency, shape = _episode_frame_table(
            source_path,
            task=task,
            episode_index=episode_index,
            task_index=task_index,
            global_start_index=total_frames,
        )
        if common_frequency is None:
            common_frequency = frequency
            common_shape = shape
        elif frequency != common_frequency or shape != common_shape:
            raise ValueError(
                f"task {task} changes frequency/resolution: "
                f"{common_frequency}/{common_shape} vs {frequency}/{shape}"
            )
        destination = parquet_dir / f"episode_{episode_index:06d}.parquet"
        table.to_parquet(destination, index=False, compression="zstd")
        episode_rows.append(
            {"episode_index": episode_index, "tasks": [instruction], "length": len(table)}
        )
        all_states.append(state)
        all_actions.append(action)
        total_frames += len(table)
        print(
            f"[OK] {task} episode {episode_index + 1}/{len(episode_paths)}: "
            f"{len(table)} frames",
            flush=True,
        )

    assert common_frequency is not None and common_shape is not None
    _write_jsonl(meta_dir / "tasks.jsonl", task_rows)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    _json_dump(meta_dir / "modality.json", _modality_metadata())
    _json_dump(
        meta_dir / "info.json",
        _info_metadata(
            episodes=len(episode_rows),
            frames=total_frames,
            tasks=len(task_rows),
            frequency=common_frequency,
            shape=common_shape,
        ),
    )
    _json_dump(
        meta_dir / "stats_gr00t.json",
        {
            "__format_version": 2,
            "__cache_config": {"mode": "abs"},
            "statistics": {
                "observation.state": _statistics(np.concatenate(all_states)),
                "action": _statistics(np.concatenate(all_actions)),
            },
        },
    )
    return {"episodes": len(episode_rows), "frames": total_frames}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-name", default="demo_clean_depth_turbovla")
    parser.add_argument("--embodiment-dir", default="aloha_agilex")
    parser.add_argument("--split", default="Clean")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("tasks", nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tasks = _read_tasks(args.task_file, args.tasks)
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("output root must be outside the raw data root")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        split_root = staging_root / args.split
        split_root.mkdir(parents=True)
        total_episodes = 0
        total_frames = 0
        for task in tasks:
            source_task = (
                raw_root / args.config_name / task / args.embodiment_dir
            )
            if not source_task.is_dir():
                raise FileNotFoundError(f"raw task does not exist: {source_task}")
            report = convert_task(
                source_task,
                split_root / task,
                task=task,
                max_episodes=args.max_episodes,
            )
            total_episodes += report["episodes"]
            total_frames += report["frames"]
        _json_dump(
            staging_root / "conversion_summary.json",
            {
                "source": str(raw_root),
                "tasks": tasks,
                "episodes": total_episodes,
                "frames": total_frames,
                "depth_unit": "millimeter",
            },
        )
        staging_root.replace(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    print(
        f"[DONE] wrote {total_episodes} episodes and {total_frames} frames to {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
