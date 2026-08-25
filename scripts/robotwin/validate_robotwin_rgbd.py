"""Validate raw RoboTwin episodes collected with synchronized RGB and depth."""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


LEGACY_CAMERAS = {
    "head_camera": "head_camera",
    "left_camera": "left_camera",
    "right_camera": "right_camera",
}
XPOLICY_CAMERAS = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
XPOLICY_JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
EPISODE_RE = re.compile(r"episode_?(\d+)\.hdf5$")
_VALID_EPISODE_CACHE: dict[tuple[str, int, int, int], "EpisodeValidation"] = {}


@dataclass(frozen=True)
class EpisodeValidation:
    path: str
    index: int
    valid: bool
    frames: int = 0
    error: str = ""
    depth_dtypes: dict[str, str] | None = None
    depth_shapes: dict[str, list[int]] | None = None


def _check_rgb_samples(dataset: h5py.Dataset, key: str) -> tuple[int, int]:
    if dataset.ndim != 1:
        raise ValueError(f"{key} must be [T] encoded images, got {dataset.shape}")
    sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    sample_shape: tuple[int, int] | None = None
    for index in sample_indices:
        encoded = dataset[index]
        if isinstance(encoded, np.ndarray):
            encoded = encoded.tobytes()
        elif isinstance(encoded, np.bytes_):
            encoded = bytes(encoded)
        if not isinstance(encoded, bytes):
            raise ValueError(f"{key}[{index}] is not an encoded image")
        encoded = encoded.rstrip(b"\0")
        if len(encoded) < 32:
            raise ValueError(f"{key}[{index}] is not a non-empty encoded image")
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                shape = (image.height, image.width)
                image.verify()
        except Exception as error:
            raise ValueError(f"{key}[{index}] cannot be decoded: {error}") from error
        if sample_shape is None:
            sample_shape = shape
        elif shape != sample_shape:
            raise ValueError(f"{key} changes resolution within one episode")
    if sample_shape is None:
        raise ValueError(f"{key} contains no images")
    return sample_shape


def _check_depth(
    dataset: h5py.Dataset, key: str, expected_shape: tuple[int, int]
) -> None:
    if dataset.ndim != 3:
        raise ValueError(f"{key} must be [T,H,W], got {dataset.shape}")
    if tuple(dataset.shape[1:]) != expected_shape:
        raise ValueError(
            f"{key} spatial shape {dataset.shape[1:]} does not match RGB {expected_shape}"
        )
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"{key} must be numeric, got {dataset.dtype}")

    has_positive_depth = False
    for start in range(0, len(dataset), 16):
        values = np.asarray(dataset[start : start + 16])
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN or Inf")
        if (values < 0).any():
            raise ValueError(f"{key} contains negative metric depth")
        has_positive_depth = has_positive_depth or bool((values > 0).any())
    if not has_positive_depth:
        raise ValueError(f"{key} is entirely zero")


def _check_numeric_matrix(dataset: h5py.Dataset, key: str) -> tuple[int, int]:
    if dataset.ndim != 2:
        raise ValueError(f"{key} must be [T,D], got {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"{key} must be numeric, got {dataset.dtype}")
    values = np.asarray(dataset)
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains NaN or Inf")
    return len(dataset), int(dataset.shape[1])


def _check_camera_streams(
    handle: h5py.File,
    camera_map: dict[str, str],
    *,
    root: str,
    rgb_field: str,
    depth_field: str,
) -> tuple[dict[str, int], dict[str, str], dict[str, list[int]]]:
    lengths: dict[str, int] = {}
    depth_dtypes: dict[str, str] = {}
    depth_shapes: dict[str, list[int]] = {}
    for output_camera, stored_camera in camera_map.items():
        rgb_key = f"/{root}/{stored_camera}/{rgb_field}"
        depth_key = f"/{root}/{stored_camera}/{depth_field}"
        if rgb_key not in handle:
            raise KeyError(f"missing {rgb_key}")

        rgb = handle[rgb_key]
        lengths[rgb_key] = len(rgb)
        rgb_shape = _check_rgb_samples(rgb, rgb_key)
        if output_camera == "head_camera":
            if depth_key not in handle:
                raise KeyError(f"missing {depth_key}")
            depth = handle[depth_key]
            lengths[depth_key] = len(depth)
            _check_depth(depth, depth_key, rgb_shape)
            depth_dtypes[output_camera] = str(depth.dtype)
            depth_shapes[output_camera] = list(depth.shape)
        elif depth_key in handle:
            raise ValueError(f"wrist depth must not be stored: {depth_key}")
    return lengths, depth_dtypes, depth_shapes


def _check_pointcloud(dataset: h5py.Dataset, key: str) -> int:
    if dataset.ndim != 3 or dataset.shape[1:] != (1024, 6):
        raise ValueError(f"{key} must be [T,1024,6], got {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"{key} must be numeric, got {dataset.dtype}")
    has_geometry = False
    for start in range(0, len(dataset), 16):
        values = np.asarray(dataset[start : start + 16])
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN or Inf")
        has_geometry = has_geometry or bool(
            (np.linalg.norm(values[..., :3], axis=-1) > 1e-6).any()
        )
        colors = values[..., 3:]
        if (colors < 0).any() or (colors > 1).any():
            raise ValueError(f"{key} RGB values must be in [0,1]")
    if not has_geometry:
        raise ValueError(f"{key} contains no non-zero XYZ points")
    return len(dataset)


def _check_legacy_schema(
    handle: h5py.File,
) -> tuple[dict[str, int], dict[str, str], dict[str, list[int]]]:
    lengths, depth_dtypes, depth_shapes = _check_camera_streams(
        handle,
        LEGACY_CAMERAS,
        root="observation",
        rgb_field="rgb",
        depth_field="depth",
    )
    state_key = "/joint_action/vector"
    if state_key not in handle:
        raise KeyError(f"missing {state_key}")
    state_length, state_width = _check_numeric_matrix(handle[state_key], state_key)
    if state_width != 14:
        raise ValueError(
            f"{state_key} must contain 14 Aloha joint/gripper values, "
            f"got {handle[state_key].shape}"
        )
    lengths[state_key] = state_length
    return lengths, depth_dtypes, depth_shapes


def _check_xpolicylab_schema(
    handle: h5py.File,
) -> tuple[dict[str, int], dict[str, str], dict[str, list[int]]]:
    if "/vision/cam_third_view" in handle:
        raise ValueError("cam_third_view must not be stored; use cam_head")
    lengths, depth_dtypes, depth_shapes = _check_camera_streams(
        handle,
        XPOLICY_CAMERAS,
        root="vision",
        rgb_field="colors",
        depth_field="depths",
    )
    for group_name in ("state", "action"):
        total_width = 0
        for field in XPOLICY_JOINT_FIELDS:
            key = f"/{group_name}/{field}"
            if key not in handle:
                raise KeyError(f"missing {key}")
            stream_length, stream_width = _check_numeric_matrix(handle[key], key)
            lengths[key] = stream_length
            total_width += stream_width
        if total_width != 14:
            raise ValueError(
                f"/{group_name} joint fields must total 14 Aloha values, got {total_width}"
            )
    pointcloud_key = "/pointclouds"
    if pointcloud_key not in handle:
        raise KeyError(f"missing {pointcloud_key}")
    lengths[pointcloud_key] = _check_pointcloud(
        handle[pointcloud_key], pointcloud_key
    )
    return lengths, depth_dtypes, depth_shapes


def validate_episode(path: Path, *, expected_index: int | None = None) -> EpisodeValidation:
    path = Path(path)
    match = EPISODE_RE.fullmatch(path.name)
    index = int(match.group(1)) if match else -1
    if expected_index is not None:
        index = expected_index

    if not path.is_file():
        return EpisodeValidation(
            path=str(path), index=index, valid=False, error="episode file is missing"
        )
    file_stat = path.stat()
    cache_key = (str(path.resolve()), file_stat.st_size, file_stat.st_mtime_ns, index)
    cached = _VALID_EPISODE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with h5py.File(path, "r") as handle:
            if "/vision/cam_head/colors" in handle:
                lengths, depth_dtypes, depth_shapes = _check_xpolicylab_schema(handle)
            else:
                lengths, depth_dtypes, depth_shapes = _check_legacy_schema(handle)

            if not lengths or min(lengths.values()) <= 0:
                raise ValueError(f"episode has no frames: {lengths}")
            if len(set(lengths.values())) != 1:
                raise ValueError(f"RGB/depth/state frame counts differ: {lengths}")

        result = EpisodeValidation(
            path=str(path),
            index=index,
            valid=True,
            frames=next(iter(lengths.values())),
            depth_dtypes=depth_dtypes,
            depth_shapes=depth_shapes,
        )
        _VALID_EPISODE_CACHE[cache_key] = result
        return result
    except Exception as error:
        return EpisodeValidation(path=str(path), index=index, valid=False, error=str(error))


def scan_task(
    output_root: Path,
    task: str,
    config_name: str,
    target_episodes: int,
    embodiment_dir: str = "aloha_agilex",
) -> dict[str, Any]:
    output_root = Path(output_root)
    task_dir_candidates = (
        output_root / config_name / task / embodiment_dir,
        output_root / task / config_name,
    )
    populated_candidates = [
        candidate
        for candidate in task_dir_candidates
        if candidate.exists()
        or (candidate / "seed.txt").exists()
        or (candidate / "data").exists()
    ]
    task_dir = populated_candidates[0] if populated_candidates else task_dir_candidates[0]
    data_dir = task_dir / "data"

    episode_paths: dict[int, Path] = {}
    if data_dir.is_dir():
        for path in data_dir.glob("episode*.hdf5"):
            match = EPISODE_RE.fullmatch(path.name)
            if match:
                episode_paths[int(match.group(1))] = path
    episodes = [
        validate_episode(
            episode_paths.get(index, data_dir / f"episode_{index:07d}.hdf5"),
            expected_index=index,
        )
        for index in range(target_episodes)
    ]
    valid_indices = [episode.index for episode in episodes if episode.valid]
    contiguous = 0
    for episode in episodes:
        if not episode.valid:
            break
        contiguous += 1
    return {
        "task": task,
        "task_dir": str(task_dir),
        "data_dir": str(data_dir),
        "target_episodes": target_episodes,
        "valid_episodes": len(valid_indices),
        "contiguous_episodes": contiguous,
        "complete": len(valid_indices) == target_episodes,
        "total_frames": sum(episode.frames for episode in episodes if episode.valid),
        "episodes": [asdict(episode) for episode in episodes],
    }


def _read_tasks(task_file: Path) -> list[str]:
    tasks = []
    for raw_line in task_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            tasks.append(line)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-name", default="demo_clean_depth_turbovla")
    parser.add_argument("--target-episodes", type=int, default=10)
    parser.add_argument("--embodiment-dir", default="aloha_agilex")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args()

    tasks = args.tasks
    if args.task_file:
        tasks.extend(_read_tasks(args.task_file))
    if not tasks:
        parser.error("provide task names or --task-file")

    reports = [
        scan_task(
            args.output_root,
            task,
            args.config_name,
            args.target_episodes,
            args.embodiment_dir,
        )
        for task in tasks
    ]
    summary = {
        "complete": all(report["complete"] for report in reports),
        "valid_episodes": sum(report["valid_episodes"] for report in reports),
        "target_episodes": args.target_episodes * len(tasks),
        "total_frames": sum(report["total_frames"] for report in reports),
        "tasks": reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
