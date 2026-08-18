from __future__ import annotations

import io
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "robotwin"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_robotwin_rgbd import scan_task, validate_episode  # noqa: E402


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_episode(path: Path, *, include_depth: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _jpeg_bytes()
    with h5py.File(path, "w") as handle:
        observation = handle.create_group("observation")
        for camera in ("head_camera", "left_camera", "right_camera"):
            camera_group = observation.create_group(camera)
            camera_group.create_dataset("rgb", data=[encoded] * 3, dtype=f"S{len(encoded)}")
            if include_depth:
                camera_group.create_dataset(
                    "depth", data=np.full((3, 6, 8), 750.0, dtype=np.float64)
                )
        joint_action = handle.create_group("joint_action")
        joint_action.create_dataset("vector", data=np.zeros((3, 14), dtype=np.float32))


def _write_xpolicylab_episode(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _jpeg_bytes()
    with h5py.File(path, "w") as handle:
        vision = handle.create_group("vision")
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            camera_group = vision.create_group(camera)
            camera_group.create_dataset(
                "colors", data=[encoded] * 3, dtype=f"S{len(encoded)}"
            )
            camera_group.create_dataset(
                "depths", data=np.full((3, 6, 8), 750.0, dtype=np.float64)
            )
        for group_name in ("state", "action"):
            group = handle.create_group(group_name)
            for field, width in (
                ("left_arm_joint_states", 6),
                ("left_ee_joint_states", 1),
                ("right_arm_joint_states", 6),
                ("right_ee_joint_states", 1),
            ):
                group.create_dataset(field, data=np.zeros((3, width), dtype=np.float32))


def test_validate_complete_rgbd_episode(tmp_path: Path) -> None:
    episode = tmp_path / "episode_0000000.hdf5"
    _write_episode(episode)

    result = validate_episode(episode)

    assert result.valid
    assert result.frames == 3
    assert result.depth_shapes["head_camera"] == [3, 6, 8]


def test_validate_rejects_missing_depth(tmp_path: Path) -> None:
    episode = tmp_path / "episode0.hdf5"
    _write_episode(episode, include_depth=False)

    result = validate_episode(episode)

    assert not result.valid
    assert "depth" in result.error


def test_validate_xpolicylab_rgbd_episode(tmp_path: Path) -> None:
    episode = tmp_path / "episode_0000000.hdf5"
    _write_xpolicylab_episode(episode)

    result = validate_episode(episode)

    assert result.valid
    assert result.frames == 3
    assert result.depth_shapes["left_camera"] == [3, 6, 8]


def test_scan_task_supports_new_and_old_robotwin_layouts(tmp_path: Path) -> None:
    new_episode = (
        tmp_path
        / "new"
        / "demo_clean_depth_turbovla"
        / "click_bell"
        / "aloha_agilex"
        / "data"
        / "episode_0000000.hdf5"
    )
    old_episode = (
        tmp_path
        / "old"
        / "click_bell"
        / "demo_clean_depth_turbovla"
        / "data"
        / "episode0.hdf5"
    )
    _write_episode(new_episode)
    _write_episode(old_episode)

    new_report = scan_task(
        tmp_path / "new", "click_bell", "demo_clean_depth_turbovla", 1
    )
    old_report = scan_task(
        tmp_path / "old", "click_bell", "demo_clean_depth_turbovla", 1
    )

    assert new_report["complete"]
    assert old_report["complete"]
    assert new_report["total_frames"] == old_report["total_frames"] == 3
