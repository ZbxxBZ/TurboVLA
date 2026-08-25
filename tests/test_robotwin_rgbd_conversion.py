from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from scripts.robotwin.convert_rgbd_to_lerobot import convert_task
from starVLA.dataloader.gr00t_lerobot.depth_io import decode_depth_entry


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True)
    frames = 3
    state = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14)
    action = state + 1
    with h5py.File(path, "w") as handle:
        vision = handle.create_group("vision")
        for camera_index, camera in enumerate(
            ("cam_head", "cam_left_wrist", "cam_right_wrist")
        ):
            group = vision.create_group(camera)
            encoded = _jpeg_bytes((20 + camera_index, 40, 60))
            group.create_dataset("colors", data=[encoded] * frames, dtype=f"S{len(encoded)}")
            if camera == "cam_head":
                group.create_dataset(
                    "depths",
                    data=np.stack(
                        [
                            np.full((6, 8), 500 + frame * 100, dtype=np.float32)
                            for frame in range(frames)
                        ]
                    ),
                )
        for group_name, vector in (("state", state), ("action", action)):
            group = handle.create_group(group_name)
            group.create_dataset("left_arm_joint_states", data=vector[:, :6])
            group.create_dataset("left_ee_joint_states", data=vector[:, 6:7])
            group.create_dataset("right_arm_joint_states", data=vector[:, 7:13])
            group.create_dataset("right_ee_joint_states", data=vector[:, 13:14])
        additional_info = handle.create_group("additional_info")
        additional_info.create_dataset("frequency", data=np.int32(15))
        handle.create_dataset(
            "instructions",
            data=json.dumps(["first instruction", "second instruction"]),
            dtype=h5py.string_dtype("utf-8"),
        )
    return state, action


def test_convert_xpolicylab_rgbd_episode_to_lerobot(tmp_path: Path) -> None:
    source_task = tmp_path / "source" / "click_bell" / "aloha_agilex"
    expected_state, expected_action = _write_episode(
        source_task / "data" / "episode_0000000.hdf5"
    )
    output_task = tmp_path / "output" / "Clean" / "click_bell"

    report = convert_task(source_task, output_task, task="click_bell")

    assert report == {"episodes": 1, "frames": 3}
    parquet = pd.read_parquet(
        output_task / "data" / "chunk-000" / "episode_000000.parquet"
    )
    np.testing.assert_array_equal(np.stack(parquet["observation.state"]), expected_state)
    np.testing.assert_array_equal(np.stack(parquet["action"]), expected_action)
    assert parquet["timestamp"].tolist() == [0.0, np.float32(1 / 15), np.float32(2 / 15)]
    assert parquet["task_index"].tolist() == [0, 0, 0]

    rgb_entry = parquet["observation.images.cam_high"].iloc[0]
    with Image.open(io.BytesIO(rgb_entry["bytes"])) as image:
        assert image.size == (8, 6)
    depth = decode_depth_entry(
        parquet["observation.depths.cam_high"].iloc[2], output_task
    )
    np.testing.assert_array_equal(depth, np.full((6, 8), 700, dtype=np.uint16))
    assert "observation.depths.cam_left_wrist" not in parquet
    assert "observation.depths.cam_right_wrist" not in parquet

    with (output_task / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    with (output_task / "meta" / "tasks.jsonl").open(encoding="utf-8") as handle:
        task = json.loads(handle.readline())
    with (output_task / "meta" / "stats_gr00t.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)

    assert info["total_videos"] == 0
    assert info["fps"] == 15
    assert info["features"]["observation.depths.cam_high"]["depth_unit"] == "millimeter"
    assert "observation.depths.cam_left_wrist" not in info["features"]
    assert "observation.depths.cam_right_wrist" not in info["features"]
    assert task == {"task_index": 0, "task": "first instruction"}
    assert stats["__cache_config"] == {"mode": "abs"}
    np.testing.assert_allclose(stats["statistics"]["action"]["mean"], expected_action.mean(0))
