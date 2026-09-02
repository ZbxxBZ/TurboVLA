import io
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.robotwin.attach_depth_to_lerobot import _augment_parquet
from scripts.robotwin.train_depth_dinov3 import RoboTwinHeadRGBD, episode_paths, task_name
from starVLA.dataloader.gr00t_lerobot.depth_io import decode_depth_entry


def test_depth_modality_schema():
    with open("experiments/robotwin/configs/modality_depth.json", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["depth"]["cam_high"]["original_key"] == "observation.depths.cam_high"


def test_uint16_png_depth_decode(tmp_path):
    expected = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
    buffer = io.BytesIO()
    Image.fromarray(expected).save(buffer, format="PNG")

    decoded = decode_depth_entry(buffer.getvalue(), tmp_path)

    np.testing.assert_array_equal(decoded, expected)


def test_attach_depth_to_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "episode0.hdf5"
    expected = np.stack(
        [np.full((4, 5), 1000, dtype=np.uint16), np.full((4, 5), 1500, dtype=np.uint16)]
    )
    states = np.arange(28, dtype=np.float32).reshape(2, 14)
    with h5py.File(raw_path, "w") as handle:
        for camera in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(f"observation/{camera}/depth", data=expected)
        handle.create_dataset("joint_action/vector", data=states)

    parquet_path = tmp_path / "episode_000000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": list(states),
        }
    ).to_parquet(parquet_path)

    height, width, frame_count = _augment_parquet(parquet_path, raw_dir)
    converted = pd.read_parquet(parquet_path)
    decoded = decode_depth_entry(
        converted["observation.depths.cam_high"].iloc[1],
        tmp_path,
    )

    assert (height, width, frame_count) == (4, 5, 2)
    np.testing.assert_array_equal(decoded, expected[1])


def test_attach_depth_rejects_mismatched_rgb_trajectory(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "episode0.hdf5"
    with h5py.File(raw_path, "w") as handle:
        for camera in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(
                f"observation/{camera}/depth",
                data=np.full((2, 4, 5), 1000, dtype=np.uint16),
            )
        handle.create_dataset("joint_action/vector", data=np.zeros((2, 14), dtype=np.float32))

    parquet_path = tmp_path / "episode_000000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": [np.zeros(14), np.ones(14)],
        }
    ).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="RGB/depth trajectory mismatch"):
        _augment_parquet(parquet_path, raw_dir)


def test_stage_one_reads_embedded_lerobot_rgbd(tmp_path):
    task_root = tmp_path / "Clean" / "click_alarmclock"
    parquet_dir = task_root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)

    rgb_buffer = io.BytesIO()
    Image.fromarray(np.full((6, 8, 3), (64, 128, 192), dtype=np.uint8)).save(rgb_buffer, format="JPEG")
    depth_buffer = io.BytesIO()
    Image.fromarray(np.full((6, 8), 1250, dtype=np.uint16)).save(depth_buffer, format="PNG")
    parquet_path = parquet_dir / "episode_000000.parquet"
    pd.DataFrame(
        {
            "observation.images.cam_high": [{"bytes": rgb_buffer.getvalue(), "path": None}],
            "observation.depths.cam_high": [{"bytes": depth_buffer.getvalue(), "path": None}],
        }
    ).to_parquet(parquet_path, index=False)

    paths = episode_paths(tmp_path)
    dataset = RoboTwinHeadRGBD(paths, image_size=4, frame_stride=1, augment=False)
    rgb, depth = dataset[0]

    assert paths == [parquet_path]
    assert task_name(parquet_path) == "click_alarmclock"
    assert rgb.shape == (3, 4, 4)
    assert depth.shape == (1, 4, 4)
    assert torch.allclose(depth, torch.full_like(depth, 1.25))
