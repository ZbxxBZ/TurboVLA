"""Build a RoboTwin policy dataset and inspect one transformed training sample."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def _describe(value, prefix: str = "sample") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _describe(item, f"{prefix}.{key}")
        return
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(f"{prefix}: shape={tuple(value.shape)} dtype={value.dtype}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        print(f"{prefix}: {type(value).__name__}[{len(value)}]")
        if value:
            _describe(value[0], f"{prefix}[0]")
        return
    print(f"{prefix}: {type(value).__name__}={value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/robotwin/configs/clean50_depth.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    data_config = config.datasets.vla_data
    data_config.data_root_dir = str(args.data_root.resolve())
    modality_path = Path(str(data_config.modality_metadata_path))
    data_config.modality_metadata_path = str(modality_path.resolve())
    data_config.num_workers = 0

    dataset = get_vla_dataset(data_cfg=data_config, seed=int(config.seed))
    total_trajectories = sum(len(item.trajectory_ids) for item in dataset.datasets)
    total_frames = sum(int(item.trajectory_lengths.sum()) for item in dataset.datasets)
    print(
        f"datasets={len(dataset.datasets)} trajectories={total_trajectories} "
        f"frames={total_frames} epoch_samples={len(dataset)}"
    )
    if len(dataset.datasets) != 50 or total_trajectories != 360 or total_frames != 65515:
        raise AssertionError("converted clean50 dataset counts do not match the collection manifest")

    sample = dataset[args.index]
    _describe(sample)


if __name__ == "__main__":
    main()
