"""Convert the released RoboTwin checkpoint to the current runtime layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def convert_state_dict(checkpoint: Path) -> dict:
    source = load_file(str(checkpoint), device="cpu")
    converted = {}
    unmapped = []

    for source_key, value in source.items():
        normalized_key = source_key[7:] if source_key.startswith("module.") else source_key
        target_key = TurboVLAFramework._legacy_rgb_checkpoint_key(normalized_key)
        if target_key is None:
            unmapped.append(source_key)
            continue
        wrapped_key = f"model.{target_key}"
        if wrapped_key in converted:
            raise RuntimeError(f"duplicate converted key: {wrapped_key}")
        converted[wrapped_key] = value.contiguous()

    if unmapped:
        raise RuntimeError(f"unmapped checkpoint keys ({len(unmapped)}): {unmapped[:20]}")
    if len(converted) != len(source):
        raise RuntimeError(f"converted {len(converted)} of {len(source)} tensors")
    return converted


def build_runtime_config(args: argparse.Namespace):
    # Parse the released file as a validity check even though the current runtime
    # uses the repository's equivalent, renamed schema.
    official = OmegaConf.load(args.official_config)
    if OmegaConf.select(official, "framework.name") != "GroundingDINODiT":
        raise ValueError("the released config does not describe GroundingDINODiT")

    config = OmegaConf.load(args.runtime_config)
    config.framework.text.bert_path = str(args.bert_path.resolve())
    config.framework.text.local_files_only = True
    config.framework.vision.model_path = str(args.dinov3_path.resolve())
    config.framework.vision.local_files_only = True
    config.framework.initialization.pretrained_ckpt = ""
    config.framework.initialization.load_pretrained = False
    config.datasets.vla_data.data_root_dir = "."
    config.run_root_dir = str(args.output_dir.resolve())
    config.run_id = "official_robotwin_55k_runtime_compat"
    config.output_dir = str(args.output_dir.resolve())
    return config


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.official_config,
        args.statistics,
        args.runtime_config,
        args.bert_path,
        args.dinov3_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    converted = convert_state_dict(args.checkpoint)
    output_checkpoint = args.output_dir / "steps_55000_ema_model.safetensors"
    save_file(converted, str(output_checkpoint))

    config = build_runtime_config(args)
    OmegaConf.save(config, args.output_dir / "config.yaml", resolve=True)
    shutil.copy2(args.statistics, args.output_dir / "dataset_statistics.json")

    print(f"converted_tensors={len(converted)}")
    print(f"checkpoint={output_checkpoint}")
    print(f"config={args.output_dir / 'config.yaml'}")
    print(f"statistics={args.output_dir / 'dataset_statistics.json'}")


if __name__ == "__main__":
    main()
