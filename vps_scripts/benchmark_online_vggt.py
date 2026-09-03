"""Benchmark online VGGT extraction plus TurboVLA/3D-MIX training.

This is a measurement-only script.  It does not change the production trainer
or write a feature cache.  It reuses one real RoboTwin observation and reports
warm VGGT latency, combined forward/backward latency, and peak VRAM.
"""

from __future__ import annotations

import gc
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

REPO = Path("/root/TurboVLA_3Dmix")
DATA = Path("/root/local_data/robotwin_lerobot_clean50_360/Clean/click_alarmclock/data/chunk-000/episode_000000.parquet")
VGGT_WEIGHTS = Path("/root/local_models/vggt-1b/model.pt")
BERT = "/root/local_models/bert-base-uncased"
DINO = "/root/local_models/dinov3-vitl16-hf"
INIT = "/root/local_models/turbovla_init.pt"

sys.path.insert(0, "/mnt/vggt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/starvla_runtime"))

from vggt.models.vggt import VGGT  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402


COLS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def decode(value):
    if isinstance(value, dict):
        value = value["bytes"]
    return Image.open(BytesIO(value)).convert("RGB")


def vggt_images(row, batch_size: int) -> torch.Tensor:
    images = torch.stack(
        [pil_to_tensor(decode(row[c]).resize((518, 518))).float() / 255.0 for c in COLS]
    )
    return images.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()


def vggt_features(model, images):
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aggregated, patch_start = model.aggregator(images)
    # Keep the exact native patch-token layout used by the current 3D-MIX code.
    return aggregated[-1][:, :, patch_start:, :].float().flatten(1, 2)


def make_framework():
    cfg = OmegaConf.load(REPO / "experiments/robotwin/configs/clean50.yaml")
    cfg.framework.text.bert_path = BERT
    cfg.framework.text.attn_implementation = "sdpa"
    cfg.framework.vision.model_path = DINO
    cfg.framework.vision.attn_implementation = "sdpa"
    cfg.framework.vision.freeze_vision_encoder = True
    cfg.framework.initialization.pretrained_ckpt = INIT
    cfg.framework.three_dmix.enabled = True
    cfg.framework.three_dmix.output_scale_init = 0.0
    cfg.trainer.use_deepspeed = False
    model = build_framework(cfg).to("cuda")
    # Match scripts/robotwin/train_3dmix.sh exactly.
    for module_name in (
        "text_encoder",
        "vision_encoder",
        "vision_projection",
        "vision_language_interaction",
    ):
        module = model
        for attr in module_name.split("."):
            module = getattr(module, attr)
        for param in module.parameters():
            param.requires_grad_(False)
    model.train()
    return model


def make_examples(row, batch_size: int, features: torch.Tensor):
    state = np.asarray(row["observation.state"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)
    images = [decode(row[c]) for c in COLS]
    examples = []
    for index in range(batch_size):
        examples.append(
            {
                "image": images,
                "lang": "Use the right arm to click the center button on the black alarm-clock",
                "state": state,
                "action": np.stack([action] * 50, axis=0),
                # The framework adapter currently normalizes tensor features
                # through CPU before batching, just as a cache path would.
                "vggt": features[index].cpu(),
            }
        )
    return examples


def sync():
    torch.cuda.synchronize()


def main():
    torch.cuda.reset_peak_memory_stats()
    row = pd.read_parquet(DATA).iloc[0]
    device = torch.device("cuda")

    print("loading VGGT", flush=True)
    vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False, feature_only=True)
    state = torch.load(VGGT_WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = vggt.load_state_dict(state, strict=False)
    print("VGGT_LOAD", len(missing), len(unexpected), flush=True)
    vggt.eval().to(device)

    for batch_size in (1, 2, 4):
        images = vggt_images(row, batch_size).to(device)
        # Warmup (model is already loaded, so this measures inference only).
        _ = vggt_features(vggt, images)
        sync()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            feats = vggt_features(vggt, images)
            sync()
            times.append(time.perf_counter() - start)
        print(
            f"VGGT_BATCH={batch_size} features={tuple(feats.shape)} "
            f"warm_seconds={[round(x, 3) for x in times]} "
            f"mean_seconds={sum(times)/len(times):.3f} "
            f"peak_vram_mb={torch.cuda.max_memory_allocated()/1024**2:.0f}",
            flush=True,
        )

    print("loading TurboVLA", flush=True)
    model = make_framework()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    torch.cuda.reset_peak_memory_stats()

    for batch_size in (1, 2, 4):
        images = vggt_images(row, batch_size).to(device)
        feats = vggt_features(vggt, images)
        examples = make_examples(row, batch_size, feats)
        # One warmup update.
        optimizer.zero_grad(set_to_none=True)
        out = model(examples)
        out["action_loss"].backward()
        optimizer.step()
        sync()

        times = []
        for _ in range(3):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            feats = vggt_features(vggt, images)
            examples = make_examples(row, batch_size, feats)
            out = model(examples)
            out["action_loss"].backward()
            optimizer.step()
            sync()
            times.append(time.perf_counter() - start)
        print(
            f"COMBINED_BATCH={batch_size} seconds={[round(x, 3) for x in times]} "
            f"mean_seconds={sum(times)/len(times):.3f} "
            f"peak_vram_mb={torch.cuda.max_memory_allocated()/1024**2:.0f} "
            f"loss={float(out['action_loss'].detach()):.6f}",
            flush=True,
        )
        del images, feats, examples, out
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
