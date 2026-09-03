"""One-sample online-VGGT TurboVLA/3D-MIX forward/backward smoke test."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path("/root/TurboVLA_3Dmix")
DATA = Path(
    "/root/local_data/robotwin_lerobot_clean50_360/Clean/click_alarmclock/"
    "data/chunk-000/episode_000000.parquet"
)

import sys

sys.path.insert(0, "/mnt/vggt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/starvla_runtime"))

from starVLA.model.framework.base_framework import build_framework  # noqa: E402


def decode(value):
    if isinstance(value, dict):
        value = value["bytes"]
    return Image.open(BytesIO(value)).convert("RGB")


def main():
    cfg = OmegaConf.load(REPO / "experiments/robotwin/configs/clean50.yaml")
    cfg.framework.text.bert_path = "/root/local_models/bert-base-uncased"
    cfg.framework.text.attn_implementation = "sdpa"
    cfg.framework.vision.model_path = "/root/local_models/dinov3-vitl16-hf"
    cfg.framework.vision.attn_implementation = "sdpa"
    cfg.framework.vision.freeze_vision_encoder = True
    cfg.framework.initialization.pretrained_ckpt = "/root/local_models/turbovla_init.pt"
    cfg.framework.three_dmix.enabled = True
    cfg.framework.three_dmix.online = True
    cfg.framework.three_dmix.vggt_model_path = "/root/local_models/vggt-1b/model.pt"
    cfg.framework.three_dmix.vggt_code_path = "/mnt/vggt"
    cfg.framework.three_dmix.vggt_input_size = 518
    cfg.trainer.use_deepspeed = False
    model = build_framework(cfg).to("cuda")
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

    row = pd.read_parquet(DATA).iloc[0]
    cols = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    images = [decode(row[col]) for col in cols]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)
    example = {
        "image": images,
        "lang": "Use the right arm to click the center button on the black alarm-clock",
        "state": state,
        "action": np.stack([action] * 50, axis=0),
    }
    torch.cuda.reset_peak_memory_stats()
    output = model([example])
    loss = output["action_loss"]
    loss.backward()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    state_keys = list(model.state_dict().keys())
    assert not any(name.startswith("_online_vggt") for name in state_keys)
    assert any(name.startswith("model.three_dmix") for name in trainable)
    assert any(name.startswith("model.action_head") for name in trainable)
    print(
        "TURBOVLA_ONLINE_SMOKE_OK",
        "loss=", float(loss.detach()),
        "trainable=", len(trainable),
        "state_keys=", len(state_keys),
        "peak_vram_mb=", round(torch.cuda.max_memory_allocated() / 1024**2),
        flush=True,
    )


if __name__ == "__main__":
    main()
