"""Dependency-light smoke test for the newly added ThreeDMix module."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "turbovla" / "models" / "three_dmix.py"
    spec = spec_from_file_location("turbovla_three_dmix_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.ThreeDMix(
        hidden_dim=256,
        vggt_dim=2048,
        semantic_pool="vl",
        output_scale_init=1.0,
    ).to(device)
    batch, n_visual, n_text, n_geo = 2, 32, 12, 64
    visual = torch.randn(batch, n_visual, 256, device=device)
    text = torch.randn(batch, n_text, 256, device=device)
    padding = torch.zeros(batch, n_text, dtype=torch.bool, device=device)
    padding[1, -2:] = True
    vggt = torch.randn(batch, n_geo, 2048, device=device)
    fused, gates = model(visual, text, padding, vggt)
    assert fused.shape == (batch, n_geo, 256), fused.shape
    assert gates.shape == fused.shape, gates.shape
    loss = fused.square().mean()
    loss.backward()
    assert model.vggt_projection.weight.grad is not None
    print(
        "THREEDMIX_SMOKE_OK",
        f"device={device}",
        f"torch={torch.__version__}",
        f"fused={tuple(fused.shape)}",
        f"gate_mean={gates.detach().mean().item():.4f}",
    )


if __name__ == "__main__":
    main()
