from pathlib import Path

import torch

from petr.config import load_config
from petr.models import build_model


def test_model_forward_shapes() -> None:
    cfg = load_config(str(Path("configs/tiny_cpu.yaml")))
    model = build_model(cfg)
    model.eval()

    bsz = 2
    num_views = int(cfg["data"]["num_views"])
    h, w = cfg["data"]["image_size"]

    x = torch.randn(bsz, num_views, 3, h, w)
    out = model(x)

    assert out["pred_logits"].shape == (
        bsz,
        int(cfg["model"]["num_queries"]),
        int(cfg["model"]["num_classes"]),
    )
    assert out["pred_boxes"].shape == (bsz, int(cfg["model"]["num_queries"]), 4)
    assert torch.all(out["pred_boxes"] >= 0)
    assert torch.all(out["pred_boxes"] <= 1)
