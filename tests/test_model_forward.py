import copy
from pathlib import Path

import torch
import pytest

from petr.config import load_config
from petr.models import build_model
from petr.data.synthetic import generate_camera_extrinsics, generate_camera_intrinsics


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


def test_petr_requires_camera_tensors_when_3d_pe_enabled() -> None:
    cfg = load_config(str(Path("configs/tiny_cpu.yaml")))
    cfg = copy.deepcopy(cfg)
    cfg["model"]["use_petr"] = True
    cfg["model"]["with_3d_pe"] = True

    model = build_model(cfg)
    model.eval()

    bsz = 2
    num_views = int(cfg["data"]["num_views"])
    h, w = cfg["data"]["image_size"]
    x = torch.randn(bsz, num_views, 3, h, w)

    with pytest.raises(ValueError):
        model(x)

    cam_intrinsics = generate_camera_intrinsics(bsz, num_views, (h, w), dtype=x.dtype)
    cam_extrinsics = generate_camera_extrinsics(bsz, num_views, dtype=x.dtype)
    out = model(x, cam_intrinsics=cam_intrinsics, cam_extrinsics=cam_extrinsics)

    assert out["pred_logits"].shape == (
        bsz,
        int(cfg["model"]["num_queries"]),
        int(cfg["model"]["num_classes"]),
    )
    assert out["pred_boxes"].shape == (bsz, int(cfg["model"]["num_queries"]), 8)


def test_petr_without_3d_pe_runs_without_camera_tensors() -> None:
    cfg = load_config(str(Path("configs/tiny_cpu.yaml")))
    cfg = copy.deepcopy(cfg)
    cfg["model"]["use_petr"] = True
    cfg["model"]["with_3d_pe"] = False

    model = build_model(cfg)
    model.eval()

    bsz = 2
    num_views = int(cfg["data"]["num_views"])
    h, w = cfg["data"]["image_size"]
    x = torch.randn(bsz, num_views, 3, h, w)
    out = model(x)

    assert out["pred_boxes"].shape == (bsz, int(cfg["model"]["num_queries"]), 8)
