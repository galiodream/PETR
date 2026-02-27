import copy
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "num_queries": 50,
        "num_classes": 10,
        "feature_size": [16, 16],
        "backbone_channels": 128,
    },
    "data": {
        "image_size": [128, 128],
        "num_views": 6,
        "train_samples": 200,
        "val_samples": 50,
        "num_workers": 2,
        "seed": 42,
    },
    "train": {
        "batch_size": 4,
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "log_interval": 20,
        "eval_interval": 1,
        "save_interval": 1,
        "bbox_loss_weight": 5.0,
        "amp": True,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    return _deep_update(cfg, data)


def update_config(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_update(cfg, overrides)
