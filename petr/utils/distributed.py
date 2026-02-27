from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def all_reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not is_dist_avail_and_initialized():
        return input_dict
    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys])
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= get_world_size()
        return {key: value for key, value in zip(keys, values)}
