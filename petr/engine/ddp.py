import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from petr.utils.logging import setup_logger


@dataclass
class DistributedState:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str


logger = setup_logger("petr.ddp")


def init_distributed_mode() -> DistributedState:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

    if distributed:
        if has_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()
        logger.info(
            "DDP initialized | backend=%s rank=%s world_size=%s local_rank=%s",
            backend,
            rank,
            world_size,
            local_rank,
        )
    else:
        logger.info("Running in single-process mode")

    return DistributedState(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
