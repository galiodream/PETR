"""
PETR Training Engine.

Supports both simple direct loss computation and Hungarian matching loss.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

from petr.utils.distributed import all_reduce_dict, get_world_size, is_main_process


def _compute_losses_simple(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    num_classes: int,
    bbox_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Simple loss computation without Hungarian matching.
    Used for PETRLite model.

    Args:
        outputs: Model outputs with pred_logits and pred_boxes
        targets: Ground truth with labels and boxes
        num_classes: Number of classes (unused, kept for compatibility)
        bbox_weight: Weight for bbox loss

    Returns:
        loss: Total loss
        loss_dict: Dictionary of individual losses
    """
    logits = outputs["pred_logits"].flatten(0, 1)
    labels = targets["labels"].flatten(0, 1)
    boxes = targets["boxes"]

    loss_cls = F.cross_entropy(logits, labels, reduction="mean")
    loss_bbox = F.l1_loss(outputs["pred_boxes"], boxes, reduction="mean")
    loss = loss_cls + bbox_weight * loss_bbox

    return loss, {"loss": loss, "loss_cls": loss_cls, "loss_bbox": loss_bbox}


def _compute_losses_hungarian(
    model: torch.nn.Module,
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Loss computation with Hungarian matching.
    Used for full PETR model.

    Args:
        model: PETR model with compute_loss method
        outputs: Model outputs with pred_logits and pred_boxes
        targets: Ground truth with labels and boxes
        bbox_weight: Weight for bbox L1 loss
        giou_weight: Weight for GIoU loss

    Returns:
        loss: Total loss
        loss_dict: Dictionary of individual losses
    """
    if hasattr(model, 'compute_loss'):
        # Full PETR model with Hungarian matching
        loss, loss_dict = model.compute_loss(outputs, targets)
        # Scale losses
        loss_dict["loss_bbox"] = loss_dict.get("loss_bbox", torch.tensor(0.0)) * bbox_weight
        loss_dict["loss_giou"] = loss_dict.get("loss_giou", torch.tensor(0.0)) * giou_weight
        loss = loss_dict["loss_ce"] + loss_dict["loss_bbox"] + loss_dict["loss_giou"]
        return loss, loss_dict
    else:
        # Fallback to simple loss
        return _compute_losses_simple(outputs, targets, 10, bbox_weight)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_classes: int,
    bbox_weight: float,
    max_norm: float = 0.0,
    log_interval: int = 20,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    use_hungarian: bool = False,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: PETR model
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        num_classes: Number of classes
        bbox_weight: Weight for bbox loss
        max_norm: Maximum gradient norm for clipping
        log_interval: Logging interval
        scaler: Gradient scaler for AMP
        use_amp: Whether to use automatic mixed precision
        use_hungarian: Whether to use Hungarian matching loss

    Returns:
        Dictionary of average losses
    """
    model.train()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    num_steps = 0
    total_samples = 0.0
    start_time = time.perf_counter()

    for step, batch in enumerate(data_loader):
        images = batch["images"].to(device, non_blocking=True)
        targets = {
            "boxes": batch["boxes"].to(device, non_blocking=True),
            "labels": batch["labels"].to(device, non_blocking=True),
        }

        # Add camera parameters if available (for 3D PE)
        if "cam_intrinsics" in batch and "cam_extrinsics" in batch:
            targets["cam_intrinsics"] = batch["cam_intrinsics"].to(device, non_blocking=True)
            targets["cam_extrinsics"] = batch["cam_extrinsics"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Forward pass with camera parameters if available
            if "cam_intrinsics" in targets:
                outputs = model(
                    images,
                    cam_intrinsics=targets.get("cam_intrinsics"),
                    cam_extrinsics=targets.get("cam_extrinsics"),
                )
            else:
                outputs = model(images)

            # Compute loss
            if use_hungarian and hasattr(model, 'compute_loss'):
                loss, loss_dict = _compute_losses_hungarian(
                    model, outputs, targets, bbox_weight
                )
            else:
                loss, loss_dict = _compute_losses_simple(
                    outputs, targets, num_classes, bbox_weight
                )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        reduced = all_reduce_dict({k: v.detach() for k, v in loss_dict.items() if k in running})
        total_samples += float(images.shape[0] * get_world_size())
        for key in running:
            if key in reduced:
                running[key] += reduced[key].item()
        num_steps += 1

        if is_main_process() and (step % log_interval == 0):
            loss_str = f"loss={reduced.get('loss', torch.tensor(0)):.4f} "
            loss_str += f"cls={reduced.get('loss_cls', reduced.get('loss_ce', torch.tensor(0))):.4f} "
            loss_str += f"bbox={reduced.get('loss_bbox', torch.tensor(0)):.4f}"
            if 'loss_giou' in reduced:
                loss_str += f" giou={reduced['loss_giou']:.4f}"
            print(f"epoch={epoch} step={step} {loss_str}", flush=True)

    epoch_time = max(time.perf_counter() - start_time, 1e-8)
    stats = {key: value / max(num_steps, 1) for key, value in running.items()}
    stats["epoch_time_sec"] = epoch_time
    stats["samples_per_sec"] = total_samples / epoch_time
    stats["steps_per_sec"] = num_steps / epoch_time
    return stats


def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    num_classes: int,
    bbox_weight: float,
    use_amp: bool = True,
    use_hungarian: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: PETR model
        data_loader: Validation data loader
        device: Device to use
        num_classes: Number of classes
        bbox_weight: Weight for bbox loss
        use_amp: Whether to use automatic mixed precision
        use_hungarian: Whether to use Hungarian matching loss

    Returns:
        Dictionary of average losses
    """
    model.eval()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    num_steps = 0
    total_samples = 0.0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in data_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = {
                "boxes": batch["boxes"].to(device, non_blocking=True),
                "labels": batch["labels"].to(device, non_blocking=True),
            }

            # Add camera parameters if available
            if "cam_intrinsics" in batch and "cam_extrinsics" in batch:
                targets["cam_intrinsics"] = batch["cam_intrinsics"].to(device, non_blocking=True)
                targets["cam_extrinsics"] = batch["cam_extrinsics"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                # Forward pass
                if "cam_intrinsics" in targets:
                    outputs = model(
                        images,
                        cam_intrinsics=targets.get("cam_intrinsics"),
                        cam_extrinsics=targets.get("cam_extrinsics"),
                    )
                else:
                    outputs = model(images)

                # Compute loss
                if use_hungarian and hasattr(model, 'compute_loss'):
                    _, loss_dict = _compute_losses_hungarian(
                        model, outputs, targets, bbox_weight
                    )
                else:
                    _, loss_dict = _compute_losses_simple(
                        outputs, targets, num_classes, bbox_weight
                    )

            reduced = all_reduce_dict({k: v.detach() for k, v in loss_dict.items() if k in running})
            total_samples += float(images.shape[0] * get_world_size())
            for key in running:
                if key in reduced:
                    running[key] += reduced[key].item()
            num_steps += 1

    eval_time = max(time.perf_counter() - start_time, 1e-8)
    stats = {key: value / max(num_steps, 1) for key, value in running.items()}
    stats["eval_time_sec"] = eval_time
    stats["samples_per_sec"] = total_samples / eval_time
    stats["steps_per_sec"] = num_steps / eval_time
    return stats
