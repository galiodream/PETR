#!/usr/bin/env python3
"""PETR DDP Training Entry Point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

from petr.config import load_config
from petr.data import build_dataloaders
from petr.engine import cleanup_distributed, evaluate, init_distributed_mode, train_one_epoch
from petr.models import build_model
from petr.utils import is_main_process, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PETR DDP training")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--output-dir", default="outputs/train", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--use-hungarian", action="store_true", help="Use Hungarian matching loss")
    return parser.parse_args()


def _resolve_device(device_arg: str, local_rank: int) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    return torch.device(device_arg)


def _to_float_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        out[key] = float(value)
    return out


def main() -> None:
    args = parse_args()
    state = init_distributed_mode()

    try:
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg["train"]["epochs"] = int(args.epochs)
        if args.batch_size is not None:
            cfg["train"]["batch_size"] = int(args.batch_size)
        if args.seed is not None:
            cfg["data"]["seed"] = int(args.seed)

        if args.amp and args.no_amp:
            raise ValueError("--amp and --no-amp cannot both be set")

        use_amp = bool(cfg["train"].get("amp", True))
        if args.amp:
            use_amp = True
        if args.no_amp:
            use_amp = False

        # Check if using Hungarian matching
        use_hungarian = args.use_hungarian or cfg.get("model", {}).get("use_petr", False)

        device = _resolve_device(args.device, state.local_rank)
        set_seed(int(cfg["data"]["seed"]) + state.rank)

        output_dir = Path(args.output_dir)
        writer = None
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle, indent=2)
            if SummaryWriter is not None:
                writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
            else:
                print("TensorBoard is unavailable (missing tensorboard package); skipping TB logging.", flush=True)

        train_loader, val_loader, train_sampler = build_dataloaders(cfg, distributed=state.distributed)

        model = build_model(cfg).to(device)
        model_without_ddp = model

        if state.distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[state.local_rank], output_device=state.local_rank)
            else:
                model = DDP(model)
            model_without_ddp = model.module

        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )

        scaler = None
        amp_enabled = use_amp and device.type == "cuda"
        if amp_enabled:
            scaler = GradScaler(enabled=True)

        start_epoch = 0
        if args.resume:
            start_epoch, _ = load_checkpoint(
                args.resume,
                model_without_ddp,
                optimizer=optimizer,
                scaler=scaler,
                map_location=device,
            )
            start_epoch += 1

        epochs = int(cfg["train"]["epochs"])
        eval_interval = int(cfg["train"]["eval_interval"])
        save_interval = int(cfg["train"]["save_interval"])

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_classes=int(cfg["model"]["num_classes"]),
                bbox_weight=float(cfg["train"]["bbox_loss_weight"]),
                max_norm=float(cfg["train"]["clip_max_norm"]),
                log_interval=int(cfg["train"]["log_interval"]),
                scaler=scaler,
                use_amp=amp_enabled,
                use_hungarian=use_hungarian,
            )

            val_stats = {}
            if (epoch + 1) % eval_interval == 0:
                val_stats = evaluate(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    num_classes=int(cfg["model"]["num_classes"]),
                    bbox_weight=float(cfg["train"]["bbox_loss_weight"]),
                    use_amp=amp_enabled,
                    use_hungarian=use_hungarian,
                )

            if is_main_process():
                train_float = _to_float_metrics(train_stats)
                val_float = _to_float_metrics(val_stats)
                metrics = {
                    "epoch": epoch,
                    "train": train_float,
                    "val": val_float,
                }
                print(json.dumps(metrics, ensure_ascii=True), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
                if writer is not None:
                    for key, value in train_float.items():
                        writer.add_scalar(f"train/{key}", value, epoch)
                    for key, value in val_float.items():
                        writer.add_scalar(f"val/{key}", value, epoch)
                    writer.flush()

                should_save = ((epoch + 1) % save_interval == 0) or (epoch + 1 == epochs)
                if should_save:
                    save_checkpoint(
                        str(output_dir / f"checkpoint_epoch_{epoch:04d}.pth"),
                        model_without_ddp,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        cfg=cfg,
                    )

        if writer is not None:
            writer.close()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()