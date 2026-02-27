#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from petr.config import load_config
from petr.data import build_dataloaders
from petr.engine import cleanup_distributed, init_distributed_mode, run_inference
from petr.models import build_model
from petr.utils import is_main_process, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PETR-lite DDP inference")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", default="outputs/infer/predictions.json", type=str)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _resolve_device(device_arg: str, local_rank: int) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    state = init_distributed_mode()

    try:
        cfg = load_config(args.config)
        if args.batch_size is not None:
            cfg["train"]["batch_size"] = int(args.batch_size)

        if args.amp and args.no_amp:
            raise ValueError("--amp and --no-amp cannot both be set")

        use_amp = bool(cfg["train"].get("amp", True))
        if args.amp:
            use_amp = True
        if args.no_amp:
            use_amp = False

        device = _resolve_device(args.device, state.local_rank)
        amp_enabled = use_amp and device.type == "cuda"

        set_seed(int(cfg["data"]["seed"]) + state.rank)

        _, val_loader, _ = build_dataloaders(cfg, distributed=state.distributed)
        model = build_model(cfg).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)

        if state.distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[state.local_rank], output_device=state.local_rank)
            else:
                model = DDP(model)

        metrics = run_inference(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=amp_enabled,
            max_samples=args.max_samples,
            output_path=args.output,
        )

        if is_main_process():
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with (out_file.parent / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            print(json.dumps(metrics, ensure_ascii=True), flush=True)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
