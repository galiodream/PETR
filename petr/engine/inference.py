from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.cuda.amp import autocast

from petr.utils.distributed import all_reduce_dict, is_main_process


def run_inference(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    use_amp: bool = True,
    max_samples: int | None = None,
    output_path: str | None = None,
) -> Dict[str, float]:
    model.eval()
    collected: List[Dict[str, list]] = []
    total = 0
    l1_sum = 0.0
    num_boxes = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["boxes"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(images)
            preds = outputs["pred_boxes"].detach()
            l1 = torch.abs(preds - targets).mean()
            l1_sum += l1.item()
            num_boxes += 1

            if is_main_process():
                for idx in range(preds.shape[0]):
                    collected.append(
                        {
                            "pred_boxes": preds[idx].cpu().tolist(),
                            "gt_boxes": targets[idx].cpu().tolist(),
                        }
                    )
            total += preds.shape[0]
            if max_samples is not None and total >= max_samples:
                break

    metrics = all_reduce_dict(
        {
            "l1": torch.tensor(l1_sum, device=device, dtype=torch.float32),
            "count": torch.tensor(num_boxes, device=device, dtype=torch.float32),
        }
    )
    mean_l1 = metrics["l1"].item() / max(metrics["count"].item(), 1.0)

    if output_path and is_main_process():
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"samples": collected, "mean_l1": mean_l1}, handle, indent=2)

    return {"mean_l1": mean_l1}
