import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch.distributed.run") is None, reason="torch.distributed.run is not available")
def test_torchrun_ddp_cpu_train_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "ddp_train"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "tools/train.py",
        "--config",
        "configs/tiny_cpu.yaml",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--no-amp",
    ]

    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            "DDP smoke failed:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    assert list(output_dir.glob("checkpoint_epoch_*.pth")), "No checkpoint from DDP train"
