import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def test_single_process_train_then_infer(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    train_out = tmp_path / "train"
    infer_out = tmp_path / "infer" / "predictions.json"

    train_cmd = [
        sys.executable,
        "tools/train.py",
        "--config",
        "configs/tiny_cpu.yaml",
        "--output-dir",
        str(train_out),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--no-amp",
    ]
    _run(train_cmd, repo)

    ckpts = sorted(train_out.glob("checkpoint_epoch_*.pth"))
    assert ckpts, "No checkpoint produced by training"

    infer_cmd = [
        sys.executable,
        "tools/infer.py",
        "--config",
        "configs/tiny_cpu.yaml",
        "--checkpoint",
        str(ckpts[-1]),
        "--output",
        str(infer_out),
        "--device",
        "cpu",
        "--no-amp",
    ]
    _run(infer_cmd, repo)

    assert infer_out.exists(), "Inference output JSON is missing"
    payload = json.loads(infer_out.read_text(encoding="utf-8"))
    assert "mean_l1" in payload
    assert "samples" in payload
