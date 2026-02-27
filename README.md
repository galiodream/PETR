# PETR Lite Multi-Node DDP Scaffold

This repository provides a PETR-style multi-view detection scaffold with single-process, single-node multi-GPU, and multi-node multi-GPU DDP support for both training and inference.

## Project Layout

- `petr/models/petr.py`: PETR-lite model (multi-view backbone + transformer encoder/decoder + query heads)
- `petr/data/synthetic.py`: synthetic multi-view dataset and distributed dataloader builder
- `petr/engine/trainer.py`: training and evaluation loops
- `petr/engine/inference.py`: inference loop with metric reduction and JSON output
- `tools/train.py`: DDP-aware training entry
- `tools/infer.py`: DDP-aware inference entry
- `scripts/launch_train_ddp.sh`: `torchrun` wrapper for train
- `scripts/launch_infer_ddp.sh`: `torchrun` wrapper for infer

## Conda Environment Setup

CPU env:

```bash
bash scripts/setup_conda_env.sh environment.cpu.yml
```

CUDA 11.8 env:

```bash
bash scripts/setup_conda_env.sh environment.cuda118.yml
```

## Training

Single process:

```bash
python tools/train.py --config configs/default.yaml --output-dir outputs/train
```

Single node multi-GPU:

```bash
NPROC_PER_NODE=8 CONFIG=configs/default.yaml OUTPUT_DIR=outputs/train bash scripts/launch_train_ddp.sh
```

Multi-node multi-GPU (example 2 nodes):

```bash
# node 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8 bash scripts/launch_train_ddp.sh

# node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8 bash scripts/launch_train_ddp.sh
```

## Inference

Single process:

```bash
python tools/infer.py --config configs/default.yaml --checkpoint outputs/train/checkpoint_epoch_0009.pth --output outputs/infer/predictions.json
```

DDP inference:

```bash
NPROC_PER_NODE=8 CONFIG=configs/default.yaml OUTPUT=outputs/infer/predictions.json bash scripts/launch_infer_ddp.sh outputs/train/checkpoint_epoch_0009.pth
```

## Validation

Run all checks:

```bash
bash scripts/run_all_tests.sh
```

This runs:

1. 1-epoch CPU training
2. checkpoint-based inference
3. pytest unit/integration tests
4. CPU 2-process `torchrun` DDP smoke
