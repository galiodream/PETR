#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
WORKDIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${WORKDIR}"

OUT_ROOT=${OUT_ROOT:-/tmp/petr_ddp_test}
TRAIN_OUT=${OUT_ROOT}/train
INFER_OUT=${OUT_ROOT}/infer/predictions.json

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

${PYTHON_BIN} tools/train.py \
  --config configs/tiny_cpu.yaml \
  --output-dir "${TRAIN_OUT}" \
  --epochs 1 \
  --device cpu \
  --no-amp

CKPT=$(ls -1 "${TRAIN_OUT}"/checkpoint_epoch_*.pth | sort | tail -n 1)

${PYTHON_BIN} tools/infer.py \
  --config configs/tiny_cpu.yaml \
  --checkpoint "${CKPT}" \
  --output "${INFER_OUT}" \
  --device cpu \
  --no-amp

${PYTHON_BIN} -m pytest -q tests

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  tools/train.py \
  --config configs/tiny_cpu.yaml \
  --output-dir "${OUT_ROOT}/ddp_train" \
  --epochs 1 \
  --device cpu \
  --no-amp

echo "All checks passed"
