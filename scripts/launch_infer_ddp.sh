#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint> [extra infer args]" >&2
  exit 1
fi

CHECKPOINT=$1
shift

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-configs/default.yaml}
OUTPUT=${OUTPUT:-outputs/infer/predictions.json}
PYTHON_BIN=${PYTHON_BIN:-python}

# DDP shell scripts
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/infer.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  "$@"
