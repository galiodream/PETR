#!/usr/bin/env bash
set -euo pipefail

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-configs/default.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/train}
PYTHON_BIN=${PYTHON_BIN:-python}

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
