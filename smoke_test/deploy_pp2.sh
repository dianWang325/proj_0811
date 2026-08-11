#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORK_DIR="/home/w00985415"
readonly TEST_DIR="${WORK_DIR}/proj_0811/smoke_test"
readonly MODEL_PATH="/mnt/a800_weight/Qwen3-30B-A3B-W8A8"
readonly SERVED_MODEL_NAME="qwen3-30b-a3b-w8a8"
readonly NPU_DEVICES="12,13"

cd "${WORK_DIR}"

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
    echo "Model configuration is not readable: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export VLLM_USE_V1=1

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${SMOKE_TEST_PORT:-18080}" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --max-model-len 512 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager
