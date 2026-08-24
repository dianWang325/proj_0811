#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

[[ -f /.dockerenv ]] || accuracy_fail "run this script inside container wd_test0811"
for command_name in python ais_bench vllm npu-smi curl git timeout stat; do
    accuracy_require_command "${command_name}"
done

python - <<'PY'
import importlib

required = ("ais_bench", "vllm", "vllm_ascend", "torch", "torch_npu", "numpy", "scipy")
for name in required:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')} ({getattr(module, '__file__', '')})")

import numpy
parts = tuple(int(part) for part in numpy.__version__.split(".")[:2])
if parts >= (2, 3):
    raise SystemExit(
        "DEPENDENCY_ERROR: NumPy must be <2.3 for the installed SciPy; "
        "do not continue until the container dependency is corrected"
    )
PY

accuracy_validate_bool ACCURACY_CPP_ENABLED "${ACCURACY_CPP_ENABLED}"
accuracy_validate_runner "${ACCURACY_RUNNER}"
accuracy_validate_topology

timeout 15 stat "${ACCURACY_MODEL_PATH}/config.json" >/dev/null || \
    accuracy_fail "model config is unavailable or the mount timed out: ${ACCURACY_MODEL_PATH}/config.json"

if accuracy_is_port_open "${ACCURACY_PORT}"; then
    accuracy_fail "port ${ACCURACY_PORT} is already in use"
fi

IFS=',' read -r -a devices <<<"${ACCURACY_NPU_DEVICES}"
for device in "${devices[@]}"; do
    accuracy_validate_device "${device}"
done

"${ACCURACY_ROOT}/scripts/01_check_datasets.sh" all
accuracy_info "environment, model, NPU topology, port and datasets passed preflight"
