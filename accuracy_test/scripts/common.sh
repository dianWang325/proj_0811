#!/usr/bin/env bash

readonly ACCURACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${ACCURACY_ROOT}/.." && pwd)"
readonly ACCURACY_CONFIG_FILE="${ACCURACY_ROOT}/configs/common.env"
readonly ACCURACY_MODEL_PROFILE_FILE="${ACCURACY_MODEL_PROFILE_FILE:-${ACCURACY_ROOT}/configs/models/deepseek_v4_flash.env}"
readonly ACCURACY_DATASET_ROOT="${PROJECT_ROOT}/predict/ais_bench/datasets"

accuracy_fail() {
    echo "ACCURACY_TEST_FAIL: $*" >&2
    return 1
}

accuracy_info() {
    echo "ACCURACY_TEST_INFO: $*"
}

accuracy_load_config() {
    [[ -r "${ACCURACY_CONFIG_FILE}" ]] || accuracy_fail "missing config: ${ACCURACY_CONFIG_FILE}"
    [[ -r "${ACCURACY_MODEL_PROFILE_FILE}" ]] || accuracy_fail "missing model profile: ${ACCURACY_MODEL_PROFILE_FILE}"
    set -a
    # shellcheck source=/dev/null
    source "${ACCURACY_CONFIG_FILE}"
    # shellcheck source=/dev/null
    source "${ACCURACY_MODEL_PROFILE_FILE}"
    set +a
}

accuracy_require_command() {
    command -v "$1" >/dev/null 2>&1 || accuracy_fail "required command is unavailable: $1"
}

accuracy_is_port_open() {
    python - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

accuracy_validate_bool() {
    local name="$1" value="$2"
    [[ "${value}" == "0" || "${value}" == "1" ]] || accuracy_fail "${name} must be 0 or 1"
}

accuracy_validate_runner() {
    [[ "$1" == "mrv1" || "$1" == "mrv2" ]] || accuracy_fail "runner must be mrv1 or mrv2"
}

accuracy_validate_device() {
    local phy_id="$1" npu_index chip_id process_line
    npu_index=$((phy_id / 2))
    chip_id=$((phy_id % 2))
    process_line="$(npu-smi info | awk -v npu="${npu_index}" -v chip="${chip_id}" '
        /^\| NPU[[:space:]]+Chip/ { in_process_table = 1; next }
        in_process_table && $0 ~ /^\|/ {
            gsub(/\|/, " ")
            if ($1 == npu && $2 == chip && $3 ~ /^[0-9]+$/) print
        }
    ')"
    [[ -z "${process_line}" ]] || accuracy_fail "physical NPU ${phy_id} already has a process:${process_line}"
}

accuracy_validate_topology() {
    local -a devices
    [[ "${ACCURACY_NPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || accuracy_fail "invalid NPU device list"
    IFS=',' read -r -a devices <<<"${ACCURACY_NPU_DEVICES}"
    local expected=$((ACCURACY_PIPELINE_PARALLEL_SIZE * ACCURACY_TENSOR_PARALLEL_SIZE))
    [[ "${#devices[@]}" -eq "${expected}" ]] || \
        accuracy_fail "device count ${#devices[@]} does not match PP*TP=${expected}"
}

accuracy_shell_quote() {
    printf '%q' "$1"
}
