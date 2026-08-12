#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORK_DIR="/home/w00985415"
readonly TEST_DIR="${WORK_DIR}/proj_0811/smoke_test"
readonly MODEL_PATH="${CPP_MODEL_PATH:-/mnt/a800_weight/Qwen3-30B-A3B-W8A8}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-qwen3-30b-a3b-w8a8}"
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-${MRV2_CPP_NPU_DEVICES:-8,9}}"
readonly SERVER_PORT="${CPP_PORT:-${MRV2_CPP_PORT:-18080}}"
readonly RUNNER="${CPP_RUNNER:-mrv2}"
readonly DYNAMIC="${CPP_DYNAMIC:-1}"
readonly EXECUTION_MODE="${CPP_EXECUTION_MODE:-eager}"
readonly REQUEST_MODE="${CPP_REQUEST_MODE:-both}"
readonly TOKEN_TARGETS="${CPP_TOKEN_TARGETS:-10000,20000,40000}"
readonly STARTUP_TIMEOUT="${CPP_STARTUP_TIMEOUT:-1200}"
readonly HCCL_PORT_RANGE="${CPP_HCCL_PORT_RANGE:-17000-17100}"
readonly CASE_NAME="${RUNNER}_$([[ "${DYNAMIC}" == 1 ]] && echo dynamic || echo static)_${EXECUTION_MODE}_${REQUEST_MODE}"
readonly LOG_FILE="${CPP_LOG_FILE:-${TEST_DIR}/serve_${CASE_NAME}.log}"
readonly PID_FILE="${TEST_DIR}/serve_cpp_functional.pid"

server_pid=""

fail() {
    echo "CPP_FUNCTIONAL_FAIL: $*" >&2
    exit 1
}

is_port_open() {
    python - "${SERVER_PORT}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        local args pgid
        args="$(ps -p "${server_pid}" -o args= 2>/dev/null || true)"
        pgid="$(ps -p "${server_pid}" -o pgid= 2>/dev/null | tr -d ' ' || true)"
        if [[ "${args}" == *"${MODEL_PATH}"* && "${pgid}" == "${server_pid}" ]]; then
            kill -TERM -- "-${server_pid}" 2>/dev/null || true
            for _ in {1..60}; do
                kill -0 "${server_pid}" 2>/dev/null || break
                sleep 1
            done
        else
            echo "Refusing to stop PID ${server_pid}: process identity changed." >&2
            status=1
        fi
    fi
    rm -f "${PID_FILE}"
    exit "${status}"
}

validate_device() {
    local phy_id="$1"
    local npu_index=$((phy_id / 2))
    local chip_id=$((phy_id % 2))
    local process_line
    process_line="$(npu-smi info | awk -v npu="${npu_index}" -v chip="${chip_id}" '
        /^\| NPU[[:space:]]+Chip/ { in_process_table = 1; next }
        in_process_table && $0 ~ /^\|/ {
            gsub(/\|/, " ")
            if ($1 == npu && $2 == chip && $3 ~ /^[0-9]+$/) print
        }
    ')"
    [[ -z "${process_line}" ]] || fail "physical NPU ${phy_id} already has a process:${process_line}"
}

wait_until_ready() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            tail -100 "${LOG_FILE}" >&2 || true
            return 1
        fi
        is_port_open && return 0
        sleep 5
    done
    tail -100 "${LOG_FILE}" >&2 || true
    return 1
}

main() {
    [[ "${RUNNER}" =~ ^mrv[12]$ ]] || fail "CPP_RUNNER must be mrv1 or mrv2"
    [[ "${DYNAMIC}" =~ ^[01]$ ]] || fail "CPP_DYNAMIC must be 0 or 1"
    [[ "${EXECUTION_MODE}" =~ ^(eager|graph)$ ]] || fail "CPP_EXECUTION_MODE must be eager or graph"
    [[ "${REQUEST_MODE}" =~ ^(sequential|concurrent|both)$ ]] || fail "invalid CPP_REQUEST_MODE"
    [[ "${NPU_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || fail "CPP_NPU_DEVICES must contain two physical IDs"
    [[ -r "${MODEL_PATH}/config.json" ]] || fail "model config is not readable"

    IFS=',' read -r first_device second_device <<<"${NPU_DEVICES}"
    [[ "${first_device}" != "${second_device}" ]] || fail "two distinct NPUs are required"
    validate_device "${first_device}"
    validate_device "${second_device}"
    is_port_open && fail "port ${SERVER_PORT} is already in use"

    local runner_v2=0
    [[ "${RUNNER}" == "mrv2" ]] && runner_v2=1
    local additional_config='{"enable_cpu_binding":false}'
    if [[ "${DYNAMIC}" == 1 ]]; then
        additional_config='{"enable_cpu_binding":false,"scheduler_config":{"profiling_chunk_config":{"enabled":true,"need_timing":true,"min_chunk":512,"max_fit_chunk":8,"trace_enabled":true}}}'
    fi
    local -a execution_args=(--enforce-eager)
    if [[ "${EXECUTION_MODE}" == "graph" ]]; then
        execution_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
    fi

    : >"${LOG_FILE}"
    echo "Starting ${CASE_NAME} on physical NPUs ${NPU_DEVICES}; log=${LOG_FILE}"
    ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" \
    HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_PORT_RANGE}" \
    VLLM_USE_V1=1 \
    VLLM_USE_V2_MODEL_RUNNER="${runner_v2}" \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
    nohup setsid vllm serve "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host 127.0.0.1 \
        --port "${SERVER_PORT}" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 2 \
        --max-model-len 49152 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.70 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-async-scheduling \
        "${execution_args[@]}" \
        --additional-config "${additional_config}" \
        >"${LOG_FILE}" 2>&1 </dev/null &
    server_pid=$!
    printf '%s\n' "${server_pid}" >"${PID_FILE}"
    trap cleanup EXIT INT TERM

    wait_until_ready || fail "server startup failed"
    CPP_MODEL_PATH="${MODEL_PATH}" \
    CPP_MODEL_NAME="${MODEL_NAME}" \
    CPP_PORT="${SERVER_PORT}" \
    CPP_REQUEST_MODE="${REQUEST_MODE}" \
    CPP_TOKEN_TARGETS="${TOKEN_TARGETS}" \
        "${TEST_DIR}/cpp_long_context_test.py"

    "${TEST_DIR}/validate_cpp_trace.py" "${LOG_FILE}" \
        --runner "${RUNNER}" \
        --dynamic "${DYNAMIC}" \
        --request-mode "${REQUEST_MODE}"
    echo "CPP_FUNCTIONAL_PASS case=${CASE_NAME} log=${LOG_FILE}"
}

main "$@"
