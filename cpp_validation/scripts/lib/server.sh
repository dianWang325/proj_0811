#!/usr/bin/env bash

cpp_validate_device() {
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
    [[ -z "${process_line}" ]] || cpp_fail "physical NPU ${phy_id} already has a process:${process_line}"
}

cpp_wait_until_devices_idle() {
    local deadline=$((SECONDS + 120)) first_device second_device
    IFS=',' read -r first_device second_device <<<"${NPU_DEVICES}"
    while ((SECONDS < deadline)); do
        if cpp_validate_device "${first_device}" 2>/dev/null && \
            cpp_validate_device "${second_device}" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "NPU devices ${NPU_DEVICES} still have processes after server shutdown." >&2
    return 1
}

cpp_validate_server_inputs() {
    [[ "${RUNNER}" =~ ^mrv[12]$ ]] || cpp_fail "CPP_RUNNER must be mrv1 or mrv2"
    [[ "${DYNAMIC}" =~ ^[01]$ ]] || cpp_fail "CPP_DYNAMIC must be 0 or 1"
    [[ "${EXECUTION_MODE}" =~ ^(eager|graph)$ ]] || cpp_fail "CPP_EXECUTION_MODE must be eager or graph"
    [[ "${REQUEST_MODE}" =~ ^(sequential|concurrent|both)$ ]] || cpp_fail "invalid CPP_REQUEST_MODE"
    [[ "${NPU_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || cpp_fail "CPP_NPU_DEVICES must contain two physical IDs"
    [[ -r "${MODEL_PATH}/config.json" ]] || cpp_fail "model config is not readable: ${MODEL_PATH}/config.json"

    local first_device second_device
    IFS=',' read -r first_device second_device <<<"${NPU_DEVICES}"
    [[ "${first_device}" != "${second_device}" ]] || cpp_fail "two distinct NPUs are required"
    cpp_validate_device "${first_device}"
    cpp_validate_device "${second_device}"
    ! cpp_is_port_open "${SERVER_PORT}" || cpp_fail "port ${SERVER_PORT} is already in use"
}

cpp_start_server() {
    local log_file="$1" compiler_dir="$2" runner_v2=0
    local additional_config='{"enable_cpu_binding":false}'
    local -a execution_args=(--enforce-eager)
    [[ "${RUNNER}" == "mrv2" ]] && runner_v2=1
    if [[ "${DYNAMIC}" == 1 ]]; then
        additional_config="{\"enable_cpu_binding\":false,\"scheduler_config\":{\"profiling_chunk_config\":{\"enabled\":true,\"need_timing\":true,\"min_chunk\":512,\"max_fit_chunk\":${MAX_FIT_CHUNK},\"trace_enabled\":true}}}"
    fi
    if [[ "${EXECUTION_MODE}" == "graph" ]]; then
        execution_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
    fi

    : >"${log_file}"
    pushd "${compiler_dir}" >/dev/null
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
        --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization 0.70 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-async-scheduling \
        "${execution_args[@]}" \
        --additional-config "${additional_config}" \
        >"${log_file}" 2>&1 </dev/null &
    CPP_SERVER_PID=$!
    export CPP_SERVER_PID
    popd >/dev/null
}

cpp_wait_until_ready() {
    local log_file="$1" deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
            tail -100 "${log_file}" >&2 || true
            return 1
        fi
        cpp_is_port_open "${SERVER_PORT}" && return 0
        sleep 5
    done
    tail -100 "${log_file}" >&2 || true
    return 1
}

cpp_stop_server() {
    [[ -n "${CPP_SERVER_PID:-}" ]] || return 0
    if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
        cpp_wait_until_devices_idle
        return
    fi
    local args pgid
    args="$(ps -p "${CPP_SERVER_PID}" -o args= 2>/dev/null || true)"
    pgid="$(ps -p "${CPP_SERVER_PID}" -o pgid= 2>/dev/null | tr -d ' ' || true)"
    if [[ "${args}" != *"${MODEL_PATH}"* || "${pgid}" != "${CPP_SERVER_PID}" ]]; then
        echo "Refusing to stop PID ${CPP_SERVER_PID}: process identity changed." >&2
        return 1
    fi
    kill -TERM -- "-${CPP_SERVER_PID}" 2>/dev/null || true
    for _ in {1..60}; do
        if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
            cpp_wait_until_devices_idle
            return
        fi
        sleep 1
    done
    echo "Server process group ${CPP_SERVER_PID} did not exit after 60s." >&2
    return 1
}
