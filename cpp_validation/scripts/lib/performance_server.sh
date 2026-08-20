#!/usr/bin/env bash

cpp_perf_validate_server_inputs() {
    [[ "${RUNNER}" =~ ^mrv[12]$ ]] || cpp_fail "CPP_RUNNER must be mrv1 or mrv2"
    [[ "${DYNAMIC}" =~ ^[01]$ ]] || cpp_fail "CPP_DYNAMIC must be 0 or 1"
    [[ "${EXECUTION_MODE}" =~ ^(eager|graph)$ ]] || cpp_fail "invalid CPP_EXECUTION_MODE"
    [[ "${PERF_DATASET}" =~ ^(fixed|variable)$ ]] || cpp_fail "invalid CPP_PERF_DATASET"
    [[ "${NPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || cpp_fail "invalid CPP_NPU_DEVICES"
    [[ -r "${MODEL_PATH}/config.json" ]] || cpp_fail "model config is not readable: ${MODEL_PATH}/config.json"
    if [[ "${EXECUTION_MODE}" == "graph" ]]; then
        [[ "${RUNNER}" == "mrv2" && "${DYNAMIC}" == 1 ]] || \
            cpp_fail "graph performance case requires CPP enabled with mrv2"
    fi

    local -a devices
    IFS=',' read -r -a devices <<<"${NPU_DEVICES}"
    local expected_count=$((PIPELINE_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE))
    [[ "${#devices[@]}" -eq "${expected_count}" ]] || \
        cpp_fail "device count ${#devices[@]} does not match PP*TP=${expected_count}"
    local device
    for device in "${devices[@]}"; do
        cpp_validate_device "${device}"
    done
    ! cpp_is_port_open "${SERVER_PORT}" || cpp_fail "port ${SERVER_PORT} is already in use"
}

cpp_perf_start_server() {
    local log_file="$1" compiler_dir="$2" runner_v2=0
    local need_timing="${NEED_TIMING:-${CPP_NEED_TIMING:-true}}"
    local tokenizer_mode="${TOKENIZER_MODE}"
    local tokenizer_path="${TOKENIZER_PATH}"
    local tokenizer_trust_remote_code="${TOKENIZER_TRUST_REMOTE_CODE}"
    local quantization="${QUANTIZATION}"
    local enable_expert_parallel="${ENABLE_EXPERT_PARALLEL}"
    local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    local max_num_batched_tokens="${CPP_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
    local safetensors_load_strategy="${SAFETENSORS_LOAD_STRATEGY}"
    local kv_cache_memory="${KV_CACHE_MEMORY}"
    local async_scheduling="${ASYNC_SCHEDULING:-${CPP_ASYNC_SCHEDULING:-0}}"
    local additional_config='{"enable_cpu_binding":false}'
    local -a execution_args=(--enforce-eager)
    local -a prefix_cache_args=(--no-enable-prefix-caching)
    local -a async_scheduling_args=(--no-async-scheduling)
    local -a load_strategy_args=()
    local -a model_args=(--tokenizer-mode "${tokenizer_mode}")
    [[ -z "${tokenizer_path}" || "${tokenizer_path}" == "${MODEL_PATH}" ]] || \
        model_args+=(--tokenizer "${tokenizer_path}")
    [[ "${tokenizer_trust_remote_code}" == "0" ]] || \
        model_args+=(--trust-remote-code)
    [[ -z "${quantization}" ]] || model_args+=(--quantization "${quantization}")
    [[ -z "${kv_cache_memory}" ]] || model_args+=(--kv-cache-memory "${kv_cache_memory}")
    [[ "${enable_expert_parallel}" == "0" || \
       "${enable_expert_parallel}" == "1" ]] || \
        cpp_fail "CPP_ENABLE_EXPERT_PARALLEL must be 0 or 1"
    [[ "${enable_expert_parallel}" == "0" ]] || \
        model_args+=(--enable-expert-parallel)
    [[ "${async_scheduling}" == "0" || "${async_scheduling}" == "1" ]] || \
        cpp_fail "CPP_ASYNC_SCHEDULING must be 0 or 1"
    [[ "${async_scheduling}" == "0" ]] || \
        async_scheduling_args=(--async-scheduling)
    # The model configuration uses "auto" as the portable logical default.
    # This vLLM revision has no CLI choice named "auto", so let vLLM select
    # its own default by omitting the option. Explicit supported strategies
    # continue to be forwarded unchanged.
    if [[ -n "${safetensors_load_strategy}" && \
          "${safetensors_load_strategy}" != "auto" && \
          "${safetensors_load_strategy}" != "None" ]]; then
        load_strategy_args=(
            --safetensors-load-strategy "${safetensors_load_strategy}"
        )
    fi
    [[ "${RUNNER}" == "mrv2" ]] && runner_v2=1
    if [[ "${DYNAMIC}" == 1 ]]; then
        [[ "${need_timing}" == "true" || "${need_timing}" == "false" ]] || \
            cpp_fail "CPP_NEED_TIMING must be true or false"
        additional_config="{\"enable_cpu_binding\":false,\"scheduler_config\":{\"profiling_chunk_config\":{\"enabled\":true,\"need_timing\":${need_timing},\"execution_mode_trace_enabled\":true,\"smooth_factor\":${CPP_SMOOTH_FACTOR}}}}"
    fi
    [[ "${PREFIX_CACHE_ENABLED}" == 1 ]] && \
        prefix_cache_args=(--enable-prefix-caching)
    if [[ "${EXECUTION_MODE}" == "graph" ]]; then
        execution_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
    fi

    : >"${log_file}"
    pushd "${compiler_dir}" >/dev/null
    ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" \
    HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_PORT_RANGE}" \
    VLLM_USE_V1=1 \
    VLLM_USE_V2_MODEL_RUNNER="${runner_v2}" \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    nohup setsid vllm serve "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host 127.0.0.1 \
        --port "${SERVER_PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${max_num_batched_tokens}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --enable-chunked-prefill \
        "${prefix_cache_args[@]}" \
        "${async_scheduling_args[@]}" \
        "${load_strategy_args[@]}" \
        "${model_args[@]}" \
        --block-size 32 \
        "${execution_args[@]}" \
        --additional-config "${additional_config}" \
        >"${log_file}" 2>&1 </dev/null &
    CPP_SERVER_PID=$!
    export CPP_SERVER_PID
    popd >/dev/null
}

cpp_perf_wait_until_ready() {
    local log_file="$1" deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
            tail -200 "${log_file}" >&2 || true
            return 1
        fi
        cpp_is_port_open "${SERVER_PORT}" && return 0
        sleep 5
    done
    tail -200 "${log_file}" >&2 || true
    return 1
}

cpp_perf_wait_until_devices_idle() {
    local deadline=$((SECONDS + 180)) device all_idle
    local -a devices
    IFS=',' read -r -a devices <<<"${NPU_DEVICES}"
    while ((SECONDS < deadline)); do
        all_idle=1
        for device in "${devices[@]}"; do
            if ! cpp_validate_device "${device}" 2>/dev/null; then
                all_idle=0
                break
            fi
        done
        [[ "${all_idle}" -eq 1 ]] && return 0
        sleep 2
    done
    echo "NPU devices ${NPU_DEVICES} still have processes after server shutdown." >&2
    return 1
}

cpp_perf_stop_server() {
    [[ -n "${CPP_SERVER_PID:-}" ]] || return 0
    if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
        cpp_perf_wait_until_devices_idle
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
    for _ in {1..90}; do
        if ! kill -0 "${CPP_SERVER_PID}" 2>/dev/null; then
            cpp_perf_wait_until_devices_idle
            return
        fi
        sleep 1
    done
    echo "Server process group ${CPP_SERVER_PID} did not exit after 90s." >&2
    return 1
}
