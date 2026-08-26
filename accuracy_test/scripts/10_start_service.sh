#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

dry_run=0
while (($#)); do
    case "$1" in
        --cpp)
            [[ $# -ge 2 ]] || accuracy_fail "--cpp requires on or off"
            case "$2" in on) ACCURACY_CPP_ENABLED=1 ;; off) ACCURACY_CPP_ENABLED=0 ;; *) accuracy_fail "--cpp requires on or off" ;; esac
            shift 2
            ;;
        --runner)
            [[ $# -ge 2 ]] || accuracy_fail "--runner requires mrv1 or mrv2"
            ACCURACY_RUNNER="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--cpp on|off] [--runner mrv1|mrv2] [--dry-run]"
            exit 0
            ;;
        *) accuracy_fail "unknown argument: $1" ;;
    esac
done

accuracy_validate_bool ACCURACY_CPP_ENABLED "${ACCURACY_CPP_ENABLED}"
accuracy_validate_bool ACCURACY_ENABLE_EXPERT_PARALLEL "${ACCURACY_ENABLE_EXPERT_PARALLEL}"
accuracy_validate_bool ACCURACY_ENABLE_PREFIX_CACHING "${ACCURACY_ENABLE_PREFIX_CACHING}"
accuracy_validate_runner "${ACCURACY_RUNNER}"
accuracy_validate_topology
for variable_name in \
    ACCURACY_MAX_MODEL_LEN \
    ACCURACY_MAX_NUM_BATCHED_TOKENS \
    ACCURACY_MAX_NUM_SEQS \
    ACCURACY_MAX_OUT_LEN \
    ACCURACY_MAX_CUDAGRAPH_CAPTURE_SIZE; do
    variable_value="${!variable_name}"
    [[ "${variable_value}" =~ ^[1-9][0-9]*$ ]] || \
        accuracy_fail "${variable_name} must be a positive integer"
done
((ACCURACY_MAX_OUT_LEN < ACCURACY_MAX_MODEL_LEN)) || \
    accuracy_fail "ACCURACY_MAX_OUT_LEN must be smaller than ACCURACY_MAX_MODEL_LEN"
case "${ACCURACY_GRAPH_MODE}" in
    NONE|FULL_DECODE_ONLY) ;;
    *) accuracy_fail "ACCURACY_GRAPH_MODE must be NONE or FULL_DECODE_ONLY" ;;
esac
[[ "${ACCURACY_CUDAGRAPH_CAPTURE_SIZES}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || \
    accuracy_fail "ACCURACY_CUDAGRAPH_CAPTURE_SIZES must be a comma-separated integer list"
for command_name in python vllm npu-smi nohup setsid timeout stat; do
    accuracy_require_command "${command_name}"
done

runner_v2=0
[[ "${ACCURACY_RUNNER}" == "mrv2" ]] && runner_v2=1
additional_config='{"enable_cpu_binding":false}'
if [[ "${ACCURACY_CPP_ENABLED}" == "1" ]]; then
    additional_config="{\"enable_cpu_binding\":false,\"scheduler_config\":{\"profiling_chunk_config\":{\"enabled\":true,\"need_timing\":${ACCURACY_CPP_NEED_TIMING},\"smooth_factor\":${ACCURACY_CPP_SMOOTH_FACTOR}}}}"
fi

command=(
    vllm serve "${ACCURACY_MODEL_PATH}"
    --served-model-name "${ACCURACY_SERVED_MODEL_NAME}"
    --host "${ACCURACY_HOST}"
    --port "${ACCURACY_PORT}"
    --tensor-parallel-size "${ACCURACY_TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size "${ACCURACY_PIPELINE_PARALLEL_SIZE}"
    --max-model-len "${ACCURACY_MAX_MODEL_LEN}"
    --max-num-batched-tokens "${ACCURACY_MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${ACCURACY_MAX_NUM_SEQS}"
    --gpu-memory-utilization "${ACCURACY_GPU_MEMORY_UTILIZATION}"
    --enable-chunked-prefill
    --no-async-scheduling
    --tokenizer-mode "${ACCURACY_TOKENIZER_MODE}"
    --tokenizer "${ACCURACY_TOKENIZER_PATH}"
    --trust-remote-code
    --quantization "${ACCURACY_QUANTIZATION}"
    --block-size 32
    --additional-config "${additional_config}"
)
if [[ "${ACCURACY_ENABLE_PREFIX_CACHING}" == "1" ]]; then
    command+=(--enable-prefix-caching)
else
    command+=(--no-enable-prefix-caching)
fi
if [[ "${ACCURACY_GRAPH_MODE}" == "NONE" ]]; then
    command+=(--enforce-eager)
else
    compilation_config="{\"cudagraph_mode\":\"${ACCURACY_GRAPH_MODE}\",\"cudagraph_capture_sizes\":[${ACCURACY_CUDAGRAPH_CAPTURE_SIZES}],\"max_cudagraph_capture_size\":${ACCURACY_MAX_CUDAGRAPH_CAPTURE_SIZE}}"
    command+=(--compilation-config "${compilation_config}")
fi
if [[ "${ACCURACY_ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    command+=(--enable-expert-parallel)
fi
if [[ -n "${ACCURACY_SAFETENSORS_LOAD_STRATEGY}" && \
      "${ACCURACY_SAFETENSORS_LOAD_STRATEGY}" != "auto" ]]; then
    command+=(--safetensors-load-strategy "${ACCURACY_SAFETENSORS_LOAD_STRATEGY}")
fi

printf 'Effective service command:\n'
printf '  %q' env \
    "ASCEND_RT_VISIBLE_DEVICES=${ACCURACY_NPU_DEVICES}" \
    "HCCL_NPU_SOCKET_PORT_RANGE=${ACCURACY_HCCL_PORT_RANGE}" \
    "VLLM_USE_V1=1" \
    "VLLM_USE_V2_MODEL_RUNNER=${runner_v2}"
printf ' %q' "${command[@]}"
printf '\n'
[[ "${dry_run}" == "0" ]] || exit 0

[[ -f /.dockerenv ]] || accuracy_fail "run this script inside container wd_test0811"
timeout 15 stat "${ACCURACY_MODEL_PATH}/config.json" >/dev/null || \
    accuracy_fail "model config is unavailable or the mount timed out: ${ACCURACY_MODEL_PATH}/config.json"

pid_file="${ACCURACY_ROOT}/state/service.pid"
state_file="${ACCURACY_ROOT}/state/service.env"
if [[ -s "${pid_file}" ]]; then
    old_pid="$(<"${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        accuracy_fail "recorded service PID ${old_pid} is still running"
    fi
    rm -f "${pid_file}" "${state_file}"
fi
! accuracy_is_port_open "${ACCURACY_PORT}" || accuracy_fail "port ${ACCURACY_PORT} is already in use"
IFS=',' read -r -a devices <<<"${ACCURACY_NPU_DEVICES}"
for device in "${devices[@]}"; do
    accuracy_validate_device "${device}"
done

run_id="$(date +%Y%m%d_%H%M%S)_${ACCURACY_RUNNER}_cpp${ACCURACY_CPP_ENABLED}"
log_file="${ACCURACY_ROOT}/logs/service/${run_id}.log"
environment_file="${ACCURACY_ROOT}/logs/environment/${run_id}.env"
compiler_dir="${ACCURACY_ROOT}/tmp/compiler/${run_id}"
mkdir -p "${compiler_dir}"
: >"${log_file}"

pythonpath="${PROJECT_ROOT}/deps/vllm:${PROJECT_ROOT}/deps/vllm-ascend:${PROJECT_ROOT}"
pushd "${compiler_dir}" >/dev/null
nohup setsid env \
    "ASCEND_RT_VISIBLE_DEVICES=${ACCURACY_NPU_DEVICES}" \
    "HCCL_NPU_SOCKET_PORT_RANGE=${ACCURACY_HCCL_PORT_RANGE}" \
    "VLLM_USE_V1=1" \
    "VLLM_USE_V2_MODEL_RUNNER=${runner_v2}" \
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600" \
    "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" \
    "PYTHONPATH=${pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}" >"${log_file}" 2>&1 </dev/null &
server_pid=$!
popd >/dev/null

printf '%s\n' "${server_pid}" >"${pid_file}"
{
    printf 'ACCURACY_SERVICE_RUN_ID=%q\n' "${run_id}"
    printf 'ACCURACY_SERVICE_PID=%q\n' "${server_pid}"
    printf 'ACCURACY_SERVICE_LOG=%q\n' "${log_file}"
    printf 'ACCURACY_CPP_ENABLED=%q\n' "${ACCURACY_CPP_ENABLED}"
    printf 'ACCURACY_RUNNER=%q\n' "${ACCURACY_RUNNER}"
    printf 'ACCURACY_MODEL_PATH=%q\n' "${ACCURACY_MODEL_PATH}"
    printf 'ACCURACY_SERVED_MODEL_NAME=%q\n' "${ACCURACY_SERVED_MODEL_NAME}"
    printf 'ACCURACY_NPU_DEVICES=%q\n' "${ACCURACY_NPU_DEVICES}"
    printf 'ACCURACY_PORT=%q\n' "${ACCURACY_PORT}"
    printf 'ACCURACY_PIPELINE_PARALLEL_SIZE=%q\n' "${ACCURACY_PIPELINE_PARALLEL_SIZE}"
    printf 'ACCURACY_TENSOR_PARALLEL_SIZE=%q\n' "${ACCURACY_TENSOR_PARALLEL_SIZE}"
    printf 'ACCURACY_MAX_MODEL_LEN=%q\n' "${ACCURACY_MAX_MODEL_LEN}"
    printf 'ACCURACY_MAX_NUM_BATCHED_TOKENS=%q\n' "${ACCURACY_MAX_NUM_BATCHED_TOKENS}"
    printf 'ACCURACY_MAX_NUM_SEQS=%q\n' "${ACCURACY_MAX_NUM_SEQS}"
    printf 'ACCURACY_ENABLE_PREFIX_CACHING=%q\n' "${ACCURACY_ENABLE_PREFIX_CACHING}"
    printf 'ACCURACY_GRAPH_MODE=%q\n' "${ACCURACY_GRAPH_MODE}"
    printf 'ACCURACY_CUDAGRAPH_CAPTURE_SIZES=%q\n' "${ACCURACY_CUDAGRAPH_CAPTURE_SIZES}"
    printf 'ACCURACY_MAX_CUDAGRAPH_CAPTURE_SIZE=%q\n' "${ACCURACY_MAX_CUDAGRAPH_CAPTURE_SIZE}"
    printf 'ACCURACY_SERVICE_STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
} | tee "${state_file}" >"${environment_file}"

accuracy_info "service started as PID ${server_pid}"
accuracy_info "log: ${log_file}"
accuracy_info "next: ${ACCURACY_ROOT}/scripts/11_check_service.sh --wait ${ACCURACY_STARTUP_TIMEOUT} --probe"
