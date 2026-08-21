#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
readonly MODEL_CONFIG="${CPP_MODEL_CONFIG:-${CPP_ROOT}/configs/models/deepseek_v4_flash.json}"
readonly SUITE_CONFIG="${CPP_SUITE_CONFIG:-${CPP_ROOT}/configs/suites/performance.json}"

source "${CPP_ROOT}/scripts/lib/common.sh"
source "${CPP_ROOT}/scripts/lib/config.sh"
source "${CPP_ROOT}/scripts/lib/server.sh"
cpp_load_performance_config_defaults "${MODEL_CONFIG}" "${SUITE_CONFIG}"

readonly MODEL_PATH="${CPP_MODEL_PATH:-/mnt/weight/DeepSeek-V4-Flash-w4a8}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-${CPP_PERF_MODEL_DEFAULTS[2]}}"
readonly MODEL_ID="${CPP_MODEL_ID:-${CPP_PERF_MODEL_DEFAULTS[6]}}"
readonly MODEL_FAMILY="${CPP_MODEL_FAMILY:-${CPP_PERF_MODEL_DEFAULTS[7]}}"
readonly TOKENIZER_PATH="${CPP_TOKENIZER_PATH:-${MODEL_PATH}}"
readonly TOKENIZER_MODE="${CPP_TOKENIZER_MODE:-${CPP_PERF_MODEL_DEFAULTS[9]}}"
readonly TOKENIZER_TRUST_REMOTE_CODE="${CPP_TOKENIZER_TRUST_REMOTE_CODE:-${CPP_PERF_MODEL_DEFAULTS[10]}}"
readonly QUANTIZATION="${CPP_QUANTIZATION-${CPP_PERF_MODEL_DEFAULTS[11]}}"
readonly SAFETENSORS_LOAD_STRATEGY="${CPP_SAFETENSORS_LOAD_STRATEGY:-${CPP_PERF_MODEL_DEFAULTS[12]}}"
readonly MAX_MODEL_LEN="${CPP_MAX_MODEL_LEN:-${CPP_PERF_SUITE_DEFAULTS[24]}}"
readonly PIPELINE_PARALLEL_SIZE=2
readonly TENSOR_PARALLEL_SIZE=4
readonly GPU_MEMORY_UTILIZATION="${CPP_GPU_MEMORY_UTILIZATION:-0.90}"
readonly KV_CACHE_MEMORY="${CPP_KV_CACHE_MEMORY:-}"
readonly ENABLE_EXPERT_PARALLEL="${CPP_ENABLE_EXPERT_PARALLEL:-0}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly SERVER_PORT="${CPP_PORT:-18080}"
readonly HCCL_PORT_RANGE="${CPP_HCCL_PORT_RANGE:-17000-17100}"
readonly SHUFFLE_SEED="${CPP_WARMUP_MATRIX_SHUFFLE_SEED:-20260821}"
readonly COOLDOWN_SECONDS="${CPP_WARMUP_MATRIX_COOLDOWN_SECONDS:-60}"
readonly DRY_RUN="${CPP_WARMUP_MATRIX_DRY_RUN:-0}"

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || \
    cpp_fail "CPP_WARMUP_MATRIX_DRY_RUN must be 0 or 1"
[[ "${COOLDOWN_SECONDS}" =~ ^[0-9]+$ ]] || \
    cpp_fail "CPP_WARMUP_MATRIX_COOLDOWN_SECONDS must be a non-negative integer"

# label category dynamic measurement warmup_enabled warmup_dataset count concurrency timing
declare -ar MATRIX_CASES=(
    "matched_variable matched 1 variable 1 variable 30 4 true"
    "matched_fixed matched 1 fixed 1 fixed 5 1 true"
    "cross_variable_fixed_c4 validation 1 variable 1 fixed 5 4 true"
    "cross_variable_fixed_c1 validation 1 variable 1 fixed 5 1 true"
    "cross_fixed_variable_c1 validation 1 fixed 1 variable 30 1 true"
    "cross_fixed_variable_c4 validation 1 fixed 1 variable 30 4 true"
    "cpp_off_variable cpp_off 0 variable 0 variable 0 4 false"
    "cpp_off_fixed cpp_off 0 fixed 0 fixed 0 1 false"
    "no_warmup_variable no_warmup 1 variable 0 variable 0 4 true"
    "no_warmup_fixed no_warmup 1 fixed 0 fixed 0 1 true"
)

build_order() {
    python3 - "${SHUFFLE_SEED}" "${MATRIX_CASES[@]}" <<'PY'
import random
import sys

seed, *cases = sys.argv[1:]
random.Random(seed).shuffle(cases)
print(*cases, sep="\n")
PY
}

print_order() {
    local position=0 spec
    printf 'position\tlabel\tcategory\tcpp\tmeasurement_dataset\twarmup_enabled\twarmup_dataset\twarmup_count\twarmup_concurrency\tneed_timing\n'
    while IFS= read -r spec; do
        position=$((position + 1))
        read -r label category dynamic dataset warmup_enabled warmup_dataset \
            warmup_count warmup_concurrency need_timing <<<"${spec}"
        printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${position}" "${label}" "${category}" "${dynamic}" "${dataset}" \
            "${warmup_enabled}" "${warmup_dataset}" "${warmup_count}" \
            "${warmup_concurrency}" "${need_timing}"
    done < <(build_order)
}

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "CPP_WARMUP_DISTRIBUTION_DRY_RUN cases=${#MATRIX_CASES[@]} seed=${SHUFFLE_SEED}"
    print_order
    exit 0
fi

select_device_group() {
    local requested="${CPP_NPU_DEVICES:-}" group device idle
    local -a candidates
    if [[ -n "${requested}" ]]; then
        candidates=("${requested}")
    else
        candidates=("0,1,2,3,4,5,6,7" "8,9,10,11,12,13,14,15")
    fi
    for group in "${candidates[@]}"; do
        IFS=',' read -r -a devices <<<"${group}"
        [[ "${#devices[@]}" -eq 8 ]] || continue
        idle=1
        for device in "${devices[@]}"; do
            if ! cpp_validate_device "${device}" >/dev/null 2>&1; then
                idle=0
                break
            fi
        done
        if [[ "${idle}" -eq 1 ]]; then
            printf '%s\n' "${group}"
            return
        fi
    done
    cpp_fail "neither requested nor preferred 8-NPU group is fully idle"
}

readonly NPU_DEVICES="$(select_device_group)"
export NPU_DEVICES
[[ -r "${MODEL_PATH}/config.json" ]] || cpp_fail "model config is not readable: ${MODEL_PATH}/config.json"
[[ ! -e "${PROJECT_ROOT}/deps/vllm-ascend/.git/index.lock" ]] || \
    cpp_fail "vllm-ascend has an active git lock"
[[ "$(git -C "${PROJECT_ROOT}" branch --show-current)" == "main" ]] || \
    cpp_fail "project repository must be on main"
[[ "$(git -C "${PROJECT_ROOT}/deps/vllm-ascend" branch --show-current)" == \
   "feat/profiling-chunk-execution-trace" ]] || \
    cpp_fail "vllm-ascend must be on feat/profiling-chunk-execution-trace"
! cpp_is_port_open "${SERVER_PORT}" || cpp_fail "port ${SERVER_PORT} is already in use"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"
python3 "${CPP_ROOT}/configuration/performance.py" global \
    --model-config "${MODEL_CONFIG}" \
    --suite-config "${SUITE_CONFIG}" \
    --artifact-root "${CPP_ARTIFACT_ROOT}" \
    --project-root "${PROJECT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --tokenizer-mode "${TOKENIZER_MODE}" \
    --tokenizer-trust-remote-code "${TOKENIZER_TRUST_REMOTE_CODE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --npu-devices "${NPU_DEVICES}" \
    --quantization "${QUANTIZATION}" \
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"

cpp_initialize_run performance-matrix
readonly ORDER_LOG="${CPP_RUN_DIR}/matrix_order.tsv"
readonly RESULT_LOG="${CPP_RUN_DIR}/matrix_results.tsv"
print_order >"${ORDER_LOG}"
printf 'position\tlabel\texit_code\n' >"${RESULT_LOG}"

echo "CPP_WARMUP_DISTRIBUTION_START run=${CPP_RUN_ID} devices=${NPU_DEVICES} cases=${#MATRIX_CASES[@]} seed=${SHUFFLE_SEED}"
position=0
while IFS= read -r spec; do
    position=$((position + 1))
    read -r label category dynamic dataset warmup_enabled warmup_dataset \
        warmup_count warmup_concurrency need_timing <<<"${spec}"
    echo "CPP_WARMUP_DISTRIBUTION_CASE_START position=${position} label=${label} dataset=${dataset} warmup=${warmup_dataset} count=${warmup_count} concurrency=${warmup_concurrency}"
    case_status=0
    env \
        CPP_GLOBAL_PREFLIGHT_DONE=1 \
        CPP_RUN_ID="${CPP_RUN_ID}" \
        CPP_RUN_DIR="${CPP_RUN_DIR}" \
        CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT}" \
        CPP_MODEL_CONFIG="${MODEL_CONFIG}" \
        CPP_SUITE_CONFIG="${SUITE_CONFIG}" \
        CPP_MODEL_PATH="${MODEL_PATH}" \
        CPP_TOKENIZER_PATH="${TOKENIZER_PATH}" \
        CPP_NPU_DEVICES="${NPU_DEVICES}" \
        CPP_PORT="${SERVER_PORT}" \
        CPP_HCCL_PORT_RANGE="${HCCL_PORT_RANGE}" \
        CPP_PIPELINE_PARALLEL_SIZE=2 \
        CPP_TENSOR_PARALLEL_SIZE=4 \
        CPP_RUNNER=mrv2 \
        CPP_DYNAMIC="${dynamic}" \
        CPP_EXECUTION_MODE=eager \
        CPP_ASYNC_SCHEDULING=0 \
        CPP_NEED_TIMING="${need_timing}" \
        CPP_PERF_DATASET="${dataset}" \
        CPP_MANUAL_WARMUP_ENABLED="${warmup_enabled}" \
        CPP_MANUAL_WARMUP_DATASET_MODE=generated \
        CPP_MANUAL_WARMUP_PERF_DATASET="${warmup_dataset}" \
        CPP_MANUAL_WARMUP_CONCURRENCY="${warmup_concurrency}" \
        CPP_WARMUP_COUNT="${warmup_count}" \
        CPP_MATRIX_ROUND=1 \
        CPP_MATRIX_POSITION="${position}" \
        "${CPP_ROOT}/scripts/run_perf_case.sh" || case_status=$?
    printf '%d\t%s\t%d\n' "${position}" "${label}" "${case_status}" >>"${RESULT_LOG}"
    python3 "${CPP_ROOT}/analysis/summarize_warmup_distribution.py" "${CPP_RUN_DIR}" || true
    if [[ "${case_status}" -ne 0 ]]; then
        echo "CPP_WARMUP_DISTRIBUTION_CASE_FAIL position=${position} label=${label} exit_code=${case_status}" >&2
        exit "${case_status}"
    fi
    echo "CPP_WARMUP_DISTRIBUTION_CASE_PASS position=${position} label=${label}"
    if [[ "${position}" -lt "${#MATRIX_CASES[@]}" && "${COOLDOWN_SECONDS}" -gt 0 ]]; then
        echo "CPP_WARMUP_DISTRIBUTION_COOLDOWN seconds=${COOLDOWN_SECONDS}"
        sleep "${COOLDOWN_SECONDS}"
    fi
done < <(build_order)

python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
python3 "${CPP_ROOT}/analysis/summarize_warmup_distribution.py" "${CPP_RUN_DIR}"
echo "CPP_WARMUP_DISTRIBUTION_DONE run=${CPP_RUN_ID} cases=${#MATRIX_CASES[@]} artifacts=${CPP_RUN_DIR}"
