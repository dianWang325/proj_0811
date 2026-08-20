#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
readonly MODEL_CONFIG="${CPP_MODEL_CONFIG:-${CPP_ROOT}/configs/models/deepseek_v4_flash.json}"
readonly SUITE_CONFIG="${CPP_SUITE_CONFIG:-${CPP_ROOT}/configs/suites/performance.json}"

source "${CPP_ROOT}/scripts/lib/common.sh"
source "${CPP_ROOT}/scripts/lib/config.sh"
cpp_load_performance_config_defaults "${MODEL_CONFIG}" "${SUITE_CONFIG}"

readonly MODEL_PATH="${CPP_MODEL_PATH:-${CPP_PERF_MODEL_DEFAULTS[0]}}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-${CPP_PERF_MODEL_DEFAULTS[2]}}"
readonly MAX_MODEL_LEN="${CPP_MAX_MODEL_LEN:-${CPP_PERF_MODEL_DEFAULTS[3]}}"
readonly PIPELINE_PARALLEL_SIZE=2
readonly TENSOR_PARALLEL_SIZE=4
readonly MODEL_ID="${CPP_MODEL_ID:-${CPP_PERF_MODEL_DEFAULTS[6]}}"
readonly MODEL_FAMILY="${CPP_MODEL_FAMILY:-${CPP_PERF_MODEL_DEFAULTS[7]}}"
readonly TOKENIZER_PATH="${CPP_TOKENIZER_PATH:-${CPP_PERF_MODEL_DEFAULTS[8]:-${MODEL_PATH}}}"
readonly TOKENIZER_MODE="${CPP_TOKENIZER_MODE:-${CPP_PERF_MODEL_DEFAULTS[9]}}"
readonly TOKENIZER_TRUST_REMOTE_CODE="${CPP_TOKENIZER_TRUST_REMOTE_CODE:-${CPP_PERF_MODEL_DEFAULTS[10]}}"
readonly QUANTIZATION="${CPP_QUANTIZATION-${CPP_PERF_MODEL_DEFAULTS[11]}}"
readonly SAFETENSORS_LOAD_STRATEGY="${CPP_SAFETENSORS_LOAD_STRATEGY:-${CPP_PERF_MODEL_DEFAULTS[12]}}"
readonly GPU_MEMORY_UTILIZATION="${CPP_GPU_MEMORY_UTILIZATION:-0.90}"
readonly KV_CACHE_MEMORY="${CPP_KV_CACHE_MEMORY:-}"
readonly ENABLE_EXPERT_PARALLEL="${CPP_ENABLE_EXPERT_PARALLEL:-0}"
readonly API_MODE="${CPP_API_MODE:-${CPP_PERF_SUITE_DEFAULTS[27]}}"
readonly PROMPT_MODE="${CPP_PROMPT_MODE:-${CPP_PERF_SUITE_DEFAULTS[28]}}"
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-8,9,10,11,12,13,14,15}"
readonly AISBENCH_AUTO_TOOLS_ROOT="${CPP_AISBENCH_AUTO_TOOLS_ROOT:-${PROJECT_ROOT}/deps/aisbench_auto_tools_prefix}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly MAX_ROUNDS="${CPP_MATRIX_MAX_ROUNDS:-3}"
readonly SHUFFLE_SEED="${CPP_MATRIX_SHUFFLE_SEED:-20260820}"
readonly DRY_RUN="${CPP_MATRIX_DRY_RUN:-0}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"

[[ "${MAX_ROUNDS}" =~ ^[1-3]$ ]] || \
    cpp_fail "CPP_MATRIX_MAX_ROUNDS must be between 1 and 3"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || \
    cpp_fail "CPP_MATRIX_DRY_RUN must be 0 or 1"

# CPP-off cases use need_timing=false because timing helpers are inactive.
# CPP-on cases cover both timing values. This yields 12 meaningful cases.
readonly -a MATRIX_CASES=(
    "mrv1 0 false 0"
    "mrv1 0 false 1"
    "mrv2 0 false 0"
    "mrv2 0 false 1"
    "mrv1 1 false 0"
    "mrv1 1 false 1"
    "mrv1 1 true 0"
    "mrv1 1 true 1"
    "mrv2 1 false 0"
    "mrv2 1 false 1"
    "mrv2 1 true 0"
    "mrv2 1 true 1"
)

build_round_order() {
    local round="$1" previous_order="$2"
    if [[ "${round}" -eq 1 ]]; then
        printf '%s\n' "${MATRIX_CASES[@]}"
        return
    fi
    python3 - "${SHUFFLE_SEED}" "${round}" "${previous_order}" \
        "${MATRIX_CASES[@]}" <<'PY'
import random
import sys

seed, round_number, previous, *canonical = sys.argv[1:]
rng = random.Random(f"{seed}:round:{round_number}")
while True:
    shuffled = canonical.copy()
    rng.shuffle(shuffled)
    serialized = "|".join(shuffled)
    if shuffled != canonical and serialized != previous:
        print(*shuffled, sep="\n")
        break
PY
}

print_dry_run() {
    local round position spec previous_order
    local runner dynamic need_timing async_scheduling
    local -a round_cases
    previous_order="$(IFS='|'; printf '%s' "${MATRIX_CASES[*]}")"
    echo "CPP_MATRIX_DRY_RUN cases_per_round=12 max_rounds=${MAX_ROUNDS} seed=${SHUFFLE_SEED} dataset=variable execution_mode=eager pp=2 tp=4"
    printf 'round\tposition\trunner\tcpp\tneed_timing\tasync_scheduling\n'
    for ((round = 1; round <= MAX_ROUNDS; round++)); do
        mapfile -t round_cases < <(build_round_order "${round}" "${previous_order}")
        previous_order="$(IFS='|'; printf '%s' "${round_cases[*]}")"
        position=0
        for spec in "${round_cases[@]}"; do
            position=$((position + 1))
            read -r runner dynamic need_timing async_scheduling <<<"${spec}"
            printf '%d\t%d\t%s\t%s\t%s\t%s\n' \
                "${round}" "${position}" "${runner}" "${dynamic}" \
                "${need_timing}" "${async_scheduling}"
        done
    done
}

if [[ "${DRY_RUN}" == "1" ]]; then
    print_dry_run
    exit 0
fi

# Only the model and fixed global test contract are checked here. Cases are
# validated individually immediately before they execute.
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
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
    --dataset variable

cpp_initialize_run performance-matrix
readonly MATRIX_ORDER_LOG="${CPP_RUN_DIR}/matrix_order.tsv"
readonly MATRIX_RESULT_LOG="${CPP_RUN_DIR}/matrix_results.tsv"
printf 'round\tposition\trunner\tcpp\tneed_timing\tasync_scheduling\n' \
    >"${MATRIX_ORDER_LOG}"
printf 'round\tposition\trunner\tcpp\tneed_timing\tasync_scheduling\texit_code\n' \
    >"${MATRIX_RESULT_LOG}"

failure_count=0
case_count=0
successful_rounds=0
attempted_rounds=0
previous_order="$(IFS='|'; printf '%s' "${MATRIX_CASES[*]}")"

for ((round = 1; round <= MAX_ROUNDS; round++)); do
    mapfile -t round_cases < <(build_round_order "${round}" "${previous_order}")
    previous_order="$(IFS='|'; printf '%s' "${round_cases[*]}")"
    attempted_rounds="${round}"
    round_failures=0
    position=0
    echo "CPP_PERFORMANCE_MATRIX_ROUND_START round=${round} cases=12 seed=${SHUFFLE_SEED}"

    for spec in "${round_cases[@]}"; do
        position=$((position + 1))
        case_count=$((case_count + 1))
        read -r runner dynamic need_timing async_scheduling <<<"${spec}"
        printf '%d\t%d\t%s\t%s\t%s\t%s\n' \
            "${round}" "${position}" "${runner}" "${dynamic}" \
            "${need_timing}" "${async_scheduling}" >>"${MATRIX_ORDER_LOG}"
        echo "CPP_PERFORMANCE_MATRIX_CASE round=${round} position=${position} runner=${runner} cpp=${dynamic} need_timing=${need_timing} async=${async_scheduling}"

        case_status=0
        env \
            CPP_GLOBAL_PREFLIGHT_DONE=1 \
            CPP_RUNNER="${runner}" \
            CPP_DYNAMIC="${dynamic}" \
            CPP_NEED_TIMING="${need_timing}" \
            CPP_ASYNC_SCHEDULING="${async_scheduling}" \
            CPP_MATRIX_ROUND="${round}" \
            CPP_MATRIX_POSITION="${position}" \
            CPP_EXECUTION_MODE=eager \
            CPP_PERF_DATASET=variable \
            CPP_PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE}" \
            CPP_TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
            CPP_AISBENCH_AUTO_TOOLS_ROOT="${AISBENCH_AUTO_TOOLS_ROOT}" \
            CPP_MODEL_CONFIG="${MODEL_CONFIG}" \
            CPP_SUITE_CONFIG="${SUITE_CONFIG}" \
            CPP_RUN_ID="${CPP_RUN_ID}" \
            CPP_RUN_DIR="${CPP_RUN_DIR}" \
            CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT}" \
            CPP_NPU_DEVICES="${NPU_DEVICES}" \
            "${CPP_ROOT}/scripts/run_perf_case.sh" || case_status=$?

        printf '%d\t%d\t%s\t%s\t%s\t%s\t%d\n' \
            "${round}" "${position}" "${runner}" "${dynamic}" \
            "${need_timing}" "${async_scheduling}" "${case_status}" \
            >>"${MATRIX_RESULT_LOG}"
        if [[ "${case_status}" -eq 78 ]]; then
            echo "CPP_PERFORMANCE_MATRIX_CONFIG_CONFLICT round=${round} position=${position}; stopping all remaining cases" >&2
            exit 78
        fi
        if [[ "${case_status}" -ne 0 ]]; then
            round_failures=$((round_failures + 1))
            failure_count=$((failure_count + 1))
        fi
    done

    python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
    if [[ "${round_failures}" -ne 0 ]]; then
        echo "CPP_PERFORMANCE_MATRIX_ROUND_FAILED round=${round} failures=${round_failures}; later rounds will not start" >&2
        break
    fi
    successful_rounds="${round}"
    echo "CPP_PERFORMANCE_MATRIX_ROUND_PASS round=${round} cases=12"
done

python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
echo "CPP_PERFORMANCE_MATRIX_DONE cases=${case_count} failures=${failure_count} successful_rounds=${successful_rounds} attempted_rounds=${attempted_rounds} artifacts=${CPP_RUN_DIR}"
[[ "${failure_count}" -eq 0 && "${successful_rounds}" -eq "${MAX_ROUNDS}" ]]
