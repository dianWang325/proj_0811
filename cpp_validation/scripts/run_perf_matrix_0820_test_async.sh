#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
readonly MODEL_CONFIG="${CPP_MODEL_CONFIG:-${CPP_ROOT}/configs/models/deepseek_v4_flash.json}"
readonly SUITE_CONFIG="${CPP_SUITE_CONFIG:-${CPP_ROOT}/configs/suites/performance.json}"
readonly MATRIX_FILE="${CPP_PERF_MATRIX_FILE:-${CPP_ROOT}/configs/matrices/performance.tsv}"

source "${CPP_ROOT}/scripts/lib/common.sh"
source "${CPP_ROOT}/scripts/lib/config.sh"
cpp_load_performance_config_defaults "${MODEL_CONFIG}" "${SUITE_CONFIG}"

readonly MODEL_PATH="${CPP_MODEL_PATH:-${CPP_PERF_MODEL_DEFAULTS[0]}}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-${CPP_PERF_MODEL_DEFAULTS[2]}}"
readonly MAX_MODEL_LEN="${CPP_MAX_MODEL_LEN:-${CPP_PERF_MODEL_DEFAULTS[3]}}"
readonly PIPELINE_PARALLEL_SIZE="${CPP_PIPELINE_PARALLEL_SIZE:-${CPP_PERF_MODEL_DEFAULTS[4]}}"
readonly TENSOR_PARALLEL_SIZE="${CPP_TENSOR_PARALLEL_SIZE:-${CPP_PERF_MODEL_DEFAULTS[5]}}"
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
readonly NEED_TIMING="${CPP_NEED_TIMING:-true}"
readonly AISBENCH_AUTO_TOOLS_ROOT="${CPP_AISBENCH_AUTO_TOOLS_ROOT:-${PROJECT_ROOT}/deps/aisbench_auto_tools_prefix}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly MAX_ROUNDS="${CPP_MATRIX_MAX_ROUNDS:-1}"
readonly SHUFFLE_AFTER_FIRST="${CPP_MATRIX_SHUFFLE_AFTER_FIRST:-0}"
readonly STOP_AFTER_FAILED_ROUND="${CPP_MATRIX_STOP_AFTER_FAILED_ROUND:-0}"
readonly RECORD_ORDER="${CPP_MATRIX_RECORD_ORDER:-0}"
readonly REPORT_ROUNDS="${CPP_MATRIX_REPORT_ROUNDS:-0}"
readonly SHUFFLE_SEED="${CPP_MATRIX_SHUFFLE_SEED:-20260820}"
readonly DRY_RUN="${CPP_MATRIX_DRY_RUN:-0}"
readonly GLOBAL_DATASET="${CPP_MATRIX_GLOBAL_DATASET:-}"
readonly CASE_COOLDOWN_SECONDS="${CPP_MATRIX_CASE_COOLDOWN_SECONDS:-0}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"

[[ "${MAX_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || \
    cpp_fail "CPP_MATRIX_MAX_ROUNDS must be a positive integer"
for flag_name in SHUFFLE_AFTER_FIRST STOP_AFTER_FAILED_ROUND RECORD_ORDER REPORT_ROUNDS DRY_RUN; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] || \
        cpp_fail "${flag_name} must be 0 or 1"
done
[[ "${NEED_TIMING}" == "true" || "${NEED_TIMING}" == "false" ]] || \
    cpp_fail "CPP_NEED_TIMING must be true or false"
[[ "${CASE_COOLDOWN_SECONDS}" =~ ^[0-9]+$ ]] || \
    cpp_fail "CPP_MATRIX_CASE_COOLDOWN_SECONDS must be a non-negative integer"
[[ -z "${GLOBAL_DATASET}" || "${GLOBAL_DATASET}" == "fixed" || \
   "${GLOBAL_DATASET}" == "variable" ]] || \
    cpp_fail "CPP_MATRIX_GLOBAL_DATASET must be fixed, variable, or empty"

declare -a MATRIX_CASES=()
load_matrix_cases() {
    local line trimmed
    if [[ -n "${CPP_MATRIX_INLINE_CASES:-}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="${line%$'\r'}"
            trimmed="${line#"${line%%[![:space:]]*}"}"
            [[ -n "${trimmed}" && "${trimmed}" != \#* ]] || continue
            MATRIX_CASES+=("${line}")
        done <<<"${CPP_MATRIX_INLINE_CASES}"
    else
        [[ -r "${MATRIX_FILE}" ]] || \
            cpp_fail "performance matrix is not readable: ${MATRIX_FILE}"
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="${line%$'\r'}"
            trimmed="${line#"${line%%[![:space:]]*}"}"
            [[ -n "${trimmed}" && "${trimmed}" != \#* ]] || continue
            MATRIX_CASES+=("${line}")
        done <"${MATRIX_FILE}"
    fi
    [[ "${#MATRIX_CASES[@]}" -gt 0 ]] || cpp_fail "performance matrix has no cases"
}

build_round_order() {
    local round="$1" previous_order="$2"
    if [[ "${round}" -eq 1 || "${SHUFFLE_AFTER_FIRST}" -eq 0 || \
          "${#MATRIX_CASES[@]}" -lt 2 ]]; then
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

parse_case_spec() {
    local spec="$1"
    read -r SPEC_RUNNER SPEC_DYNAMIC SPEC_EXECUTION_MODE SPEC_DATASET \
        SPEC_NEED_TIMING SPEC_ASYNC_SCHEDULING SPEC_EXTRA <<<"${spec}"
    [[ -n "${SPEC_RUNNER:-}" && -n "${SPEC_DYNAMIC:-}" && \
       -n "${SPEC_EXECUTION_MODE:-}" ]] || cpp_fail "invalid matrix row: ${spec}"
    [[ -z "${SPEC_EXTRA:-}" ]] || cpp_fail "invalid matrix row: ${spec}"
    SPEC_NEED_TIMING="${SPEC_NEED_TIMING:-${NEED_TIMING}}"
    SPEC_ASYNC_SCHEDULING="${SPEC_ASYNC_SCHEDULING:-0}"
    [[ "${SPEC_NEED_TIMING}" == "true" || "${SPEC_NEED_TIMING}" == "false" ]] || \
        cpp_fail "invalid need_timing in matrix row: ${spec}"
    [[ "${SPEC_ASYNC_SCHEDULING}" == "0" || \
       "${SPEC_ASYNC_SCHEDULING}" == "1" ]] || \
        cpp_fail "invalid async_scheduling in matrix row: ${spec}"
}

print_dry_run() {
    local round spec dataset previous_order
    local -a round_cases datasets
    previous_order="$(IFS='|'; printf '%s' "${MATRIX_CASES[*]}")"
    if [[ -n "${CPP_MATRIX_DRY_RUN_LABEL:-}" ]]; then
        echo "CPP_MATRIX_DRY_RUN ${CPP_MATRIX_DRY_RUN_LABEL}"
    else
        echo "CPP_MATRIX_DRY_RUN specs_per_round=${#MATRIX_CASES[@]} max_rounds=${MAX_ROUNDS} seed=${SHUFFLE_SEED}"
    fi
    printf 'round\tposition\trunner\tcpp\texecution_mode\tdataset\tneed_timing\tasync_scheduling\n'
    for ((round = 1; round <= MAX_ROUNDS; round++)); do
        mapfile -t round_cases < <(build_round_order "${round}" "${previous_order}")
        previous_order="$(IFS='|'; printf '%s' "${round_cases[*]}")"
        position=0
        for spec in "${round_cases[@]}"; do
            parse_case_spec "${spec}"
            if [[ -n "${SPEC_DATASET:-}" ]]; then
                datasets=("${SPEC_DATASET}")
            else
                datasets=(fixed variable)
            fi
            for dataset in "${datasets[@]}"; do
                position=$((position + 1))
                printf '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${round}" "${position}" "${SPEC_RUNNER}" \
                    "${SPEC_DYNAMIC}" "${SPEC_EXECUTION_MODE}" "${dataset}" \
                    "${SPEC_NEED_TIMING}" "${SPEC_ASYNC_SCHEDULING}"
            done
        done
    done
}

load_matrix_cases
if [[ "${DRY_RUN}" == "1" ]]; then
    print_dry_run
    exit 0
fi

# This checks only the selected model and global test contract. Matrix cases
# are deliberately not semantically prevalidated; each case validates itself.
global_preflight_args=(
    global
    --model-config "${MODEL_CONFIG}"
    --suite-config "${SUITE_CONFIG}"
    --artifact-root "${CPP_ARTIFACT_ROOT}"
    --project-root "${PROJECT_ROOT}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --tokenizer-mode "${TOKENIZER_MODE}"
    --tokenizer-trust-remote-code "${TOKENIZER_TRUST_REMOTE_CODE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --npu-devices "${NPU_DEVICES}"
    --quantization "${QUANTIZATION}"
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"
)
[[ -z "${GLOBAL_DATASET}" ]] || global_preflight_args+=(--dataset "${GLOBAL_DATASET}")
python3 "${CPP_ROOT}/configuration/performance.py" "${global_preflight_args[@]}"

cpp_initialize_run performance-matrix
if [[ "${RECORD_ORDER}" == "1" ]]; then
    readonly MATRIX_ORDER_LOG="${CPP_RUN_DIR}/matrix_order.tsv"
    readonly MATRIX_RESULT_LOG="${CPP_RUN_DIR}/matrix_results.tsv"
    printf 'round\tposition\trunner\tcpp\texecution_mode\tdataset\tneed_timing\tasync_scheduling\n' \
        >"${MATRIX_ORDER_LOG}"
    printf 'round\tposition\trunner\tcpp\texecution_mode\tdataset\tneed_timing\tasync_scheduling\texit_code\n' \
        >"${MATRIX_RESULT_LOG}"
fi

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
    [[ "${REPORT_ROUNDS}" == "0" ]] || \
        echo "CPP_PERFORMANCE_MATRIX_ROUND_START round=${round} specs=${#round_cases[@]} seed=${SHUFFLE_SEED}"

    for spec in "${round_cases[@]}"; do
        parse_case_spec "${spec}"
        if [[ -n "${SPEC_DATASET:-}" ]]; then
            [[ "${SPEC_DATASET}" == "fixed" || "${SPEC_DATASET}" == "variable" ]] || \
                cpp_fail "invalid performance dataset in matrix: ${SPEC_DATASET}"
            datasets=("${SPEC_DATASET}")
        else
            # Preserve compatibility with focused three-column matrix files.
            datasets=(fixed variable)
        fi

        for dataset in "${datasets[@]}"; do
            position=$((position + 1))
            case_count=$((case_count + 1))
            if [[ "${RECORD_ORDER}" == "1" ]]; then
                printf '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${round}" "${position}" "${SPEC_RUNNER}" \
                    "${SPEC_DYNAMIC}" "${SPEC_EXECUTION_MODE}" "${dataset}" \
                    "${SPEC_NEED_TIMING}" "${SPEC_ASYNC_SCHEDULING}" \
                    >>"${MATRIX_ORDER_LOG}"
                echo "CPP_PERFORMANCE_MATRIX_CASE round=${round} position=${position} runner=${SPEC_RUNNER} cpp=${SPEC_DYNAMIC} need_timing=${SPEC_NEED_TIMING} async=${SPEC_ASYNC_SCHEDULING}"
            fi

            case_env=(
                CPP_GLOBAL_PREFLIGHT_DONE=1
                CPP_NEED_TIMING="${SPEC_NEED_TIMING}"
                CPP_ASYNC_SCHEDULING="${SPEC_ASYNC_SCHEDULING}"
                CPP_AISBENCH_AUTO_TOOLS_ROOT="${AISBENCH_AUTO_TOOLS_ROOT}"
                CPP_RUNNER="${SPEC_RUNNER}"
                CPP_DYNAMIC="${SPEC_DYNAMIC}"
                CPP_EXECUTION_MODE="${SPEC_EXECUTION_MODE}"
                CPP_PERF_DATASET="${dataset}"
                CPP_MODEL_CONFIG="${MODEL_CONFIG}"
                CPP_SUITE_CONFIG="${SUITE_CONFIG}"
                CPP_RUN_ID="${CPP_RUN_ID}"
                CPP_RUN_DIR="${CPP_RUN_DIR}"
                CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT}"
                CPP_NPU_DEVICES="${NPU_DEVICES}"
                CPP_PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE}"
                CPP_TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}"
            )
            if [[ "${RECORD_ORDER}" == "1" ]]; then
                case_env+=(
                    CPP_MATRIX_ROUND="${round}"
                    CPP_MATRIX_POSITION="${position}"
                )
            fi

            case_status=0
            env "${case_env[@]}" "${CPP_ROOT}/scripts/run_perf_case.sh" || \
                case_status=$?
            if [[ "${RECORD_ORDER}" == "1" ]]; then
                printf '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n' \
                    "${round}" "${position}" "${SPEC_RUNNER}" \
                    "${SPEC_DYNAMIC}" "${SPEC_EXECUTION_MODE}" "${dataset}" \
                    "${SPEC_NEED_TIMING}" "${SPEC_ASYNC_SCHEDULING}" \
                    "${case_status}" >>"${MATRIX_RESULT_LOG}"
            fi
            if [[ "${case_status}" -eq 78 ]]; then
                if [[ "${RECORD_ORDER}" == "1" ]]; then
                    echo "CPP_PERFORMANCE_MATRIX_CONFIG_CONFLICT round=${round} position=${position}; stopping all remaining cases" >&2
                else
                    echo "CPP_PERFORMANCE_MATRIX_CONFIG_CONFLICT case=${case_count}; stopping remaining cases" >&2
                fi
                exit 78
            fi
            if [[ "${case_status}" -ne 0 ]]; then
                round_failures=$((round_failures + 1))
                failure_count=$((failure_count + 1))
            fi
            if [[ "${CASE_COOLDOWN_SECONDS}" -gt 0 ]]; then
                echo "CPP_PERFORMANCE_MATRIX_COOLDOWN seconds=${CASE_COOLDOWN_SECONDS} round=${round} position=${position}"
                sleep "${CASE_COOLDOWN_SECONDS}"
            fi
        done
    done

    if [[ "${REPORT_ROUNDS}" == "1" ]]; then
        python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
    fi
    if [[ "${round_failures}" -ne 0 ]]; then
        if [[ "${STOP_AFTER_FAILED_ROUND}" == "1" ]]; then
            echo "CPP_PERFORMANCE_MATRIX_ROUND_FAILED round=${round} failures=${round_failures}; later rounds will not start" >&2
            break
        fi
    else
        successful_rounds="${round}"
        [[ "${REPORT_ROUNDS}" == "0" ]] || \
            echo "CPP_PERFORMANCE_MATRIX_ROUND_PASS round=${round} cases=${position}"
    fi
done

python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
if [[ "${REPORT_ROUNDS}" == "1" ]]; then
    echo "CPP_PERFORMANCE_MATRIX_DONE cases=${case_count} failures=${failure_count} successful_rounds=${successful_rounds} attempted_rounds=${attempted_rounds} artifacts=${CPP_RUN_DIR}"
else
    echo "CPP_PERFORMANCE_MATRIX_DONE cases=${case_count} failures=${failure_count} artifacts=${CPP_RUN_DIR}"
fi
[[ "${failure_count}" -eq 0 && "${successful_rounds}" -eq "${MAX_ROUNDS}" ]]
