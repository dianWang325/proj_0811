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
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-8,9,10,11,12,13,14,15}"
readonly NEED_TIMING="${CPP_NEED_TIMING:-true}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"

[[ -r "${MATRIX_FILE}" ]] || cpp_fail "performance matrix is not readable: ${MATRIX_FILE}"
[[ "${NEED_TIMING}" == "true" || "${NEED_TIMING}" == "false" ]] || \
    cpp_fail "CPP_NEED_TIMING must be true or false"
cpp_initialize_run performance-matrix

failure_count=0
case_count=0
while read -r runner dynamic execution_mode dataset extra; do
    [[ -n "${runner:-}" && "${runner}" != \#* ]] || continue
    [[ -z "${extra:-}" ]] || \
        cpp_fail "invalid matrix row: ${runner} ${dynamic} ${execution_mode} ${dataset} ${extra}"
    if [[ -n "${dataset:-}" ]]; then
        [[ "${dataset}" == "fixed" || "${dataset}" == "variable" ]] || \
            cpp_fail "invalid performance dataset in matrix: ${dataset}"
        datasets=("${dataset}")
    else
        # Preserve compatibility with focused three-column matrix files.
        datasets=(fixed variable)
    fi
    for dataset in "${datasets[@]}"; do
        case_count=$((case_count + 1))
        if ! env \
            CPP_NEED_TIMING="${NEED_TIMING}" \
            CPP_RUNNER="${runner}" \
            CPP_DYNAMIC="${dynamic}" \
            CPP_EXECUTION_MODE="${execution_mode}" \
            CPP_PERF_DATASET="${dataset}" \
            CPP_MODEL_CONFIG="${MODEL_CONFIG}" \
            CPP_SUITE_CONFIG="${SUITE_CONFIG}" \
            CPP_RUN_ID="${CPP_RUN_ID}" \
            CPP_RUN_DIR="${CPP_RUN_DIR}" \
            CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT}" \
            CPP_NPU_DEVICES="${NPU_DEVICES}" \
            "${CPP_ROOT}/scripts/run_perf_case.sh"; then
            failure_count=$((failure_count + 1))
        fi
    done
done <"${MATRIX_FILE}"

python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
echo "CPP_PERFORMANCE_MATRIX_DONE cases=${case_count} failures=${failure_count} artifacts=${CPP_RUN_DIR}"
[[ "${failure_count}" -eq 0 ]]
