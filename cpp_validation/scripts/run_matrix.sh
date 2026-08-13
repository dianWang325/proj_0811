#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
readonly MODEL_CONFIG="${CPP_MODEL_CONFIG:-${CPP_ROOT}/configs/models/qwen3_30b_a3b.json}"
readonly SUITE_CONFIG="${CPP_SUITE_CONFIG:-${CPP_ROOT}/configs/suites/functional.json}"

source "${CPP_ROOT}/scripts/lib/common.sh"
source "${CPP_ROOT}/scripts/lib/config.sh"
cpp_load_config_defaults "${MODEL_CONFIG}" "${SUITE_CONFIG}"

readonly MODEL_PATH="${CPP_MODEL_PATH:-${CPP_MODEL_DEFAULTS[0]}}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-${CPP_MODEL_DEFAULTS[1]}}"
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-${MRV2_CPP_NPU_DEVICES:-8,9}}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly REQUEST_MODE="${CPP_REQUEST_MODE:-${CPP_SUITE_DEFAULTS[0]}}"
readonly MATRIX_FILE="${CPP_MATRIX_FILE:-${CPP_ROOT}/configs/matrices/compatibility.tsv}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"

[[ -r "${MATRIX_FILE}" ]] || cpp_fail "matrix is not readable: ${MATRIX_FILE}"
[[ "${NPU_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || cpp_fail "CPP_NPU_DEVICES must contain two physical IDs"
cpp_initialize_run functional-matrix

failure_count=0
case_count=0
while read -r runner dynamic execution_mode extra; do
    [[ -n "${runner:-}" && "${runner}" != \#* ]] || continue
    [[ -z "${extra:-}" ]] || cpp_fail "invalid matrix row: ${runner} ${dynamic} ${execution_mode} ${extra}"
    case_count=$((case_count + 1))
    if ! env CPP_RUNNER="${runner}" \
        CPP_DYNAMIC="${dynamic}" \
        CPP_EXECUTION_MODE="${execution_mode}" \
        CPP_REQUEST_MODE="${REQUEST_MODE}" \
        CPP_RUN_ID="${CPP_RUN_ID}" \
        CPP_RUN_DIR="${CPP_RUN_DIR}" \
        CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT}" \
        "${CPP_ROOT}/scripts/run_case.sh"; then
        failure_count=$((failure_count + 1))
    fi
done <"${MATRIX_FILE}"

python3 "${CPP_ROOT}/analysis/summarize_run.py" "${CPP_RUN_DIR}" || true
echo "CPP_REGRESSION_MATRIX_DONE cases=${case_count} failures=${failure_count} artifacts=${CPP_RUN_DIR}"
[[ "${failure_count}" -eq 0 ]]
