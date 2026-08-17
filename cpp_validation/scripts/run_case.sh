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
readonly MAX_MODEL_LEN="${CPP_MAX_MODEL_LEN:-${CPP_MODEL_DEFAULTS[2]}}"
readonly PIPELINE_PARALLEL_SIZE="${CPP_PIPELINE_PARALLEL_SIZE:-${CPP_MODEL_DEFAULTS[3]}}"
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-${MRV2_CPP_NPU_DEVICES:-8,9}}"
readonly SERVER_PORT="${CPP_PORT:-${MRV2_CPP_PORT:-18080}}"
readonly RUNNER="${CPP_RUNNER:-mrv2}"
readonly DYNAMIC="${CPP_DYNAMIC:-1}"
readonly EXECUTION_MODE="${CPP_EXECUTION_MODE:-eager}"
readonly REQUEST_MODE="${CPP_REQUEST_MODE:-${CPP_SUITE_DEFAULTS[0]}}"
readonly TOKEN_TARGETS="${CPP_TOKEN_TARGETS:-${CPP_SUITE_DEFAULTS[1]}}"
readonly MAX_OUTPUT_TOKENS="${CPP_MAX_OUTPUT_TOKENS:-${CPP_SUITE_DEFAULTS[2]}}"
readonly MAX_NUM_BATCHED_TOKENS="${CPP_MAX_NUM_BATCHED_TOKENS:-${CPP_SUITE_DEFAULTS[3]}}"
readonly MAX_NUM_SEQS="${CPP_MAX_NUM_SEQS:-${CPP_SUITE_DEFAULTS[4]}}"
readonly DATA_GENERATOR="${CPP_DATA_GENERATOR:-${CPP_SUITE_DEFAULTS[5]}}"
readonly AISBENCH_AUTO_TOOLS_ROOT="${CPP_AISBENCH_AUTO_TOOLS_ROOT:-/home/w00985415/tools/aisbench_auto_tools_prefix}"
readonly REQUEST_REPEATS="${CPP_REQUEST_REPEATS:-1}"
readonly MAX_FIT_CHUNK="${CPP_MAX_FIT_CHUNK:-0}"
readonly STARTUP_TIMEOUT="${CPP_STARTUP_TIMEOUT:-1200}"
readonly HCCL_PORT_RANGE="${CPP_HCCL_PORT_RANGE:-17000-17100}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly TOKEN_TARGET_COUNT="$(awk -F, '{print NF}' <<<"${TOKEN_TARGETS}")"
if [[ "${REQUEST_MODE}" == "both" ]]; then
    readonly REQUEST_PHASE_COUNT=2
else
    readonly REQUEST_PHASE_COUNT=1
fi
readonly EXPECTED_REQUEST_COUNT="$((TOKEN_TARGET_COUNT * REQUEST_REPEATS * REQUEST_PHASE_COUNT))"
readonly CASE_ID="${RUNNER}_$([[ "${DYNAMIC}" == 1 ]] && echo dynamic || echo static)_${EXECUTION_MODE}_${REQUEST_MODE}_${DATA_GENERATOR}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"
source "${CPP_ROOT}/scripts/lib/server.sh"

case_stage="initializing"
case_started_at="$(cpp_utc_timestamp)"
case_dir=""
server_pid_file=""
owns_run=0

cleanup() {
    local status=$? stop_status=0 final_state="passed"
    trap - EXIT INT TERM
    cpp_stop_server || stop_status=$?
    [[ "${status}" -ne 0 ]] || status="${stop_status}"
    [[ "${status}" -eq 0 ]] || final_state="failed"
    if [[ -n "${server_pid_file}" ]]; then
        rm -f "${server_pid_file}"
    fi
    if [[ -n "${case_dir}" ]]; then
        cpp_write_case_status "${case_dir}/status.json" "${final_state}" "${status}" \
            "${case_stage}" "${case_started_at}"
    fi
    if [[ "${owns_run}" -eq 1 && -n "${CPP_RUN_DIR:-}" ]]; then
        python3 "${CPP_ROOT}/analysis/summarize_run.py" "${CPP_RUN_DIR}" || true
    fi
    exit "${status}"
}

trap cleanup EXIT INT TERM

cpp_validate_server_inputs
[[ "${DATA_GENERATOR}" == "aisbench" || "${DATA_GENERATOR}" == "script" ]] || \
    cpp_fail "CPP_DATA_GENERATOR must be aisbench or script"
if [[ "${DATA_GENERATOR}" == "aisbench" ]]; then
    [[ -r "${AISBENCH_AUTO_TOOLS_ROOT}/generate_dataset.py" ]] || \
        cpp_fail "aisbench_auto_tools_prefix is not installed at ${AISBENCH_AUTO_TOOLS_ROOT}"
fi

if [[ -z "${CPP_RUN_DIR:-}" ]]; then
    cpp_initialize_run functional
    owns_run=1
else
    [[ -f "${CPP_RUN_DIR}/run.json" ]] || cpp_fail "CPP_RUN_DIR has no run.json: ${CPP_RUN_DIR}"
fi

case_dir="${CPP_RUN_DIR}/cases/${CASE_ID}"
cpp_initialize_case "${case_dir}" "${CASE_ID}" "${RUNNER}" "${DYNAMIC}" \
    "${EXECUTION_MODE}" "${REQUEST_MODE}" "${TOKEN_TARGETS}" \
    "${REQUEST_REPEATS}" "${EXPECTED_REQUEST_COUNT}" "${MAX_FIT_CHUNK}" \
    "${DATA_GENERATOR}"
server_pid_file="${case_dir}/server.pid"

dataset_path=""
if [[ "${DATA_GENERATOR}" == "aisbench" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    case_stage="generating_dataset"
    dataset_path="${case_dir}/raw/dataset.jsonl"
    python3 "${CPP_ROOT}/workloads/data_generation.py" \
        --backend aisbench \
        --model-path "${MODEL_PATH}" \
        --lengths "${TOKEN_TARGETS}" \
        --output-tokens "${MAX_OUTPUT_TOKENS}" \
        --seed 811 \
        --aisbench-auto-tools-root "${AISBENCH_AUTO_TOOLS_ROOT}" \
        --output "${dataset_path}" \
        --metadata-output "${case_dir}/raw/dataset_metadata.json" \
        > >(tee "${case_dir}/logs/dataset_generation.log") 2>&1
fi

case_stage="starting_server"
echo "Starting ${CASE_ID} on physical NPUs ${NPU_DEVICES}; run=${CPP_RUN_ID}"
cpp_start_server "${case_dir}/logs/server.log" "${case_dir}/compiler"
printf '%s\n' "${CPP_SERVER_PID}" >"${server_pid_file}"

case_stage="waiting_for_health"
cpp_wait_until_ready "${case_dir}/logs/server.log" || cpp_fail "server startup failed"

case_stage="running_workload"
CPP_MODEL_PATH="${MODEL_PATH}" \
CPP_MODEL_NAME="${MODEL_NAME}" \
CPP_PORT="${SERVER_PORT}" \
CPP_REQUEST_MODE="${REQUEST_MODE}" \
CPP_TOKEN_TARGETS="${TOKEN_TARGETS}" \
CPP_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}" \
CPP_REQUEST_REPEATS="${REQUEST_REPEATS}" \
CPP_DATA_GENERATOR="${DATA_GENERATOR}" \
CPP_DATASET_PATH="${dataset_path}" \
CPP_RESULT_PATH="${case_dir}/raw/requests.json" \
    python3 "${CPP_ROOT}/workloads/functional/long_context.py" \
    > >(tee "${case_dir}/logs/client.log") 2>&1

case_stage="validating_results"
python3 "${CPP_ROOT}/validators/functional/cpp_trace.py" \
    "${case_dir}/logs/server.log" \
    --runner "${RUNNER}" \
    --dynamic "${DYNAMIC}" \
    --request-mode "${REQUEST_MODE}" \
    --events-output "${case_dir}/raw/cpp_trace.jsonl" \
    --result-output "${case_dir}/results/functional_result.json" \
    | tee -a "${case_dir}/logs/client.log"

case_stage="completed"
echo "CPP_FUNCTIONAL_PASS case=${CASE_ID} run=${CPP_RUN_ID} artifacts=${case_dir}"
