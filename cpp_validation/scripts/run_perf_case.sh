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
readonly FALLBACK_MODEL_PATH="${CPP_FALLBACK_MODEL_PATH:-${CPP_PERF_MODEL_DEFAULTS[1]}}"
readonly MODEL_NAME="${CPP_MODEL_NAME:-${CPP_PERF_MODEL_DEFAULTS[2]}}"
readonly MAX_MODEL_LEN="${CPP_MAX_MODEL_LEN:-${CPP_PERF_MODEL_DEFAULTS[3]}}"
readonly PIPELINE_PARALLEL_SIZE="${CPP_PIPELINE_PARALLEL_SIZE:-${CPP_PERF_MODEL_DEFAULTS[4]}}"
readonly TENSOR_PARALLEL_SIZE="${CPP_TENSOR_PARALLEL_SIZE:-${CPP_PERF_MODEL_DEFAULTS[5]}}"
readonly WARMUP_COUNT="${CPP_WARMUP_COUNT:-${CPP_PERF_SUITE_DEFAULTS[0]}}"
readonly MAX_OUTPUT_TOKENS="${CPP_MAX_OUTPUT_TOKENS:-${CPP_PERF_SUITE_DEFAULTS[1]}}"
readonly DATA_GENERATOR="${CPP_DATA_GENERATOR:-${CPP_PERF_SUITE_DEFAULTS[15]}}"
readonly MANUAL_WARMUP_ENABLED="${CPP_MANUAL_WARMUP_ENABLED:-${CPP_PERF_SUITE_DEFAULTS[16]}}"
readonly MANUAL_WARMUP_INPUT_TOKENS="${CPP_MANUAL_WARMUP_INPUT_TOKENS:-${CPP_PERF_SUITE_DEFAULTS[17]}}"
readonly MANUAL_WARMUP_OUTPUT_TOKENS="${CPP_MANUAL_WARMUP_OUTPUT_TOKENS:-${CPP_PERF_SUITE_DEFAULTS[18]}}"
readonly MANUAL_WARMUP_COUNT="${CPP_MANUAL_WARMUP_COUNT:-${CPP_PERF_SUITE_DEFAULTS[19]}}"
readonly MANUAL_WARMUP_CONCURRENCY="${CPP_MANUAL_WARMUP_CONCURRENCY:-${CPP_PERF_SUITE_DEFAULTS[20]}}"
readonly MANUAL_WARMUP_REQUEST_RATE="${CPP_MANUAL_WARMUP_REQUEST_RATE:-${CPP_PERF_SUITE_DEFAULTS[21]}}"
readonly CPP_SMOOTH_FACTOR="${CPP_SMOOTH_FACTOR:-${CPP_PERF_SUITE_DEFAULTS[22]}}"
readonly CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST="${CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST:-${CPP_PERF_SUITE_DEFAULTS[23]}}"
readonly POST_MANUAL_REWARM_COUNT="${CPP_POST_MANUAL_REWARM_COUNT:-${CPP_PERF_SUITE_DEFAULTS[24]}}"
readonly AISBENCH_AUTO_TOOLS_ROOT="${CPP_AISBENCH_AUTO_TOOLS_ROOT:-/home/w00985415/tools/aisbench_auto_tools_prefix}"
readonly NPU_DEVICES="${CPP_NPU_DEVICES:-8,9,10,11,12,13,14,15}"
readonly SERVER_PORT="${CPP_PORT:-18080}"
readonly RUNNER="${CPP_RUNNER:-mrv2}"
readonly DYNAMIC="${CPP_DYNAMIC:-1}"
readonly EXECUTION_MODE="${CPP_EXECUTION_MODE:-eager}"
readonly PERF_DATASET="${CPP_PERF_DATASET:-fixed}"
if [[ "${PERF_DATASET}" == "fixed" ]]; then
    readonly REQUEST_COUNT="${CPP_REQUEST_COUNT:-${CPP_PERF_SUITE_DEFAULTS[3]}}"
    readonly CONCURRENCY="${CPP_PERF_CONCURRENCY:-${CPP_PERF_SUITE_DEFAULTS[4]}}"
    readonly REQUEST_RATE="${CPP_PERF_REQUEST_RATE:-${CPP_PERF_SUITE_DEFAULTS[5]}}"
    readonly MAX_NUM_BATCHED_TOKENS="${CPP_MAX_NUM_BATCHED_TOKENS:-${CPP_PERF_SUITE_DEFAULTS[6]}}"
    readonly PREFIX_CACHE_ENABLED="${CPP_PREFIX_CACHE_ENABLED:-${CPP_PERF_SUITE_DEFAULTS[25]}}"
else
    readonly REQUEST_COUNT="${CPP_REQUEST_COUNT:-${CPP_PERF_SUITE_DEFAULTS[10]}}"
    readonly CONCURRENCY="${CPP_PERF_CONCURRENCY:-${CPP_PERF_SUITE_DEFAULTS[11]}}"
    readonly REQUEST_RATE="${CPP_PERF_REQUEST_RATE:-${CPP_PERF_SUITE_DEFAULTS[12]}}"
    readonly MAX_NUM_BATCHED_TOKENS="${CPP_MAX_NUM_BATCHED_TOKENS:-${CPP_PERF_SUITE_DEFAULTS[13]}}"
    readonly PREFIX_CACHE_ENABLED="${CPP_PREFIX_CACHE_ENABLED:-${CPP_PERF_SUITE_DEFAULTS[26]}}"
fi
readonly PREFIX_REPEAT_RATE="${CPP_PREFIX_REPEAT_RATE:-${CPP_PERF_SUITE_DEFAULTS[27]}}"
readonly PREFIX_TEST="${CPP_PREFIX_TEST:-${CPP_PERF_SUITE_DEFAULTS[28]}}"
readonly STARTUP_TIMEOUT="${CPP_STARTUP_TIMEOUT:-3600}"
readonly REQUEST_TIMEOUT="${CPP_REQUEST_TIMEOUT:-7200}"
readonly HCCL_PORT_RANGE="${CPP_HCCL_PORT_RANGE:-17000-17100}"
readonly CPP_ARTIFACT_ROOT="${CPP_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cpp}"
readonly CASE_ID="${RUNNER}_cpp${DYNAMIC}_${EXECUTION_MODE}_${PERF_DATASET}_${DATA_GENERATOR}_pp${PIPELINE_PARALLEL_SIZE}_tp${TENSOR_PARALLEL_SIZE}"

source "${CPP_ROOT}/scripts/lib/artifacts.sh"
source "${CPP_ROOT}/scripts/lib/server.sh"
source "${CPP_ROOT}/scripts/lib/performance_server.sh"

case_stage="initializing"
case_started_at="$(cpp_utc_timestamp)"
case_dir=""
server_pid_file=""
owns_run=0

cleanup() {
    local status=$? stop_status=0 final_state="passed"
    trap - EXIT INT TERM
    cpp_perf_stop_server || stop_status=$?
    [[ "${status}" -ne 0 ]] || status="${stop_status}"
    [[ "${status}" -eq 0 ]] || final_state="failed"
    [[ -z "${server_pid_file}" ]] || rm -f "${server_pid_file}"
    if [[ -n "${case_dir}" ]]; then
        cpp_write_case_status "${case_dir}/status.json" "${final_state}" "${status}" \
            "${case_stage}" "${case_started_at}"
    fi
    if [[ "${owns_run}" -eq 1 && -n "${CPP_RUN_DIR:-}" ]]; then
        python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

cpp_perf_validate_server_inputs
if [[ "${PERF_DATASET}" == "fixed" ]]; then
    [[ "${REQUEST_COUNT}" -eq 5 ]] || cpp_fail "fixed performance request count must be 5"
    [[ "${MAX_NUM_BATCHED_TOKENS}" -eq 32768 ]] || \
        cpp_fail "fixed max_num_batched_tokens must be 32768"
else
    [[ "${REQUEST_COUNT}" -eq 64 ]] || cpp_fail "variable performance request count must be 64"
    [[ "${MAX_NUM_BATCHED_TOKENS}" -eq 20480 ]] || \
        cpp_fail "variable max_num_batched_tokens must be 20480"
fi
[[ "${MAX_OUTPUT_TOKENS}" -eq 1 ]] || cpp_fail "performance output length must be 1"
[[ "${WARMUP_COUNT}" -gt 0 ]] || cpp_fail "warmup count must be positive"
[[ "${DATA_GENERATOR}" == "aisbench" || "${DATA_GENERATOR}" == "script" ]] || \
    cpp_fail "CPP_DATA_GENERATOR must be aisbench or script"
[[ "${MANUAL_WARMUP_ENABLED}" == "0" || "${MANUAL_WARMUP_ENABLED}" == "1" ]] || \
    cpp_fail "CPP_MANUAL_WARMUP_ENABLED must be 0 or 1"
[[ "${PREFIX_CACHE_ENABLED}" == "0" || "${PREFIX_CACHE_ENABLED}" == "1" ]] || \
    cpp_fail "CPP_PREFIX_CACHE_ENABLED must be 0 or 1"
[[ "${CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST}" == "0" || \
   "${CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST}" == "1" ]] || \
    cpp_fail "CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST must be 0 or 1"
if [[ "${MANUAL_WARMUP_ENABLED}" == "1" ]]; then
    [[ -r "${AISBENCH_AUTO_TOOLS_ROOT}/aisbench_test.py" ]] || \
        cpp_fail "aisbench_auto_tools_prefix is not installed at ${AISBENCH_AUTO_TOOLS_ROOT}"
    [[ "${MANUAL_WARMUP_INPUT_TOKENS}" -gt 0 ]] || cpp_fail "manual warmup input length must be positive"
    [[ "${MANUAL_WARMUP_OUTPUT_TOKENS}" -gt 0 ]] || cpp_fail "manual warmup output length must be positive"
    [[ "${MANUAL_WARMUP_COUNT}" -gt 0 ]] || cpp_fail "manual warmup request count must be positive"
    [[ "${MANUAL_WARMUP_CONCURRENCY}" -gt 0 ]] || cpp_fail "manual warmup concurrency must be positive"
fi

if [[ -z "${CPP_RUN_DIR:-}" ]]; then
    cpp_initialize_run performance
    owns_run=1
else
    [[ -f "${CPP_RUN_DIR}/run.json" ]] || cpp_fail "CPP_RUN_DIR has no run.json: ${CPP_RUN_DIR}"
fi

case_dir="${CPP_RUN_DIR}/cases/${CASE_ID}"
cpp_initialize_performance_case "${case_dir}" "${CASE_ID}"
server_pid_file="${case_dir}/server.pid"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
case_stage="generating_dataset"
mkdir -p "${CPP_RUN_DIR}/datasets"
dataset_variant="prefix${PREFIX_CACHE_ENABLED}_${PREFIX_REPEAT_RATE//%/pct}"
dataset_cache="${CPP_RUN_DIR}/datasets/${PERF_DATASET}_${DATA_GENERATOR}_${dataset_variant}.jsonl"
dataset_metadata_cache="${CPP_RUN_DIR}/datasets/${PERF_DATASET}_${DATA_GENERATOR}_${dataset_variant}.json"
if [[ ! -f "${dataset_cache}" || ! -f "${dataset_metadata_cache}" ]]; then
    generation_args=(
        --backend "${DATA_GENERATOR}"
        --model-path "${MODEL_PATH}"
        --performance-dataset "${PERF_DATASET}"
        --output-tokens "${MAX_OUTPUT_TOKENS}"
        --seed "${CPP_PERF_SUITE_DEFAULTS[14]}"
        --aisbench-auto-tools-root "${AISBENCH_AUTO_TOOLS_ROOT}"
        --output "${dataset_cache}"
        --metadata-output "${dataset_metadata_cache}"
    )
    if [[ "${PREFIX_CACHE_ENABLED}" == "1" ]]; then
        generation_args+=(--prefix-repeat-rate "${PREFIX_REPEAT_RATE}")
        [[ "${PREFIX_TEST}" == "1" ]] && generation_args+=(--prefix-test)
    fi
    python3 "${CPP_ROOT}/workloads/data_generation.py" "${generation_args[@]}" \
        > >(tee "${case_dir}/logs/dataset_generation.log") 2>&1
else
    echo "CPP_DATASET_REUSED backend=${DATA_GENERATOR} dataset=${PERF_DATASET} source=${dataset_cache}" \
        | tee "${case_dir}/logs/dataset_generation.log"
fi
ln -s "../../../datasets/${PERF_DATASET}_${DATA_GENERATOR}_${dataset_variant}.jsonl" \
    "${case_dir}/raw/dataset.jsonl"
ln -s "../../../datasets/${PERF_DATASET}_${DATA_GENERATOR}_${dataset_variant}.json" \
    "${case_dir}/raw/dataset_metadata.json"

case_stage="starting_server"
echo "Starting ${CASE_ID} on physical NPUs ${NPU_DEVICES}; run=${CPP_RUN_ID}"
cpp_perf_start_server "${case_dir}/logs/server.log" "${case_dir}/compiler"
printf '%s\n' "${CPP_SERVER_PID}" >"${server_pid_file}"

case_stage="waiting_for_health"
cpp_perf_wait_until_ready "${case_dir}/logs/server.log" || \
    cpp_fail "server startup failed; fallback model available at ${FALLBACK_MODEL_PATH}"

run_distribution_warmup() {
    local count="$1" output_name="$2" log_name="$3"
    python3 "${CPP_ROOT}/workloads/performance/prepare.py" \
        --mode warmup \
        --dataset "${PERF_DATASET}" \
        --model-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --port "${SERVER_PORT}" \
        --count "${count}" \
        --concurrency "${CONCURRENCY}" \
        --timeout "${REQUEST_TIMEOUT}" \
        --dataset-path "${case_dir}/raw/dataset.jsonl" \
        --output "${case_dir}/raw/${output_name}" \
        > >(tee "${case_dir}/logs/${log_name}") 2>&1
}

# CPP learns from a bounded number of the first timed prefill steps. Feed it
# the target distribution before the fixed 128K hardware warmup; otherwise the
# manual warmup can consume the timing budget and leave variable cases fitted
# to the wrong distribution.
if [[ "${DYNAMIC}" == "1" && \
      "${CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST}" == "1" ]]; then
    case_stage="running_cpp_online_calibration"
    run_distribution_warmup "${WARMUP_COUNT}" \
        "cpp_online_calibration.json" "cpp_online_calibration.log"
fi

if [[ "${MANUAL_WARMUP_ENABLED}" == "1" ]]; then
    case_stage="running_manual_warmup"
    python3 "${CPP_ROOT}/workloads/aisbench_auto_tools.py" \
        --tool-root "${AISBENCH_AUTO_TOOLS_ROOT}" \
        --workspace "${case_dir}/raw/aisbench_manual_warmup" \
        --model-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --port "${SERVER_PORT}" \
        --input-len "${MANUAL_WARMUP_INPUT_TOKENS}" \
        --output-len "${MANUAL_WARMUP_OUTPUT_TOKENS}" \
        --data-num "${MANUAL_WARMUP_COUNT}" \
        --concurrency "${MANUAL_WARMUP_CONCURRENCY}" \
        --request-rate "${MANUAL_WARMUP_REQUEST_RATE}" \
        --metadata-output "${case_dir}/raw/manual_warmup.json" \
        > >(tee "${case_dir}/logs/manual_warmup.log") 2>&1
fi

if [[ "${DYNAMIC}" == "1" && \
      "${CPP_CALIBRATION_SAME_DISTRIBUTION_FIRST}" == "1" ]]; then
    if [[ "${POST_MANUAL_REWARM_COUNT}" -gt 0 ]]; then
        case_stage="running_post_manual_rewarm"
        run_distribution_warmup "${POST_MANUAL_REWARM_COUNT}" \
            "warmup.json" "warmup.log"
    fi
else
    case_stage="running_warmup"
    run_distribution_warmup "${WARMUP_COUNT}" "warmup.json" "warmup.log"
fi

if [[ "${PREFIX_CACHE_ENABLED}" == "1" && "${PREFIX_TEST}" == "1" ]]; then
    case_stage="running_prefix_prime"
    python3 "${CPP_ROOT}/workloads/performance/prepare.py" \
        --mode prefix-prime \
        --dataset "${PERF_DATASET}" \
        --model-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --port "${SERVER_PORT}" \
        --timeout "${REQUEST_TIMEOUT}" \
        --dataset-path "${case_dir}/raw/dataset.jsonl" \
        --dataset-metadata "${case_dir}/raw/dataset_metadata.json" \
        --output "${case_dir}/raw/prefix_prime.json" \
        > >(tee "${case_dir}/logs/prefix_prime.log") 2>&1
fi

if [[ "${EXECUTION_MODE}" == "graph" ]]; then
    case_stage="running_graph_probe"
    python3 "${CPP_ROOT}/workloads/performance/prepare.py" \
        --mode graph-probe \
        --dataset "${PERF_DATASET}" \
        --model-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --port "${SERVER_PORT}" \
        --timeout "${REQUEST_TIMEOUT}" \
        --server-log "${case_dir}/logs/server.log" \
        --output "${case_dir}/raw/graph_probe.json" \
        > >(tee "${case_dir}/logs/graph_probe.log") 2>&1
fi

case_stage="running_aisbench"
python3 "${CPP_ROOT}/workloads/performance/render_aisbench_config.py" \
    --template "${CPP_ROOT}/configs/aisbench/performance.py" \
    --output "${case_dir}/raw/aisbench_config.py" \
    --dataset "${PERF_DATASET}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --port "${SERVER_PORT}" \
    --concurrency "${CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --dataset-path "${case_dir}/raw/dataset.jsonl"
python3 "${CPP_ROOT}/workloads/performance/run_aisbench.py" \
    "${case_dir}/raw/aisbench_config.py" \
    --summarizer default_perf \
    --mode perf \
    --num-prompts "${REQUEST_COUNT}" \
    --num-warmups 0 \
    --work-dir "${case_dir}/raw/aisbench" \
    > >(tee "${case_dir}/logs/aisbench.log") 2>&1

case_stage="validating_results"
validator_args=(
    --aisbench-root "${case_dir}/raw/aisbench"
    --server-log "${case_dir}/logs/server.log"
    --dataset "${PERF_DATASET}"
    --runner "${RUNNER}"
    --dynamic "${DYNAMIC}"
    --execution-mode "${EXECUTION_MODE}"
    --pp-size "${PIPELINE_PARALLEL_SIZE}"
    --tp-size "${TENSOR_PARALLEL_SIZE}"
    --data-generator "${DATA_GENERATOR}"
    --dataset-metadata "${case_dir}/raw/dataset_metadata.json"
    --output "${case_dir}/results/performance_result.json"
)
if [[ "${MANUAL_WARMUP_ENABLED}" == "1" ]]; then
    validator_args+=(--manual-warmup "${case_dir}/raw/manual_warmup.json")
fi
if [[ "${PREFIX_CACHE_ENABLED}" == "1" && "${PREFIX_TEST}" == "1" ]]; then
    validator_args+=(--prefix-prime "${case_dir}/raw/prefix_prime.json")
fi
if [[ "${EXECUTION_MODE}" == "graph" ]]; then
    validator_args+=(--graph-probe "${case_dir}/raw/graph_probe.json")
fi
python3 "${CPP_ROOT}/validators/performance/result.py" "${validator_args[@]}" \
    | tee -a "${case_dir}/logs/aisbench.log"

case_stage="completed"
echo "CPP_PERFORMANCE_PASS case=${CASE_ID} run=${CPP_RUN_ID} artifacts=${case_dir}"
