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

source "${CPP_ROOT}/scripts/lib/artifacts.sh"

[[ -r "${MATRIX_FILE}" ]] || cpp_fail "performance matrix is not readable: ${MATRIX_FILE}"
[[ "${NEED_TIMING}" == "true" || "${NEED_TIMING}" == "false" ]] || \
    cpp_fail "CPP_NEED_TIMING must be true or false"

# This checks only the selected model and global test contract. Matrix rows are
# deliberately not parsed or prevalidated here; each case validates itself.
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
        case_status=0
        env \
            CPP_GLOBAL_PREFLIGHT_DONE=1 \
            CPP_NEED_TIMING="${NEED_TIMING}" \
            CPP_AISBENCH_AUTO_TOOLS_ROOT="${AISBENCH_AUTO_TOOLS_ROOT}" \
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
            "${CPP_ROOT}/scripts/run_perf_case.sh" || case_status=$?
        if [[ "${case_status}" -eq 78 ]]; then
            echo "CPP_PERFORMANCE_MATRIX_CONFIG_CONFLICT case=${case_count}; stopping remaining cases" >&2
            exit 78
        fi
        if [[ "${case_status}" -ne 0 ]]; then
            failure_count=$((failure_count + 1))
        fi
    done
done <"${MATRIX_FILE}"

python3 "${CPP_ROOT}/analysis/summarize_performance.py" "${CPP_RUN_DIR}" || true
echo "CPP_PERFORMANCE_MATRIX_DONE cases=${case_count} failures=${failure_count} artifacts=${CPP_RUN_DIR}"
[[ "${failure_count}" -eq 0 ]]
