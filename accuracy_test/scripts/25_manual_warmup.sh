#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

run_id=""
result_dir=""
while (($#)); do
    case "$1" in
        --run-id)
            [[ $# -ge 2 && "$2" =~ ^[A-Za-z0-9_.-]+$ ]] || \
                accuracy_fail "invalid --run-id"
            run_id="$2"
            shift 2
            ;;
        --result-dir)
            [[ $# -ge 2 && -n "$2" ]] || accuracy_fail "--result-dir requires a path"
            result_dir="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --run-id ID --result-dir DIR"
            exit 0
            ;;
        *) accuracy_fail "unknown argument: $1" ;;
    esac
done

[[ -n "${run_id}" ]] || accuracy_fail "--run-id is required"
[[ -n "${result_dir}" ]] || accuracy_fail "--result-dir is required"

state_file="${ACCURACY_ROOT}/state/service.env"
[[ -r "${state_file}" ]] || \
    accuracy_fail "missing service state: start the service with 10_start_service.sh"
# shellcheck source=/dev/null
source "${state_file}"

if [[ "${ACCURACY_CPP_ENABLED}" != "1" ]]; then
    accuracy_info "manual fixed-length warmup skipped because CPP is disabled"
    exit 0
fi

for name in \
    ACCURACY_MANUAL_WARMUP_INPUT_LEN \
    ACCURACY_MANUAL_WARMUP_OUTPUT_LEN \
    ACCURACY_MANUAL_WARMUP_REQUESTS \
    ACCURACY_MANUAL_WARMUP_CONCURRENCY \
    ACCURACY_MANUAL_WARMUP_SEED \
    ACCURACY_MANUAL_WARMUP_TIMEOUT; do
    value="${!name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || accuracy_fail "${name} must be a positive integer"
done
((ACCURACY_MANUAL_WARMUP_CONCURRENCY <= ACCURACY_MANUAL_WARMUP_REQUESTS)) || \
    accuracy_fail "manual warmup concurrency cannot exceed request count"
((ACCURACY_MANUAL_WARMUP_INPUT_LEN + ACCURACY_MANUAL_WARMUP_OUTPUT_LEN <= \
  ACCURACY_MAX_MODEL_LEN)) || \
    accuracy_fail "manual warmup input_len + output_len exceeds ACCURACY_MAX_MODEL_LEN"

accuracy_require_command python
"${ACCURACY_ROOT}/scripts/11_check_service.sh"

warmup_root="${result_dir}/manual_warmup"
dataset_file="${warmup_root}/dataset.jsonl"
dataset_metadata="${warmup_root}/dataset_metadata.json"
warmup_result="${warmup_root}/manual_warmup.json"
dataset_log="${ACCURACY_ROOT}/logs/warmup/${run_id}_dataset.log"
warmup_log="${ACCURACY_ROOT}/logs/warmup/${run_id}.log"
[[ ! -e "${dataset_file}" && ! -e "${dataset_metadata}" && \
   ! -e "${warmup_result}" ]] || \
    accuracy_fail "manual warmup artifacts already exist under: ${warmup_root}"
mkdir -p "${warmup_root}" "$(dirname "${warmup_log}")"

lengths=""
for ((index = 0; index < ACCURACY_MANUAL_WARMUP_REQUESTS; index++)); do
    [[ -z "${lengths}" ]] || lengths+=","
    lengths+="${ACCURACY_MANUAL_WARMUP_INPUT_LEN}"
done

generate_command=(
    python "${PROJECT_ROOT}/cpp_validation/workloads/data_generation.py"
    --backend aisbench
    --model-path "${ACCURACY_TOKENIZER_PATH}"
    --lengths "${lengths}"
    --output-tokens "${ACCURACY_MANUAL_WARMUP_OUTPUT_LEN}"
    --seed "${ACCURACY_MANUAL_WARMUP_SEED}"
    --output "${dataset_file}"
    --metadata-output "${dataset_metadata}"
)
warmup_command=(
    python "${PROJECT_ROOT}/cpp_validation/workloads/performance/prepare.py"
    --mode warmup
    --dataset-mode generated
    --dataset fixed
    --model-path "${ACCURACY_TOKENIZER_PATH}"
    --model-name "${ACCURACY_SERVED_MODEL_NAME}"
    --host "${ACCURACY_HOST}"
    --port "${ACCURACY_PORT}"
    --count "${ACCURACY_MANUAL_WARMUP_REQUESTS}"
    --concurrency "${ACCURACY_MANUAL_WARMUP_CONCURRENCY}"
    --output-tokens "${ACCURACY_MANUAL_WARMUP_OUTPUT_LEN}"
    --timeout "${ACCURACY_MANUAL_WARMUP_TIMEOUT}"
    --dataset-path "${dataset_file}"
    --dataset-metadata "${dataset_metadata}"
    --output "${warmup_result}"
)

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Generating fixed-length AISBench warmup data:' | tee "${dataset_log}"
printf ' %q' "${generate_command[@]}" | tee -a "${dataset_log}"
printf '\n' | tee -a "${dataset_log}"
set +e
"${generate_command[@]}" 2>&1 | tee -a "${dataset_log}"
status=${PIPESTATUS[0]}
set -e
[[ "${status}" -eq 0 ]] || \
    accuracy_fail "warmup dataset generation exited with status ${status}; log: ${dataset_log}"

printf 'Running manual fixed-length warmup:' | tee "${warmup_log}"
printf ' %q' "${warmup_command[@]}" | tee -a "${warmup_log}"
printf '\n' | tee -a "${warmup_log}"
set +e
"${warmup_command[@]}" 2>&1 | tee -a "${warmup_log}"
status=${PIPESTATUS[0]}
set -e

[[ "${status}" -eq 0 ]] || \
    accuracy_fail "manual warmup exited with status ${status}; log: ${warmup_log}"
accuracy_info "manual warmup dataset metadata: ${dataset_metadata}"
accuracy_info "manual warmup result: ${warmup_result}"
accuracy_info "manual warmup log: ${warmup_log}"
