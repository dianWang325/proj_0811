#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

dataset=all
num_prompts=""
debug=0
run_id=""
while (($#)); do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || accuracy_fail "--dataset requires all, gsm8k or gpqa"
            dataset="$2"
            shift 2
            ;;
        --num-prompts)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || accuracy_fail "--num-prompts requires a positive integer"
            num_prompts="$2"
            shift 2
            ;;
        --full)
            num_prompts=""
            shift
            ;;
        --debug)
            debug=1
            shift
            ;;
        --run-id)
            [[ $# -ge 2 && "$2" =~ ^[A-Za-z0-9_.-]+$ ]] || accuracy_fail "invalid --run-id"
            run_id="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dataset all|gsm8k|gpqa] [--num-prompts N|--full] [--debug] [--run-id ID]"
            exit 0
            ;;
        *) accuracy_fail "unknown argument: $1" ;;
    esac
done

[[ "${dataset}" == "all" || "${dataset}" == "gsm8k" || "${dataset}" == "gpqa" ]] || \
    accuracy_fail "dataset must be one of: all, gsm8k, gpqa"
[[ "${ACCURACY_MAX_OUT_LEN}" =~ ^[1-9][0-9]*$ ]] || \
    accuracy_fail "ACCURACY_MAX_OUT_LEN must be a positive integer"
[[ "${ACCURACY_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
    accuracy_fail "ACCURACY_BATCH_SIZE must be a positive integer"
((ACCURACY_MAX_OUT_LEN < ACCURACY_MAX_MODEL_LEN)) || \
    accuracy_fail "ACCURACY_MAX_OUT_LEN must be smaller than ACCURACY_MAX_MODEL_LEN"
accuracy_require_command ais_bench
accuracy_require_command python
"${ACCURACY_ROOT}/scripts/01_check_datasets.sh" "${dataset}"
"${ACCURACY_ROOT}/scripts/11_check_service.sh"

python - <<'PY'
import numpy
import scipy

parts = tuple(int(part) for part in numpy.__version__.split(".")[:2])
if parts >= (2, 3):
    raise SystemExit(
        f"DEPENDENCY_ERROR: NumPy {numpy.__version__} is incompatible with "
        f"SciPy {scipy.__version__}; ask the environment owner to install NumPy <2.3"
    )
PY

tasks=()
[[ "${dataset}" == "all" || "${dataset}" == "gsm8k" ]] && \
    tasks+=(gsm8k_gen_0_shot_cot_chat_prompt)
[[ "${dataset}" == "all" || "${dataset}" == "gpqa" ]] && \
    tasks+=(gpqa_gen_0_shot_cot_chat_prompt)

state_file="${ACCURACY_ROOT}/state/service.env"
[[ -r "${state_file}" ]] || accuracy_fail "missing service state: start the service with 10_start_service.sh"
# shellcheck source=/dev/null
source "${state_file}"

if [[ -z "${run_id}" ]]; then
    prompt_label="full"
    [[ -z "${num_prompts}" ]] || prompt_label="n${num_prompts}"
    run_id="$(date +%Y%m%d_%H%M%S)_${ACCURACY_RUNNER}_cpp${ACCURACY_CPP_ENABLED}_${dataset}_${prompt_label}"
fi

result_dir="${ACCURACY_ROOT}/results/${run_id}"
benchmark_log="${ACCURACY_ROOT}/logs/benchmark/${run_id}.log"
environment_log="${ACCURACY_ROOT}/logs/environment/${run_id}.env"
report_file="${ACCURACY_ROOT}/reports/generated/${run_id}.md"
mkdir -p "${result_dir}"

{
    printf 'RUN_ID=%q\n' "${run_id}"
    printf 'DATASET_SELECTION=%q\n' "${dataset}"
    printf 'DATASET_TASKS=%q\n' "${tasks[*]}"
    printf 'NUM_PROMPTS=%q\n' "${num_prompts:-full}"
    printf 'DEBUG=%q\n' "${debug}"
    printf 'ACCURACY_MAX_OUT_LEN=%q\n' "${ACCURACY_MAX_OUT_LEN}"
    printf 'ACCURACY_BATCH_SIZE=%q\n' "${ACCURACY_BATCH_SIZE}"
    printf 'ACCURACY_TEMPERATURE=%q\n' "${ACCURACY_TEMPERATURE}"
    printf 'ACCURACY_REPETITION_PENALTY=%q\n' "${ACCURACY_REPETITION_PENALTY}"
    printf 'FORMAL_AISBENCH_NUM_WARMUPS=0\n'
    printf 'MANUAL_WARMUP_REQUIRED=%q\n' "${ACCURACY_CPP_ENABLED}"
    printf 'ACCURACY_MANUAL_WARMUP_INPUT_LEN=%q\n' "${ACCURACY_MANUAL_WARMUP_INPUT_LEN}"
    printf 'ACCURACY_MANUAL_WARMUP_OUTPUT_LEN=%q\n' "${ACCURACY_MANUAL_WARMUP_OUTPUT_LEN}"
    printf 'ACCURACY_MANUAL_WARMUP_REQUESTS=%q\n' "${ACCURACY_MANUAL_WARMUP_REQUESTS}"
    printf 'ACCURACY_MANUAL_WARMUP_CONCURRENCY=%q\n' "${ACCURACY_MANUAL_WARMUP_CONCURRENCY}"
    printf 'ACCURACY_MANUAL_WARMUP_SEED=%q\n' "${ACCURACY_MANUAL_WARMUP_SEED}"
    printf 'ACCURACY_MANUAL_WARMUP_TIMEOUT=%q\n' "${ACCURACY_MANUAL_WARMUP_TIMEOUT}"
    printf 'STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
    cat "${state_file}"
    python --version 2>&1 | sed 's/^/PYTHON_VERSION=/'
    python -m pip show ais-bench-benchmark 2>/dev/null | sed -n 's/^Version: /AISBENCH_VERSION=/p'
} >"${environment_log}"

export ACCURACY_TOKENIZER_PATH ACCURACY_SERVED_MODEL_NAME
export ACCURACY_HOST ACCURACY_PORT ACCURACY_MAX_OUT_LEN ACCURACY_BATCH_SIZE
export ACCURACY_REQUEST_RATE ACCURACY_TEMPERATURE ACCURACY_REPETITION_PENALTY
export AIS_BENCH_DATASETS_CACHE="${PROJECT_ROOT}/predict"

manual_warmup_status=0
if [[ "${ACCURACY_CPP_ENABLED}" == "1" ]]; then
    "${ACCURACY_ROOT}/scripts/25_manual_warmup.sh" \
        --run-id "${run_id}" \
        --result-dir "${result_dir}" || manual_warmup_status=$?
fi
printf 'MANUAL_WARMUP_EXIT_CODE=%s\n' "${manual_warmup_status}" >>"${environment_log}"
if [[ "${manual_warmup_status}" -ne 0 ]]; then
    printf 'EXIT_CODE=%s\n' "${manual_warmup_status}" >>"${environment_log}"
    accuracy_fail "manual CPP warmup failed; formal accuracy run was not started"
fi

command=(
    ais_bench
    --config-dir "${ACCURACY_ROOT}/configs/aisbench"
    --models deepseek_v4_flash_accuracy
    --datasets "${tasks[@]}"
    --summarizer example
    --dump-eval-details
    --num-warmups 0
    --work-dir "${result_dir}"
)
[[ -z "${num_prompts}" ]] || command+=(--num-prompts "${num_prompts}")
[[ "${debug}" == "0" ]] || command+=(--debug)

printf 'Running:' | tee "${benchmark_log}"
printf ' %q' "${command[@]}" | tee -a "${benchmark_log}"
printf '\n' | tee -a "${benchmark_log}"

cd "${PROJECT_ROOT}/predict"
set +e
"${command[@]}" 2>&1 | tee -a "${benchmark_log}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" -eq 0 ]]; then
    if ! python - "${result_dir}" "${#tasks[@]}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_tasks = int(sys.argv[2])
failed_logs = []
for path in root.rglob("*.out"):
    relative = path.relative_to(root).as_posix()
    if "/logs/infer/" not in f"/{relative}" and "/logs/eval/" not in f"/{relative}":
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text or "UNKNOWN applicaiton exception" in text:
        failed_logs.append(path)

predictions = list(root.rglob("predictions/**/*.json"))
predictions.extend(root.rglob("predictions/**/*.jsonl"))
evaluated = list(root.rglob("results/**/*.json"))
if failed_logs or len(predictions) < expected_tasks or len(evaluated) < expected_tasks:
    for path in failed_logs:
        print(f"AISBENCH_TASK_FAILED {path}", file=sys.stderr)
    print(
        f"AISBENCH_OUTPUT_INCOMPLETE predictions={len(predictions)} "
        f"evaluated={len(evaluated)} expected_tasks={expected_tasks}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
    then
        status=1
    fi
fi
printf 'EXIT_CODE=%s\n' "${status}" >>"${environment_log}"

if [[ "${status}" -eq 0 ]]; then
    python "${ACCURACY_ROOT}/scripts/40_generate_report.py" \
        --run-id "${run_id}" \
        --results-dir "${result_dir}" \
        --metadata "${environment_log}" \
        --output "${report_file}"
    accuracy_info "report: ${report_file}"
else
    accuracy_fail "AISBench exited with status ${status}; log: ${benchmark_log}"
fi
