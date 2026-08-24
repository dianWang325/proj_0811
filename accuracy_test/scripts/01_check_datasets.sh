#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

selection="${1:-all}"
[[ "${selection}" == "all" || "${selection}" == "gsm8k" || "${selection}" == "gpqa" ]] || \
    accuracy_fail "dataset must be one of: all, gsm8k, gpqa"

check_file() {
    [[ -s "$1" ]] || accuracy_fail "required dataset file is missing or empty: $1"
    printf 'DATASET_OK %s\n' "$1"
}

if [[ "${selection}" == "all" || "${selection}" == "gsm8k" ]]; then
    check_file "${ACCURACY_DATASET_ROOT}/gsm8k/test.jsonl"
fi

if [[ "${selection}" == "all" || "${selection}" == "gpqa" ]]; then
    check_file "${ACCURACY_DATASET_ROOT}/gpqa/gpqa_diamond.csv"
    check_file "${ACCURACY_DATASET_ROOT}/gpqa/license.txt"
fi
