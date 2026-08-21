#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

matrix_max_rounds="${CPP_MATRIX_MAX_ROUNDS:-3}"
[[ "${matrix_max_rounds}" =~ ^[1-3]$ ]] || {
    echo "CPP_MATRIX_MAX_ROUNDS must be between 1 and 3" >&2
    exit 1
}

# Fixed server-14 experiment contract. The shared matrix runner owns all
# execution, failure, preflight, artifact, and summary behavior.
export CPP_PIPELINE_PARALLEL_SIZE=2
export CPP_TENSOR_PARALLEL_SIZE=4
export CPP_MATRIX_MAX_ROUNDS="${matrix_max_rounds}"
export CPP_MATRIX_SHUFFLE_AFTER_FIRST=1
export CPP_MATRIX_STOP_AFTER_FAILED_ROUND=1
export CPP_MATRIX_RECORD_ORDER=1
export CPP_MATRIX_REPORT_ROUNDS=1
export CPP_MATRIX_GLOBAL_DATASET=variable
export CPP_MATRIX_SHUFFLE_SEED="${CPP_MATRIX_SHUFFLE_SEED:-20260820}"
export CPP_MATRIX_DRY_RUN_LABEL="cases_per_round=12 max_rounds=${CPP_MATRIX_MAX_ROUNDS} seed=${CPP_MATRIX_SHUFFLE_SEED} dataset=variable execution_mode=eager pp=2 tp=4"

# CPP-off cases use need_timing=false because timing helpers are inactive.
# CPP-on cases cover both timing values. This yields 12 meaningful cases.
export CPP_MATRIX_INLINE_CASES=$(cat <<'CASES'
mrv1 0 eager variable false 0
mrv1 0 eager variable false 1
mrv2 0 eager variable false 0
mrv2 0 eager variable false 1
mrv1 1 eager variable false 0
mrv1 1 eager variable false 1
mrv1 1 eager variable true 0
mrv1 1 eager variable true 1
mrv2 1 eager variable false 0
mrv2 1 eager variable false 1
mrv2 1 eager variable true 0
mrv2 1 eager variable true 1
CASES
)

exec "${CPP_ROOT}/scripts/run_perf_matrix_0820_test_async.sh" "$@"
