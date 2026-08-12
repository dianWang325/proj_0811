#!/usr/bin/env bash
set -Eeuo pipefail

readonly TEST_DIR="/home/w00985415/proj_0811/smoke_test"
readonly REQUEST_MODE="${CPP_REQUEST_MODE:-both}"

cases=(
    "mrv1 0 eager"
    "mrv1 1 eager"
    "mrv2 0 eager"
    "mrv2 1 eager"
    "mrv1 1 graph"
    "mrv2 1 graph"
)

for case_spec in "${cases[@]}"; do
    read -r runner dynamic execution_mode <<<"${case_spec}"
    CPP_RUNNER="${runner}" \
    CPP_DYNAMIC="${dynamic}" \
    CPP_EXECUTION_MODE="${execution_mode}" \
    CPP_REQUEST_MODE="${REQUEST_MODE}" \
    CPP_LOG_FILE="${TEST_DIR}/serve_${runner}_${dynamic}_${execution_mode}_${REQUEST_MODE}.log" \
        "${TEST_DIR}/run_cpp_functional_case.sh"
done

echo "CPP_REGRESSION_MATRIX_PASS cases=${#cases[@]} request_mode=${REQUEST_MODE}"
