#!/usr/bin/env bash
set -Eeuo pipefail

export CPP_RUNNER=mrv2
export CPP_DYNAMIC=1
export CPP_EXECUTION_MODE="${CPP_EXECUTION_MODE:-eager}"

exec /home/w00985415/proj_0811/smoke_test/run_cpp_functional_case.sh "$@"
