#!/usr/bin/env bash
set -Eeuo pipefail

readonly CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CPP_RUNNER=mrv2
export CPP_DYNAMIC=1
export CPP_EXECUTION_MODE="${CPP_EXECUTION_MODE:-eager}"

exec "${CPP_ROOT}/scripts/run_case.sh" "$@"
