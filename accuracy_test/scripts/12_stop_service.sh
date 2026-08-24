#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

pid_file="${ACCURACY_ROOT}/state/service.pid"
state_file="${ACCURACY_ROOT}/state/service.env"
if [[ ! -s "${pid_file}" ]]; then
    accuracy_info "no recorded accuracy-test service"
    exit 0
fi

pid="$(<"${pid_file}")"
[[ "${pid}" =~ ^[0-9]+$ ]] || accuracy_fail "invalid PID file: ${pid_file}"
if [[ -r "${state_file}" ]]; then
    # shellcheck source=/dev/null
    source "${state_file}"
fi

if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}" "${state_file}"
    accuracy_info "recorded PID ${pid} is no longer running"
    exit 0
fi

args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
pgid="$(ps -p "${pid}" -o pgid= 2>/dev/null | tr -d ' ' || true)"
if [[ "${args}" != *"vllm serve"* || "${args}" != *"${ACCURACY_MODEL_PATH}"* || \
      "${pgid}" != "${pid}" ]]; then
    accuracy_fail "refusing to stop PID ${pid}: process identity does not match this accuracy service"
fi

kill -TERM -- "-${pid}"
for _ in {1..90}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${pid_file}" "${state_file}"
        accuracy_info "stopped service process group ${pid}"
        exit 0
    fi
    sleep 1
done

accuracy_fail "service process group ${pid} did not stop within 90 seconds; no SIGKILL was sent"
