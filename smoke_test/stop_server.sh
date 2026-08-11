#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORK_DIR="/home/w00985415"
readonly TEST_DIR="${WORK_DIR}/proj_0811/smoke_test"
readonly PID_FILE="${TEST_DIR}/serve.pid"
readonly EXPECTED_MODEL="/mnt/a800_weight/Qwen3-30B-A3B-W8A8"

cd "${WORK_DIR}"

if [[ ! -s "${PID_FILE}" ]]; then
    echo "No smoke-test PID file found; nothing to stop."
    exit 0
fi

pid="$(<"${PID_FILE}")"
if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "Refusing to act on invalid PID file: ${PID_FILE}" >&2
    exit 1
fi

if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Recorded PID ${pid} is no longer running."
    rm -f "${PID_FILE}"
    exit 0
fi

args="$(ps -p "${pid}" -o args=)"
pgid="$(ps -p "${pid}" -o pgid= | tr -d ' ')"
if [[ "${args}" != *"${EXPECTED_MODEL}"* ]] || [[ "${pgid}" != "${pid}" ]]; then
    echo "Refusing to stop PID ${pid}: process identity does not match this smoke test." >&2
    echo "Observed command: ${args}" >&2
    exit 1
fi

kill -TERM -- "-${pid}"
for _ in {1..60}; do
    if ! kill -0 -- "-${pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        echo "Stopped smoke-test process group ${pid}."
        exit 0
    fi
    sleep 1
done

echo "Smoke-test process group ${pid} did not exit after 60s; leaving it for manual inspection." >&2
exit 1
