#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORK_DIR="/home/w00985415"
readonly TEST_DIR="${WORK_DIR}/proj_0811/smoke_test"
readonly PID_FILE="${TEST_DIR}/serve.pid"
readonly LOG_FILE="${TEST_DIR}/serve.log"
readonly CARD_ID=6

cd "${WORK_DIR}"
mkdir -p "${TEST_DIR}"

if [[ -s "${PID_FILE}" ]]; then
    old_pid="$(<"${PID_FILE}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "Refusing to start: recorded smoke-test PID ${old_pid} is still running." >&2
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

card_status="$(npu-smi info)"
if ! grep -q "No running processes found in NPU ${CARD_ID}" <<<"${card_status}"; then
    echo "Refusing to start: NPU ${CARD_ID} is no longer process-free." >&2
    printf '%s\n' "${card_status}" >&2
    exit 1
fi

if python - "${SMOKE_TEST_PORT:-18080}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
then
    echo "Refusing to start: port ${SMOKE_TEST_PORT:-18080} is already in use." >&2
    exit 1
fi

: >"${LOG_FILE}"
nohup setsid "${TEST_DIR}/deploy_pp2.sh" >"${LOG_FILE}" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_FILE}"
echo "Started smoke-test server as PID ${server_pid}; log: ${LOG_FILE}"
