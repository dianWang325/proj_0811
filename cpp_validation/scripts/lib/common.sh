#!/usr/bin/env bash

cpp_fail() {
    echo "CPP_VALIDATION_FAIL: $*" >&2
    return 1
}

cpp_is_port_open() {
    local port="$1"
    python3 - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

cpp_git_revision() {
    local directory="$1"
    git -C "${directory}" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
}

cpp_utc_timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}
