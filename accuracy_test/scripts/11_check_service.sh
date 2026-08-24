#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
accuracy_load_config

wait_seconds=0
probe=0
while (($#)); do
    case "$1" in
        --wait)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || accuracy_fail "--wait requires seconds"
            wait_seconds="$2"
            shift 2
            ;;
        --probe)
            probe=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--wait SECONDS] [--probe]"
            exit 0
            ;;
        *) accuracy_fail "unknown argument: $1" ;;
    esac
done

state_file="${ACCURACY_ROOT}/state/service.env"
if [[ -r "${state_file}" ]]; then
    # shellcheck source=/dev/null
    source "${state_file}"
fi

deadline=$((SECONDS + wait_seconds))
while ! accuracy_is_port_open "${ACCURACY_PORT}"; do
    ((SECONDS < deadline)) || accuracy_fail "service port ${ACCURACY_PORT} is not ready"
    if [[ -n "${ACCURACY_SERVICE_PID:-}" ]] && ! kill -0 "${ACCURACY_SERVICE_PID}" 2>/dev/null; then
        [[ -z "${ACCURACY_SERVICE_LOG:-}" ]] || tail -100 "${ACCURACY_SERVICE_LOG}" >&2 || true
        accuracy_fail "service process ${ACCURACY_SERVICE_PID} exited before becoming ready"
    fi
    sleep 5
done

models_json="$(curl --fail --silent --show-error --max-time 30 \
    "http://${ACCURACY_HOST}:${ACCURACY_PORT}/v1/models")"
python - "${ACCURACY_SERVED_MODEL_NAME}" "${models_json}" <<'PY'
import json
import sys

expected = sys.argv[1]
payload = json.loads(sys.argv[2])
ids = [item.get("id") for item in payload.get("data", [])]
if expected not in ids:
    raise SystemExit(f"served model mismatch: expected={expected!r}, available={ids!r}")
print(f"SERVICE_MODELS_OK model={expected}")
PY

if [[ "${probe}" == "1" ]]; then
    response="$({ python - "${ACCURACY_SERVED_MODEL_NAME}" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0,
    "max_tokens": 8,
}))
PY
    } | curl --fail --silent --show-error --max-time 300 \
        -H 'Content-Type: application/json' \
        --data-binary @- \
        "http://${ACCURACY_HOST}:${ACCURACY_PORT}/v1/chat/completions")"
    python - "${response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
choices = payload.get("choices") or []
content = choices[0].get("message", {}).get("content") if choices else None
if not content:
    raise SystemExit(f"chat probe returned no content: {payload}")
print("SERVICE_CHAT_OK")
PY
fi
