#!/usr/bin/env python3
"""Wait for the local vLLM API and submit one minimal chat request."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TEST_DIR = Path("/home/w00985415/proj_0811/smoke_test")
PORT = int(os.environ.get("SMOKE_TEST_PORT", "18080"))
BASE_URL = f"http://127.0.0.1:{PORT}"
TIMEOUT_SECONDS = int(os.environ.get("SMOKE_TEST_TIMEOUT", "1200"))
MODEL_NAME = "qwen3-30b-a3b-w8a8"


def request_json(path: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_until_ready() -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    last_error = "server has not responded"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become healthy within {TIMEOUT_SECONDS}s: {last_error}")


def main() -> int:
    wait_until_ready()
    models = request_json("/v1/models")
    model_ids = {item.get("id") for item in models.get("data", [])}
    if MODEL_NAME not in model_ids:
        raise RuntimeError(f"Expected model {MODEL_NAME!r}; API returned {sorted(model_ids)!r}")

    result = request_json(
        "/v1/chat/completions",
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=180,
    )
    if not result.get("choices"):
        raise RuntimeError(f"Completion response has no choices: {result!r}")

    output_path = TEST_DIR / "smoke_result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"SMOKE_TEST_PASS result={output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE_TEST_FAIL: {exc}", file=sys.stderr)
        raise
