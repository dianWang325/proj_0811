#!/usr/bin/env python3
"""Submit exact-token long-context requests to a local vLLM server."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("CPP_MODEL_PATH", "/mnt/a800_weight/Qwen3-30B-A3B-W8A8")
MODEL_NAME = os.environ.get("CPP_MODEL_NAME", "qwen3-30b-a3b-w8a8")
PORT = int(os.environ.get("CPP_PORT", "18080"))
MODE = os.environ.get("CPP_REQUEST_MODE", "both")
TARGETS = [int(value) for value in os.environ.get("CPP_TOKEN_TARGETS", "10000,20000,40000").split(",")]
TIMEOUT = int(os.environ.get("CPP_REQUEST_TIMEOUT", "1800"))
MAX_OUTPUT_TOKENS = int(os.environ.get("CPP_MAX_OUTPUT_TOKENS", "1"))
REQUEST_REPEATS = int(os.environ.get("CPP_REQUEST_REPEATS", "1"))
RESULT_PATH = Path(os.environ.get("CPP_RESULT_PATH", "cpp_long_context_result.json"))
BASE_URL = f"http://127.0.0.1:{PORT}"


def request_json(path: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_exact_prompt(tokenizer, target_tokens: int, unit_text: str) -> str:
    unit_ids = tokenizer.encode(unit_text, add_special_tokens=False)
    if len(unit_ids) != 1:
        raise RuntimeError(f"Expected a one-token prompt unit, got {unit_ids}")
    expected_ids = unit_ids * target_tokens
    prompt = tokenizer.decode(expected_ids, skip_special_tokens=False)
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if actual_ids != expected_ids:
        raise RuntimeError(f"Tokenizer round trip changed target={target_tokens} to actual={len(actual_ids)}")
    return prompt


def submit(prompt: str, target_tokens: int, phase: str, repeat_index: int) -> dict:
    started = time.perf_counter()
    result = request_json(
        "/v1/completions",
        {"model": MODEL_NAME, "prompt": prompt, "temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS},
    )
    elapsed = time.perf_counter() - started
    if not result.get("choices"):
        raise RuntimeError(f"Completion response has no choices for {target_tokens} tokens")
    actual_tokens = result.get("usage", {}).get("prompt_tokens")
    if actual_tokens != target_tokens:
        raise RuntimeError(f"API token count mismatch: target={target_tokens}, actual={actual_tokens}")
    record = {
        "phase": phase,
        "repeat_index": repeat_index,
        "target_tokens": target_tokens,
        "prompt_tokens": actual_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": result["choices"][0].get("finish_reason"),
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main() -> None:
    if MODE not in {"sequential", "concurrent", "both"}:
        raise ValueError("CPP_REQUEST_MODE must be sequential, concurrent, or both")
    if TARGETS != sorted(TARGETS) or any(target <= 0 for target in TARGETS):
        raise ValueError("CPP_TOKEN_TARGETS must be positive and ascending")
    if REQUEST_REPEATS <= 0:
        raise ValueError("CPP_REQUEST_REPEATS must be a positive integer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    prompt_units = (" hello", " world", " test")
    if len(TARGETS) > len(prompt_units):
        raise ValueError(f"At most {len(prompt_units)} token targets are supported")
    prompts = {
        target: build_exact_prompt(tokenizer, target, unit)
        for target, unit in zip(TARGETS, prompt_units, strict=True)
    }
    records: list[dict] = []

    if MODE in {"sequential", "both"}:
        for repeat_index in range(REQUEST_REPEATS):
            for target in TARGETS:
                records.append(submit(prompts[target], target, "sequential", repeat_index))
    if MODE in {"concurrent", "both"}:
        requests = [
            (repeat_index, target)
            for repeat_index in range(REQUEST_REPEATS)
            for target in TARGETS
        ]
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = {
                executor.submit(
                    submit, prompts[target], target, "concurrent", repeat_index
                ): (repeat_index, target)
                for repeat_index, target in requests
            }
            for future in as_completed(futures):
                records.append(future.result())

    phase_count = 2 if MODE == "both" else 1
    expected_request_count = len(TARGETS) * REQUEST_REPEATS * phase_count
    if len(records) != expected_request_count:
        raise RuntimeError(
            f"Request count mismatch: expected={expected_request_count}, actual={len(records)}"
        )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"CPP_LONG_CONTEXT_PASS mode={MODE} requests={len(records)} "
        f"result={RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
