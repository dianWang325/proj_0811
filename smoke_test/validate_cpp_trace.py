#!/usr/bin/env python3
"""Validate structured CPP diagnostics from a functional test run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRACE_PATTERN = re.compile(r"\[CPP_TRACE\]\s+(\{.*\})")
SCHEDULER_FIELDS = {
    "iteration",
    "req_id",
    "num_computed_tokens",
    "hist_seq_len",
    "remaining_prefill_tokens",
    "target_latency_ms",
    "predicted_chunk_size",
    "actual_scheduled_chunk_size",
    "predicted_latency_ms",
    "actual_execution_time_ms",
    "with_history_ready",
    "history_fitted",
    "fit_sample_count",
    "predictor_updated",
    "disable_profiling_timing",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--dynamic", choices=("0", "1"), required=True)
    parser.add_argument("--request-mode", choices=("sequential", "concurrent", "both"), required=True)
    args = parser.parse_args()

    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    require("Traceback" not in text, "server log contains Traceback")
    require(not re.search(r"(^|\s)ERROR(\s|$)", text, re.MULTILINE), "server log contains ERROR")

    events = []
    for line in text.splitlines():
        match = TRACE_PATTERN.search(line)
        if match:
            events.append(json.loads(match.group(1)))

    if args.dynamic == "0":
        require(not events, "static chunk case unexpectedly emitted CPP trace events")
        print("CPP_TRACE_VALIDATION_PASS dynamic=0")
        return

    completed = [event for event in events if event.get("event") == "startup_profile_completed"]
    require(completed, "missing startup_profile_completed")
    require(completed[-1].get("initial_samples") == 64, "startup profiling did not collect 64 samples")
    require(completed[-1].get("is_ready") is True, "startup predictor is not ready")

    samples = [
        event
        for event in events
        if event.get("event") == "startup_profile_sample" and event.get("runner") == args.runner
    ]
    require(samples, f"missing startup samples for {args.runner}")
    require(all(event.get("execution_mode") == "NONE" for event in samples), "startup profile was not eager")
    if args.runner == "mrv2":
        require(all(event.get("inherited_dummy_run_entered") is True for event in samples), "MRv2 dummy_run missing")
        require(all(event.get("npu_execute_model_entered") is True for event in samples), "NPU execute_model missing")
        require(
            all(event.get("upstream_execute_model_completed") is True for event in samples),
            "upstream execute_model did not complete",
        )

    iterations = [event for event in events if event.get("event") == "scheduler_iteration"]
    require(iterations, "missing scheduler_iteration events")
    for event in iterations:
        missing = SCHEDULER_FIELDS - event.keys()
        require(not missing, f"scheduler iteration missing fields: {sorted(missing)}")
    chunk_sizes = {event["actual_scheduled_chunk_size"] for event in iterations}
    require(len(chunk_sizes) >= 2, "scheduled chunk size never changed")
    require(any(event.get("with_history_ready") is True for event in iterations), "history model never became ready")

    if args.request_mode != "concurrent":
        require(any(event.get("history_fitted") is True for event in iterations), "history fit never completed")
        require(
            any(event.get("event") == "online_calibration_completed" for event in events),
            "online calibration completion was not traced",
        )
        require(
            any(event.get("event") == "worker_profiling_timing_disabled" for event in events),
            "worker did not stop profiling timing",
        )

    print(
        "CPP_TRACE_VALIDATION_PASS "
        f"runner={args.runner} iterations={len(iterations)} chunks={sorted(chunk_sizes)}"
    )


if __name__ == "__main__":
    main()
