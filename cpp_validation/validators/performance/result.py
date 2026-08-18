#!/usr/bin/env python3
"""Validate AISBench output and CPP execution-mode isolation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from cpp_validation.workloads.performance.dataset import performance_input_lengths

TRACE_PATTERN = re.compile(r"\[CPP_EXECUTION_MODE_TRACE\]\s+(\{.*\})")


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        raise ValueError(f"could not parse numeric metric value: {value!r}")
    return float(match.group(0))


def first_leaf(value: Any) -> Any:
    if isinstance(value, dict):
        for preferred in ("Total", "total", "All", "all", "Overall", "overall"):
            if preferred in value:
                return first_leaf(value[preferred])
        if not value:
            raise ValueError("metric dictionary is empty")
        return first_leaf(next(iter(value.values())))
    return value


def common_metric(data: dict, name: str) -> float:
    if name not in data:
        raise KeyError(f"AISBench common metric is missing: {name}")
    return parse_number(first_leaf(data[name]))


def find_single(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob(filename) if "performances" in path.parts]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one AISBench {filename}, found {len(matches)} under {root}"
        )
    return matches[0]


def load_aisbench_metrics(root: Path, dataset: str) -> dict:
    common_path = find_single(root, f"cpp_{dataset}.json")
    request_path = find_single(root, f"cpp_{dataset}.csv")
    common = json.loads(common_path.read_text(encoding="utf-8"))

    ttft_metrics = None
    with request_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Performance Parameters") == "TTFT":
                ttft_metrics = {
                    "average_ttft_ms": parse_number(row["Average"]),
                    "ttft_p50_ms": parse_number(row.get("P50") or row["Median"]),
                    "ttft_p90_ms": parse_number(row["P90"]),
                    "ttft_p95_ms": parse_number(row["P95"]),
                }
                break
    if ttft_metrics is None:
        raise RuntimeError(f"TTFT statistics were not found in {request_path}")

    benchmark_duration_ms = common_metric(common, "Benchmark Duration")
    if benchmark_duration_ms <= 0:
        raise RuntimeError(
            f"AISBench Benchmark Duration must be positive: {benchmark_duration_ms}"
        )
    total_input_tokens = int(common_metric(common, "Total Input Tokens"))
    reported_input_throughput = common_metric(common, "Input Token Throughput")
    measurement_wall_time_seconds = benchmark_duration_ms / 1000.0
    calculated_input_throughput = round(
        total_input_tokens / measurement_wall_time_seconds, 4
    )

    return {
        **ttft_metrics,
        "measurement_wall_time_seconds": measurement_wall_time_seconds,
        "input_throughput_tokens_per_second": calculated_input_throughput,
        "aisbench_reported_input_throughput_tokens_per_second": (
            reported_input_throughput
        ),
        "success_requests": int(common_metric(common, "Success Requests")),
        "failed_requests": int(common_metric(common, "Failed Requests")),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": int(common_metric(common, "Total Generated Tokens")),
        "aisbench_common_result": str(common_path),
        "aisbench_request_result": str(request_path),
    }


def load_trace_events(log_path: Path) -> list[dict]:
    events = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACE_PATTERN.search(line)
        if match:
            events.append(json.loads(match.group(1)))
    return events


def validate_case(
    *,
    aisbench_root: Path,
    server_log: Path,
    dataset: str,
    runner: str,
    dynamic: bool,
    execution_mode: str,
    pp_size: int,
    tp_size: int,
    graph_probe: Path | None,
    data_generator: str = "script",
    dataset_metadata: Path | None = None,
    manual_warmup: Path | None = None,
    warmup_count: int | None = None,
    warmup_dataset_mode: str | None = None,
    warmup_dataset_metadata: Path | None = None,
    prefix_prime: Path | None = None,
) -> tuple[dict, list[str]]:
    errors = []
    metrics = load_aisbench_metrics(aisbench_root, dataset)
    expected_lengths = performance_input_lengths(dataset)
    expected_input_tokens = sum(expected_lengths)
    generation_metadata = None
    if dataset_metadata is not None:
        if not dataset_metadata.is_file():
            errors.append(f"dataset metadata is missing: {dataset_metadata}")
        else:
            generation_metadata = json.loads(
                dataset_metadata.read_text(encoding="utf-8")
            )
            if generation_metadata.get("backend") != data_generator:
                errors.append(
                    "dataset generator metadata mismatch: "
                    f"expected={data_generator}, "
                    f"actual={generation_metadata.get('backend')}"
                )
            if generation_metadata.get("input_token_lengths") != expected_lengths:
                errors.append("generated input token lengths violate the suite contract")
            prefix_metadata = generation_metadata.get("prefix_cache", {})
            expected_prefix_cache = dataset == "variable"
            if prefix_metadata.get("enabled") is not expected_prefix_cache:
                errors.append(
                    "generated prefix-cache mode violates the suite contract: "
                    f"expected={expected_prefix_cache}, "
                    f"actual={prefix_metadata.get('enabled')}"
                )
            if expected_prefix_cache:
                if abs(
                    float(prefix_metadata.get("mean_planned_prefix_hit_ratio", 0.0))
                    - 0.9
                ) > 1e-4:
                    errors.append("variable dataset planned prefix-hit ratio is not 90%")
                if float(
                    prefix_metadata.get("mean_cacheable_prefix_hit_ratio", 0.0)
                ) < 0.89:
                    errors.append(
                        "variable dataset cacheable prefix-hit ratio is below 89%"
                    )
                if prefix_metadata.get("prefix_test") is not True:
                    errors.append("variable dataset prefix-test warmup is not enabled")
    manual_warmup_metadata = None
    if manual_warmup is not None:
        if not manual_warmup.is_file():
            errors.append(f"manual warmup metadata is missing: {manual_warmup}")
        else:
            manual_warmup_metadata = json.loads(
                manual_warmup.read_text(encoding="utf-8")
            )
            if manual_warmup_metadata.get("mode") != "warmup":
                errors.append("manual warmup result has the wrong mode")
            if manual_warmup_metadata.get("dataset") != dataset:
                errors.append("manual warmup dataset does not match the measurement")
            if manual_warmup_metadata.get("dataset_mode") != warmup_dataset_mode:
                errors.append("manual warmup dataset mode does not match the case")
            if manual_warmup_metadata.get("request_count") != warmup_count:
                errors.append("manual warmup request count does not match the case")
            records = manual_warmup_metadata.get("records", [])
            if len(records) != warmup_count or any(
                record.get("completion_tokens") != 1 for record in records
            ):
                errors.append("manual warmup requests did not all complete successfully")
        if warmup_dataset_metadata is None or not warmup_dataset_metadata.is_file():
            errors.append("manual warmup dataset metadata is missing")
        else:
            warmup_generation = json.loads(
                warmup_dataset_metadata.read_text(encoding="utf-8")
            )
            if warmup_generation.get("input_token_lengths") != expected_lengths:
                errors.append("manual warmup data violates the target length distribution")
            if warmup_dataset_mode == "generated":
                if not warmup_generation.get("disjoint_from"):
                    errors.append("generated manual warmup has no isolation evidence")
                if generation_metadata is not None and (
                    warmup_generation.get("seed") == generation_metadata.get("seed")
                ):
                    errors.append("generated manual warmup reused the measurement seed")
    prefix_prime_metadata = None
    prefix_config = (
        generation_metadata.get("prefix_cache", {})
        if generation_metadata is not None
        else {}
    )
    if prefix_config.get("prefix_test") is True:
        if prefix_prime is None or not prefix_prime.is_file():
            errors.append("prefix-prime result is missing")
        else:
            prefix_prime_metadata = json.loads(
                prefix_prime.read_text(encoding="utf-8")
            )
            cacheable = prefix_config.get("cacheable_prefix_token_lengths") or \
                prefix_config.get("planned_prefix_token_lengths") or []
            expected_prefix_tokens = max(cacheable) if cacheable else None
            records = prefix_prime_metadata.get("records") or [{}]
            if not (
                prefix_prime_metadata.get("mode") == "prefix-prime"
                and prefix_prime_metadata.get("request_count") == 1
                and prefix_prime_metadata.get("expected_prefix_tokens")
                == expected_prefix_tokens
                and records[0].get("prompt_tokens") == expected_prefix_tokens
                and records[0].get("completion_tokens") == 1
            ):
                errors.append("prefix-prime request did not match the shared prefix plan")

    reported_throughput = metrics[
        "aisbench_reported_input_throughput_tokens_per_second"
    ]
    calculated_throughput = metrics["input_throughput_tokens_per_second"]
    if abs(reported_throughput - calculated_throughput) > max(
        0.01, calculated_throughput * 0.001
    ):
        errors.append(
            "AISBench input throughput does not match total input tokens / "
            f"measurement wall time: reported={reported_throughput}, "
            f"calculated={calculated_throughput}"
        )

    if metrics["success_requests"] != len(expected_lengths):
        errors.append(
            f"successful requests={metrics['success_requests']}, expected={len(expected_lengths)}"
        )
    if metrics["failed_requests"] != 0:
        errors.append(f"failed requests={metrics['failed_requests']}, expected=0")
    if metrics["total_input_tokens"] != expected_input_tokens:
        errors.append(
            f"total input tokens={metrics['total_input_tokens']}, expected={expected_input_tokens}"
        )
    if metrics["total_output_tokens"] != len(expected_lengths):
        errors.append(
            f"total output tokens={metrics['total_output_tokens']}, expected={len(expected_lengths)}"
        )

    events = load_trace_events(server_log)
    rank_zero_startup = [
        event
        for event in events
        if event.get("event") == "startup_profile_execution_mode"
        and event.get("runner") == runner
        and event.get("pp_rank") == 0
        and event.get("tp_rank") == 0
    ]
    rank_zero_warmup = [
        event
        for event in events
        if event.get("event") == "startup_profile_warmup_execution_mode"
        and event.get("runner") == runner
        and event.get("pp_rank") == 0
        and event.get("tp_rank") == 0
    ]
    # Runs started before warm-up events were named separately have one
    # duplicate max-chunk event followed by sample indexes 1..64.
    legacy_indexes = [event.get("sample_index") for event in rank_zero_startup]
    if (
        len(rank_zero_startup) == 65
        and legacy_indexes == list(range(65))
        and rank_zero_startup[0].get("num_tokens")
        == rank_zero_startup[1].get("num_tokens")
    ):
        rank_zero_warmup.append(rank_zero_startup[0])
        rank_zero_startup = rank_zero_startup[1:]
    rank_zero_inference = [
        event
        for event in events
        if event.get("event") == "inference_execution_mode_observed"
        and event.get("runner") == runner
        and event.get("pp_rank") == 0
        and event.get("tp_rank") == 0
    ]
    startup_modes = sorted(
        {
            event.get("actual_execution_mode")
            for event in rank_zero_startup
            if event.get("actual_execution_mode") is not None
        }
    )
    inference_modes = sorted(
        {
            event.get("actual_execution_mode")
            for event in rank_zero_inference
            if event.get("actual_execution_mode") is not None
        }
    )
    configured_modes = sorted(
        {
            event.get("configured_cudagraph_mode")
            for event in rank_zero_warmup + rank_zero_startup + rank_zero_inference
            if event.get("configured_cudagraph_mode") is not None
        }
    )
    startup_all_eager = bool(rank_zero_startup) and all(
        event.get("actual_execution_mode") == "NONE"
        and event.get("need_eager") is True
        for event in rank_zero_warmup + rank_zero_startup
    )

    if dynamic:
        if len(rank_zero_startup) != 64:
            errors.append(
                f"rank-zero CPP startup samples={len(rank_zero_startup)}, expected=64"
            )
        if not startup_all_eager:
            errors.append(
                f"CPP startup profiling was not entirely eager: modes={startup_modes}"
            )

    graph_probe_ok = False
    if execution_mode == "graph":
        if graph_probe is None or not graph_probe.is_file():
            errors.append("graph probe result is missing")
        else:
            probe = json.loads(graph_probe.read_text(encoding="utf-8"))
            graph_probe_ok = (
                probe.get("request_count") == 1
                and probe.get("expected_output_tokens") == 2
                and probe.get("records", [{}])[0].get("prompt_tokens") == 128
                and probe.get("records", [{}])[0].get("completion_tokens") == 2
                and probe.get("observed_graph") is True
            )
            if not graph_probe_ok:
                errors.append("graph probe request did not complete with two output tokens")
        if "FULL_DECODE_ONLY" not in configured_modes:
            errors.append(
                f"effective configured graph mode was not FULL_DECODE_ONLY: {configured_modes}"
            )
        graph_events = [
            event
            for event in rank_zero_inference
            if event.get("actual_execution_mode") != "NONE"
            and event.get("need_eager") is False
        ]
        if not graph_events:
            errors.append(
                f"normal inference never observed graph replay: modes={inference_modes}"
            )
        graph_probe_ok = graph_probe_ok and bool(graph_events)

    device_count = pp_size * tp_size
    result = {
        "schema_version": 1,
        "suite": "performance",
        "passed": not errors,
        "dataset": dataset,
        "runner": runner,
        "dynamic": dynamic,
        "execution_mode": execution_mode,
        "data_generator": data_generator,
        "dataset_generation_metadata": generation_metadata,
        "prefix_cache": (
            generation_metadata.get("prefix_cache")
            if generation_metadata is not None
            else None
        ),
        "manual_warmup": manual_warmup_metadata,
        "prefix_prime": prefix_prime_metadata,
        **metrics,
        "input_throughput_per_card_tokens_per_second": round(
            metrics["input_throughput_tokens_per_second"] / device_count, 4
        ),
        "configured_cudagraph_mode": (
            configured_modes[0] if len(configured_modes) == 1 else None
        ),
        "configured_cudagraph_modes": configured_modes,
        "startup_profile_sample_count": len(rank_zero_startup),
        "startup_profile_warmup_count": len(rank_zero_warmup),
        "startup_profile_observed_modes": startup_modes,
        "startup_profile_all_eager": startup_all_eager if dynamic else None,
        "inference_observed_modes": inference_modes,
        "graph_probe_observed_graph": graph_probe_ok if execution_mode == "graph" else None,
        "cpp_graph_isolation_pass": (
            not errors and startup_all_eager and graph_probe_ok
            if execution_mode == "graph"
            else None
        ),
        "validation_errors": errors,
    }
    return result, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aisbench-root", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fixed", "variable"), required=True)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--dynamic", choices=("0", "1"), required=True)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--graph-probe", type=Path)
    parser.add_argument("--data-generator", choices=("aisbench", "script"), default="script")
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--manual-warmup", type=Path)
    parser.add_argument("--warmup-count", type=int)
    parser.add_argument(
        "--warmup-dataset-mode", choices=("generated", "reuse")
    )
    parser.add_argument("--warmup-dataset-metadata", type=Path)
    parser.add_argument("--prefix-prime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result, errors = validate_case(
        aisbench_root=args.aisbench_root,
        server_log=args.server_log,
        dataset=args.dataset,
        runner=args.runner,
        dynamic=args.dynamic == "1",
        execution_mode=args.execution_mode,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        graph_probe=args.graph_probe,
        data_generator=args.data_generator,
        dataset_metadata=args.dataset_metadata,
        manual_warmup=args.manual_warmup,
        warmup_count=args.warmup_count,
        warmup_dataset_mode=args.warmup_dataset_mode,
        warmup_dataset_metadata=args.warmup_dataset_metadata,
        prefix_prime=args.prefix_prime,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"CPP_PERFORMANCE_VALIDATION_ERROR: {error}")
        return 1
    print(
        "CPP_PERFORMANCE_VALIDATION_PASS "
        f"dataset={args.dataset} runner={args.runner} "
        f"ttft_ms={result['average_ttft_ms']} "
        f"input_tps={result['input_throughput_tokens_per_second']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
