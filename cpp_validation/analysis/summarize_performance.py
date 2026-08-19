#!/usr/bin/env python3
"""Build JSON, CSV, and Markdown summaries for a CPP performance run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_case_state(status: dict, result_exists: bool) -> str:
    """A performance case cannot pass without its validated result artifact."""
    state = status.get("state", "incomplete")
    if state == "passed" and not result_exists:
        return "incomplete"
    return state


def percent_change(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round((current - baseline) / baseline * 100.0, 4)


def ttft_improvement(current: float, baseline: float) -> float | None:
    change = percent_change(current, baseline)
    return None if change is None else -change


def comparison(name: str, current: dict, baseline: dict) -> dict:
    return {
        "name": name,
        "dataset": current["dataset"],
        "data_generator": current.get("data_generator", "script"),
        "current_case": current["case_id"],
        "baseline_case": baseline["case_id"],
        "ttft_improvement_percent": ttft_improvement(
            current["average_ttft_ms"], baseline["average_ttft_ms"]
        ),
        "input_throughput_change_percent": percent_change(
            current["input_throughput_tokens_per_second"],
            baseline["input_throughput_tokens_per_second"],
        ),
        "per_card_throughput_change_percent": percent_change(
            current["input_throughput_per_card_tokens_per_second"],
            baseline["input_throughput_per_card_tokens_per_second"],
        ),
    }


def build_comparisons(complete_cases: list[dict]) -> list[dict]:
    index = {
        (
            case["dataset"],
            case["runner"],
            case["cpp_mode"],
            case["execution_mode"],
            case.get("data_generator", "script"),
        ): case
        for case in complete_cases
    }
    comparisons = []
    generators = sorted(
        {case.get("data_generator", "script") for case in complete_cases}
    )
    for generator in generators:
        for dataset in ("fixed", "variable"):
            static_mrv1 = index.get(
                (dataset, "mrv1", "static", "eager", generator)
            )
            cpp_mrv1 = index.get(
                (dataset, "mrv1", "dynamic", "eager", generator)
            )
            static_mrv2 = index.get(
                (dataset, "mrv2", "static", "eager", generator)
            )
            cpp_mrv2 = index.get(
                (dataset, "mrv2", "dynamic", "eager", generator)
            )
            cpp_graph = index.get(
                (dataset, "mrv2", "dynamic", "graph", generator)
            )
            if static_mrv1 and cpp_mrv1:
                comparisons.append(
                    comparison("cpp_mrv1_vs_static", cpp_mrv1, static_mrv1)
                )
            if static_mrv2 and cpp_mrv2:
                comparisons.append(
                    comparison(
                        "cpp_mrv2_vs_mrv2_static",
                        cpp_mrv2,
                        static_mrv2,
                    )
                )
            if cpp_mrv1 and cpp_mrv2:
                comparisons.append(
                    comparison("cpp_mrv2_vs_mrv1", cpp_mrv2, cpp_mrv1)
                )
            if cpp_mrv2 and cpp_graph:
                comparisons.append(
                    comparison("graph_vs_eager_mrv2", cpp_graph, cpp_mrv2)
                )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "run.json").is_file():
        parser.error(f"run.json not found under {run_dir}")

    cases = []
    cases_root = run_dir / "cases"
    if cases_root.is_dir():
        for case_dir in sorted(cases_root.iterdir()):
            case_path = case_dir / "case.json"
            if not case_path.is_file():
                continue
            case = load_json(case_path)
            if case.get("suite") != "performance":
                continue
            status_path = case_dir / "status.json"
            result_path = case_dir / "results/performance_result.json"
            status = (
                load_json(status_path)
                if status_path.is_file()
                else {"state": "incomplete", "exit_code": None}
            )
            result = load_json(result_path) if result_path.is_file() else {}
            cases.append(
                {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "runner": case["runner"],
                    "cpp_mode": case["cpp_mode"],
                    "need_timing": case.get("cpp_tuning", {}).get("need_timing"),
                    "execution_mode": case["execution_mode"],
                    "data_generator": case.get("data_generator", "script"),
                    "pp": case["pipeline_parallel_size"],
                    "tp": case["tensor_parallel_size"],
                    "concurrency": case.get("concurrency"),
                    "request_rate": case.get("request_rate"),
                    "state": resolved_case_state(status, result_path.is_file()),
                    "exit_code": status.get("exit_code"),
                    "average_ttft_ms": result.get("average_ttft_ms"),
                    "ttft_p50_ms": result.get("ttft_p50_ms"),
                    "ttft_p90_ms": result.get("ttft_p90_ms"),
                    "ttft_p95_ms": result.get("ttft_p95_ms"),
                    "measurement_wall_time_seconds": result.get(
                        "measurement_wall_time_seconds"
                    ),
                    "input_throughput_tokens_per_second": result.get(
                        "input_throughput_tokens_per_second"
                    ),
                    "input_throughput_per_card_tokens_per_second": result.get(
                        "input_throughput_per_card_tokens_per_second"
                    ),
                    "success_requests": result.get("success_requests"),
                    "failed_requests": result.get("failed_requests"),
                    "configured_cudagraph_mode": result.get(
                        "configured_cudagraph_mode",
                        case.get("configured_cudagraph_mode"),
                    ),
                    "startup_profile_sample_count": result.get(
                        "startup_profile_sample_count"
                    ),
                    "startup_profile_observed_modes": result.get(
                        "startup_profile_observed_modes", []
                    ),
                    "startup_profile_all_eager": result.get(
                        "startup_profile_all_eager"
                    ),
                    "inference_observed_modes": result.get(
                        "inference_observed_modes", []
                    ),
                    "graph_probe_observed_graph": result.get(
                        "graph_probe_observed_graph"
                    ),
                    "cpp_graph_isolation_pass": result.get("cpp_graph_isolation_pass"),
                }
            )

    complete_cases = [
        case
        for case in cases
        if case["state"] == "passed"
        and case["average_ttft_ms"] is not None
        and case["input_throughput_tokens_per_second"] is not None
    ]
    comparisons = build_comparisons(complete_cases)

    reports = run_dir / "reports"
    reports.mkdir(exist_ok=True)
    failed = sum(case["state"] != "passed" for case in cases)
    summary = {
        "schema_version": 1,
        "run_id": load_json(run_dir / "run.json")["run_id"],
        "case_count": len(cases),
        "passed_count": len(cases) - failed,
        "failed_or_incomplete_count": failed,
        "cases": cases,
        "comparisons": comparisons,
    }
    (reports / "performance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    columns = [
        "case_id",
        "dataset",
        "runner",
        "cpp_mode",
        "need_timing",
        "execution_mode",
        "data_generator",
        "pp",
        "tp",
        "concurrency",
        "request_rate",
        "state",
        "exit_code",
        "average_ttft_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "measurement_wall_time_seconds",
        "input_throughput_tokens_per_second",
        "input_throughput_per_card_tokens_per_second",
        "success_requests",
        "failed_requests",
        "startup_profile_all_eager",
        "configured_cudagraph_mode",
        "startup_profile_sample_count",
        "startup_profile_observed_modes",
        "inference_observed_modes",
        "graph_probe_observed_graph",
        "cpp_graph_isolation_pass",
    ]
    with (reports / "performance_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["startup_profile_observed_modes"] = ";".join(
                row["startup_profile_observed_modes"]
            )
            row["inference_observed_modes"] = ";".join(row["inference_observed_modes"])
            writer.writerow(row)

    lines = [
        f"# CPP performance run {summary['run_id']}",
        "",
        f"Cases: {len(cases)}; passed: {len(cases) - failed}; failed/incomplete: {failed}.",
        "",
        "| Case | Dataset | Generator | Runner | CPP | Need timing | Mode | Load (C/RPS) | TTFT avg/P50/P90/P95 (ms) | Input tok/s | Input tok/s/card | CG config | Profile samples/modes/eager | Inference modes | Probe graph | Isolation | Status |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['case_id']} | {case['dataset']} | {case['data_generator']} "
            f"| {case['runner']} "
            f"| {case['cpp_mode']} | {case['need_timing']} "
            f"| {case['execution_mode']} "
            f"| {case['concurrency']}/{case['request_rate']} "
            f"| {case['average_ttft_ms']}/{case['ttft_p50_ms']}/"
            f"{case['ttft_p90_ms']}/{case['ttft_p95_ms']} "
            f"| {case['input_throughput_tokens_per_second']} "
            f"| {case['input_throughput_per_card_tokens_per_second']} "
            f"| {case['configured_cudagraph_mode']} "
            f"| {case['startup_profile_sample_count']}/"
            f"{','.join(case['startup_profile_observed_modes'])}/"
            f"{case['startup_profile_all_eager']} "
            f"| {','.join(case['inference_observed_modes'])} "
            f"| {case['graph_probe_observed_graph']} "
            f"| {case['cpp_graph_isolation_pass']} | {case['state']} |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Comparisons",
                "",
                "| Comparison | Dataset | Generator | TTFT improvement | Input throughput change | Per-card change |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                f"| {item['name']} | {item['dataset']} | {item['data_generator']} "
                f"| {item['ttft_improvement_percent']}% "
                f"| {item['input_throughput_change_percent']}% "
                f"| {item['per_card_throughput_change_percent']}% |"
            )
    (reports / "performance_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"CPP_PERFORMANCE_SUMMARY run={run_dir} cases={len(cases)} failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
