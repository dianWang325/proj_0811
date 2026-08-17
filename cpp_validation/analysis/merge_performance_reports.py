#!/usr/bin/env python3
"""Merge later completed measurements into an immutable performance report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_performance import build_comparisons


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measurement_complete(case: dict) -> bool:
    return (
        case.get("average_ttft_ms") is not None
        and case.get("input_throughput_tokens_per_second") is not None
        and case.get("input_throughput_per_card_tokens_per_second") is not None
        and case.get("success_requests") == 64
        and case.get("failed_requests") == 0
    )


def result_details(run_dir: Path, case_id: str) -> dict:
    path = run_dir / "cases" / case_id / "results" / "performance_result.json"
    return load_json(path) if path.is_file() else {}


def enrich(case: dict, run_dir: Path) -> dict:
    merged = dict(case)
    details = result_details(run_dir, case["case_id"])
    complete = measurement_complete(case)
    validated = case.get("state") == "passed" and complete
    if validated:
        report_state = "validated_pass"
    elif complete:
        report_state = "measurement_complete_validation_failed"
    else:
        report_state = "failed_or_incomplete_no_measurement"
    merged.update(
        source_run_id=run_dir.name,
        source_case_dir=str(run_dir / "cases" / case["case_id"]),
        measurement_complete=complete,
        validation_passed=validated,
        report_state=report_state,
        validation_errors=details.get("validation_errors", []),
        total_input_tokens=details.get("total_input_tokens"),
        total_output_tokens=details.get("total_output_tokens"),
    )
    return merged


def merge_cases(base_run: Path, overlay_runs: list[Path]) -> list[dict]:
    base = load_json(base_run / "reports" / "performance_summary.json")
    overlays: dict[str, tuple[dict, Path]] = {}
    for run_dir in overlay_runs:
        summary = load_json(run_dir / "reports" / "performance_summary.json")
        for case in summary["cases"]:
            if measurement_complete(case):
                overlays[case["case_id"]] = (case, run_dir)

    merged = []
    for base_case in base["cases"]:
        case, source_run = overlays.get(base_case["case_id"], (base_case, base_run))
        merged.append(enrich(case, source_run))
    return merged


def add_comparison_status(items: list[dict], cases: list[dict]) -> list[dict]:
    index = {case["case_id"]: case for case in cases}
    output = []
    for item in items:
        current = index[item["current_case"]]
        baseline = index[item["baseline_case"]]
        annotated = dict(item)
        annotated["comparison_validated"] = (
            current["validation_passed"] and baseline["validation_passed"]
        )
        output.append(annotated)
    return output


def write_reports(output_dir: Path, base_run: Path, overlays: list[Path]) -> None:
    cases = merge_cases(base_run, overlays)
    complete_cases = [case for case in cases if case["measurement_complete"]]
    comparisons = add_comparison_status(build_comparisons(complete_cases), cases)
    validated = sum(case["validation_passed"] for case in cases)
    complete = len(complete_cases)
    summary = {
        "schema_version": 1,
        "report_kind": "merged_performance_report",
        "base_run_id": base_run.name,
        "overlay_run_ids": [run.name for run in overlays],
        "case_count": len(cases),
        "measurement_complete_count": complete,
        "validated_pass_count": validated,
        "measurement_complete_validation_failed_count": sum(
            case["report_state"] == "measurement_complete_validation_failed"
            for case in cases
        ),
        "failed_or_incomplete_no_measurement_count": len(cases) - complete,
        "cases": cases,
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "performance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    columns = [
        "case_id",
        "dataset",
        "runner",
        "cpp_mode",
        "execution_mode",
        "data_generator",
        "average_ttft_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "input_throughput_tokens_per_second",
        "input_throughput_per_card_tokens_per_second",
        "success_requests",
        "failed_requests",
        "measurement_complete",
        "validation_passed",
        "report_state",
        "source_run_id",
        "validation_errors",
    ]
    with (output_dir / "performance_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["validation_errors"] = "; ".join(row["validation_errors"])
            writer.writerow(row)

    lines = [
        "# Merged CPP performance report",
        "",
        f"Base run: `{base_run.name}`",
        f"Overlay runs: {', '.join(f'`{run.name}`' for run in overlays)}",
        "",
        (
            f"Cases: {len(cases)}; complete measurements: {complete}; "
            f"validated passes: {validated}; complete but validation failed: "
            f"{summary['measurement_complete_validation_failed_count']}; "
            f"without measurement: {summary['failed_or_incomplete_no_measurement_count']}."
        ),
        "",
        "| Case | Dataset | Generator | Runner | CPP | Mode | TTFT avg/P50/P90/P95 (ms) | Input tok/s | Input tok/s/card | Requests | Report state | Source run |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for case in cases:
        requests = (
            f"{case['success_requests']}/{case['success_requests'] + case['failed_requests']}"
            if case.get("success_requests") is not None
            and case.get("failed_requests") is not None
            else "-"
        )
        lines.append(
            f"| {case['case_id']} | {case['dataset']} "
            f"| {case.get('data_generator', 'script')} | {case['runner']} "
            f"| {case['cpp_mode']} | {case['execution_mode']} "
            f"| {case['average_ttft_ms']}/{case.get('ttft_p50_ms')}/"
            f"{case.get('ttft_p90_ms')}/{case.get('ttft_p95_ms')} "
            f"| {case['input_throughput_tokens_per_second']} "
            f"| {case['input_throughput_per_card_tokens_per_second']} "
            f"| {requests} | {case['report_state']} | {case['source_run_id']} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Comparison | Dataset | Generator | TTFT improvement | Input throughput change | Per-card change | Validated |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['name']} | {item['dataset']} "
            f"| {item.get('data_generator', 'script')} "
            f"| {item['ttft_improvement_percent']}% "
            f"| {item['input_throughput_change_percent']}% "
            f"| {item['per_card_throughput_change_percent']}% "
            f"| {item['comparison_validated']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "`measurement_complete_validation_failed` means all 64 measured requests completed and the performance numbers are retained, but the case is not a validated pass. It must not be used as final acceptance evidence until rerun.",
        ]
    )
    (output_dir / "performance_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--overlay-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_reports(
        args.output_dir.resolve(),
        args.base_run.resolve(),
        [run.resolve() for run in args.overlay_run],
    )
    print(f"CPP_MERGED_PERFORMANCE_REPORT output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
