#!/usr/bin/env python3
"""Summarize the dedicated CPP manual-warmup distribution experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "average_ttft_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "measurement_wall_time_seconds",
    "input_throughput_tokens_per_second",
    "input_throughput_per_card_tokens_per_second",
    "success_requests",
    "failed_requests",
    "startup_profile_sample_count",
    "startup_profile_all_eager",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return round((current - baseline) / baseline * 100.0, 4)


def load_order(run_dir: Path) -> dict[int, dict[str, str]]:
    path = run_dir / "matrix_order.tsv"
    if not path.is_file():
        raise ValueError(f"matrix order is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["position"]): row for row in csv.DictReader(handle, delimiter="\t")}


def collect_cases(run_dir: Path, order: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    cases = []
    cases_root = run_dir / "cases"
    if not cases_root.is_dir():
        return cases
    for case_dir in cases_root.iterdir():
        case_path = case_dir / "case.json"
        if not case_path.is_file():
            continue
        case = load_json(case_path)
        position = case.get("matrix", {}).get("position")
        if position not in order:
            continue
        result_path = case_dir / "results" / "performance_result.json"
        status_path = case_dir / "status.json"
        result = load_json(result_path) if result_path.is_file() else {}
        status = load_json(status_path) if status_path.is_file() else {}
        spec = order[position]
        warmup = case.get("manual_warmup", {})
        row: dict[str, Any] = {
            "position": position,
            "label": spec["label"],
            "category": spec["category"],
            "case_id": case["case_id"],
            "state": status.get("state", "incomplete"),
            "cpp_enabled": case.get("dynamic", False),
            "measurement_dataset": case["dataset"],
            "warmup_enabled": warmup.get("enabled", False),
            "warmup_dataset": warmup.get("dataset"),
            "warmup_count": warmup.get("request_count"),
            "warmup_concurrency": warmup.get("concurrency"),
            "need_timing": case.get("cpp_tuning", {}).get("need_timing"),
            "async_scheduling": case.get("async_scheduling"),
        }
        row.update({field: result.get(field) for field in METRIC_FIELDS})
        successes = row.get("success_requests")
        failures = row.get("failed_requests")
        total = (successes or 0) + (failures or 0)
        row["success_rate_percent"] = (
            round(successes / total * 100.0, 4) if total else None
        )
        cases.append(row)
    return sorted(cases, key=lambda item: item["position"])


def compare(
    name: str,
    current_label: str,
    baseline_label: str,
    index: dict[str, dict[str, Any]],
    comparison_type: str,
) -> dict[str, Any]:
    current = index.get(current_label)
    baseline = index.get(baseline_label)
    if current is None or baseline is None:
        return {
            "name": name,
            "type": comparison_type,
            "current_label": current_label,
            "baseline_label": baseline_label,
            "complete": False,
        }
    ttft_change = percent_change(
        current.get("average_ttft_ms"), baseline.get("average_ttft_ms")
    )
    throughput_change = percent_change(
        current.get("input_throughput_per_card_tokens_per_second"),
        baseline.get("input_throughput_per_card_tokens_per_second"),
    )
    complete = (
        current.get("state") == "passed"
        and baseline.get("state") == "passed"
        and ttft_change is not None
        and throughput_change is not None
    )
    degradation = bool(
        complete
        and ttft_change > 0
        and throughput_change < 0
        and (ttft_change >= 5 or throughput_change <= -5)
    )
    return {
        "name": name,
        "type": comparison_type,
        "current_label": current_label,
        "baseline_label": baseline_label,
        "complete": complete,
        "average_ttft_change_percent": ttft_change,
        "per_card_input_throughput_change_percent": throughput_change,
        "observed_degradation": degradation,
    }


def build_comparisons(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {case["label"]: case for case in cases}
    specs = (
        (
            "variable_test_fixed_vs_variable_warmup_c4",
            "cross_variable_fixed_c4",
            "matched_variable",
            "primary_distribution",
        ),
        (
            "fixed_test_variable_vs_fixed_warmup_c1",
            "cross_fixed_variable_c1",
            "matched_fixed",
            "primary_distribution",
        ),
        (
            "variable_test_fixed_warmup_c1_vs_c4",
            "cross_variable_fixed_c1",
            "cross_variable_fixed_c4",
            "concurrency_sensitivity",
        ),
        (
            "fixed_test_variable_warmup_c4_vs_c1",
            "cross_fixed_variable_c4",
            "cross_fixed_variable_c1",
            "concurrency_sensitivity",
        ),
        (
            "variable_matched_warmup_vs_cpp_off",
            "matched_variable",
            "cpp_off_variable",
            "cpp_context",
        ),
        (
            "fixed_matched_warmup_vs_cpp_off",
            "matched_fixed",
            "cpp_off_fixed",
            "cpp_context",
        ),
        (
            "variable_matched_warmup_vs_no_warmup",
            "matched_variable",
            "no_warmup_variable",
            "warmup_context",
        ),
        (
            "fixed_matched_warmup_vs_no_warmup",
            "matched_fixed",
            "no_warmup_fixed",
            "warmup_context",
        ),
    )
    return [compare(name, current, baseline, index, kind) for name, current, baseline, kind in specs]


def render_markdown(cases: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# CPP Manual-Warmup Distribution Experiment",
        "",
        "Single-run exploratory result. A primary comparison is flagged only when "
        "average TTFT worsens, per-card input throughput worsens, and at least one "
        "change reaches 5%.",
        "",
        "## Cases",
        "",
        "| Pos | Label | State | Test | Warmup | Count | Conc. | Avg TTFT ms | P95 ms | Per-card input tok/s | Success |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            "| {position} | {label} | {state} | {measurement_dataset} | "
            "{warmup_dataset} | {warmup_count} | {warmup_concurrency} | "
            "{average_ttft_ms} | {ttft_p95_ms} | "
            "{input_throughput_per_card_tokens_per_second} | "
            "{success_rate_percent} |".format(**case)
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Comparison | Type | Complete | Avg TTFT change | Per-card throughput change | Degradation |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for item in comparisons:
        ttft = item.get("average_ttft_change_percent")
        throughput = item.get("per_card_input_throughput_change_percent")
        lines.append(
            f"| {item['name']} | {item['type']} | {item['complete']} | "
            f"{ttft if ttft is not None else '—'}% | "
            f"{throughput if throughput is not None else '—'}% | "
            f"{item.get('observed_degradation', False)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "run.json").is_file():
        parser.error(f"run.json is missing under {run_dir}")

    order = load_order(run_dir)
    cases = collect_cases(run_dir, order)
    comparisons = build_comparisons(cases)
    primary = [item for item in comparisons if item["type"] == "primary_distribution"]
    summary = {
        "schema_version": 1,
        "experiment": "cpp_manual_warmup_distribution",
        "exploratory_single_run": True,
        "expected_case_count": len(order),
        "observed_case_count": len(cases),
        "passed_case_count": sum(case["state"] == "passed" for case in cases),
        "primary_comparisons_complete": all(item["complete"] for item in primary),
        "observed_primary_degradation": any(
            item.get("observed_degradation", False) for item in primary
        ),
        "cases": cases,
        "comparisons": comparisons,
    }
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "warmup_distribution.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(cases[0]) if cases else ["position", "label", "state"]
    with (reports / "warmup_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    (reports / "warmup_distribution.md").write_text(
        render_markdown(cases, comparisons), encoding="utf-8"
    )
    print(
        "CPP_WARMUP_DISTRIBUTION_SUMMARY "
        f"cases={len(cases)}/{len(order)} "
        f"passed={summary['passed_case_count']} "
        f"output={reports / 'warmup_distribution.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
