#!/usr/bin/env python3
"""Create JSON, CSV, and Markdown summaries for one CPP validation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "run.json").is_file():
        parser.error(f"run.json not found under {run_dir}")

    cases = []
    for case_dir in sorted((run_dir / "cases").iterdir()):
        if not case_dir.is_dir() or not (case_dir / "case.json").is_file():
            continue
        case = load_json(case_dir / "case.json")
        status_path = case_dir / "status.json"
        result_path = case_dir / "results/functional_result.json"
        requests_path = case_dir / "raw/requests.json"
        status = load_json(status_path) if status_path.is_file() else {"state": "incomplete", "exit_code": None}
        result = load_json(result_path) if result_path.is_file() else {}
        requests = load_json(requests_path) if requests_path.is_file() else []
        cases.append({
            "case_id": case["case_id"],
            "suite": case.get("suite"),
            "runner": case.get("runner"),
            "cpp_mode": case.get("cpp_mode"),
            "execution_mode": case.get("execution_mode"),
            "request_mode": case.get("request_mode"),
            "data_generator": case.get("data_generator", "script"),
            "request_count": len(requests),
            "expected_request_count": case.get("expected_request_count"),
            "max_fit_chunk": case.get("max_fit_chunk"),
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "scheduler_iterations": result.get("scheduler_iteration_count"),
            "chunk_sizes": result.get("scheduled_chunk_sizes", []),
        })

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
    }
    (reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    columns = [
        "case_id", "suite", "runner", "cpp_mode", "execution_mode",
        "request_mode", "data_generator", "request_count", "expected_request_count", "max_fit_chunk",
        "state", "exit_code", "scheduler_iterations", "chunk_sizes",
    ]
    with (reports / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["chunk_sizes"] = ";".join(map(str, row["chunk_sizes"]))
            writer.writerow(row)

    lines = [
        f"# CPP run {summary['run_id']}", "",
        f"Cases: {len(cases)}; passed: {len(cases) - failed}; failed/incomplete: {failed}.", "",
        "| Case | Mode | Execution | Request mode | Generator | Requests | Max fit chunk | Status |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['case_id']} | {case['cpp_mode']} | {case['execution_mode']} "
            f"| {case['request_mode']} | {case['data_generator']} "
            f"| {case['request_count']} "
            f"| {case['max_fit_chunk']} | {case['state']} |"
        )
    (reports / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CPP_RUN_SUMMARY run={run_dir} cases={len(cases)} failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
