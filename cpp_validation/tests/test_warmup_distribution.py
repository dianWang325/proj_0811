from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_performance_case_metadata_records_cross_warmup(tmp_path: Path) -> None:
    artifacts = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "artifacts.sh"
    case_dir = tmp_path / "case"
    values = {
        "RUNNER": "mrv2",
        "DYNAMIC": "1",
        "EXECUTION_MODE": "eager",
        "PERF_DATASET": "variable",
        "SERVER_PORT": "18080",
        "NPU_DEVICES": "0,1,2,3,4,5,6,7",
        "HCCL_PORT_RANGE": "17000-17100",
        "PIPELINE_PARALLEL_SIZE": "2",
        "TENSOR_PARALLEL_SIZE": "4",
        "MAX_MODEL_LEN": "132000",
        "MAX_NUM_BATCHED_TOKENS": "20480",
        "REQUEST_COUNT": "64",
        "WARMUP_COUNT": "5",
        "CONCURRENCY": "4",
        "REQUEST_RATE": "0",
        "MAX_OUTPUT_TOKENS": "1",
        "DATA_GENERATOR": "aisbench",
        "MANUAL_WARMUP_ENABLED": "1",
        "MANUAL_WARMUP_ENABLED_CONFIG": "auto",
        "MANUAL_WARMUP_DATASET_MODE": "generated",
        "MANUAL_WARMUP_PERF_DATASET": "fixed",
        "MANUAL_WARMUP_CONCURRENCY": "1",
        "MANUAL_WARMUP_SEED": "812",
        "PREFIX_CACHE_ENABLED": "1",
        "PREFIX_REPEAT_RATE": "90%",
        "PREFIX_TEST": "1",
        "CPP_SMOOTH_FACTOR": "0.8",
        "NEED_TIMING": "true",
        "ASYNC_SCHEDULING": "0",
        "MATRIX_ROUND": "1",
        "MATRIX_POSITION": "3",
        "MODEL_ID": "deepseek-v4-flash-w4a8",
        "MODEL_FAMILY": "deepseek",
        "TOKENIZER_PATH": "/models/deepseek",
        "TOKENIZER_MODE": "deepseek_v4",
        "TOKENIZER_TRUST_REMOTE_CODE": "1",
        "QUANTIZATION": "ascend",
        "GPU_MEMORY_UTILIZATION": "0.9",
        "KV_CACHE_MEMORY": "",
        "ENABLE_EXPERT_PARALLEL": "0",
        "SAFETENSORS_LOAD_STRATEGY": "auto",
        "API_MODE": "completions",
        "PROMPT_MODE": "raw",
    }
    assignments = " ".join(f"{key}={value!s}" for key, value in values.items())
    subprocess.run(
        [
            "bash",
            "-c",
            f"{assignments}; source {artifacts}; "
            f"cpp_initialize_performance_case {case_dir} example",
        ],
        check=True,
    )
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert metadata["manual_warmup"] == {
        "enabled": True,
        "configured_enabled": "auto",
        "dataset_mode": "generated",
        "dataset": "fixed",
        "seed": 812,
        "request_count": 5,
        "concurrency": 1,
        "request_rate": 0.0,
        "output_tokens": 1,
    }


def test_warmup_distribution_matrix_dry_run_has_ten_unique_cases() -> None:
    script = (
        PROJECT_ROOT
        / "cpp_validation"
        / "scripts"
        / "run_perf_warmup_distribution_matrix.sh"
    )
    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CPP_WARMUP_MATRIX_DRY_RUN": "1"},
    )
    lines = completed.stdout.splitlines()
    rows = list(csv.DictReader(lines[1:], delimiter="\t"))

    assert len(rows) == 10
    assert len({row["label"] for row in rows}) == 10
    fixed_warmups = [
        row for row in rows if row["warmup_enabled"] == "1" and row["warmup_dataset"] == "fixed"
    ]
    variable_warmups = [
        row
        for row in rows
        if row["warmup_enabled"] == "1" and row["warmup_dataset"] == "variable"
    ]
    assert {row["warmup_count"] for row in fixed_warmups} == {"5"}
    assert {row["warmup_count"] for row in variable_warmups} == {"30"}
    assert {row["need_timing"] for row in rows if row["cpp"] == "0"} == {"false"}


def test_warmup_summary_flags_primary_degradation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n", encoding="utf-8")
    order_rows = [
        (1, "matched_variable", "matched", "variable", "variable", 30, 4),
        (2, "cross_variable_fixed_c4", "validation", "variable", "fixed", 5, 4),
    ]
    with (run_dir / "matrix_order.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "position",
                "label",
                "category",
                "cpp",
                "measurement_dataset",
                "warmup_enabled",
                "warmup_dataset",
                "warmup_count",
                "warmup_concurrency",
                "need_timing",
            ]
        )
        for position, label, category, dataset, warmup, count, concurrency in order_rows:
            writer.writerow(
                [position, label, category, 1, dataset, 1, warmup, count, concurrency, "true"]
            )

    for position, label, _, dataset, warmup, count, concurrency in order_rows:
        case_dir = cases_dir / label
        (case_dir / "results").mkdir(parents=True)
        (case_dir / "case.json").write_text(
            json.dumps(
                {
                    "case_id": label,
                    "dataset": dataset,
                    "dynamic": True,
                    "async_scheduling": False,
                    "matrix": {"round": 1, "position": position},
                    "manual_warmup": {
                        "enabled": True,
                        "dataset": warmup,
                        "request_count": count,
                        "concurrency": concurrency,
                    },
                    "cpp_tuning": {"need_timing": True},
                }
            ),
            encoding="utf-8",
        )
        (case_dir / "status.json").write_text(
            json.dumps({"state": "passed"}), encoding="utf-8"
        )
        degraded = label.startswith("cross_")
        (case_dir / "results" / "performance_result.json").write_text(
            json.dumps(
                {
                    "average_ttft_ms": 110 if degraded else 100,
                    "input_throughput_per_card_tokens_per_second": (
                        90 if degraded else 100
                    ),
                    "success_requests": 64,
                    "failed_requests": 0,
                }
            ),
            encoding="utf-8",
        )

    script = (
        PROJECT_ROOT
        / "cpp_validation"
        / "analysis"
        / "summarize_warmup_distribution.py"
    )
    subprocess.run(["python3", str(script), str(run_dir)], check=True)
    summary = json.loads(
        (run_dir / "reports" / "warmup_distribution.json").read_text(
            encoding="utf-8"
        )
    )
    primary = next(
        item
        for item in summary["comparisons"]
        if item["name"] == "variable_test_fixed_vs_variable_warmup_c4"
    )
    assert primary["average_ttft_change_percent"] == 10.0
    assert primary["per_card_input_throughput_change_percent"] == -10.0
    assert primary["observed_degradation"] is True
