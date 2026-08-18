from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from cpp_validation.analysis.summarize_performance import (
    build_comparisons,
    resolved_case_state,
)
from cpp_validation.validators.performance.result import validate_case
from cpp_validation.workloads.performance.dataset import (
    FIXED_INPUT_TOKENS,
    FIXED_REQUEST_COUNT,
    REQUEST_COUNT,
    VARIABLE_REQUEST_COUNT,
    VARIABLE_MAX_TOKENS,
    VARIABLE_MEAN_TOKENS,
    VARIABLE_MIN_TOKENS,
    performance_input_lengths,
    variable_input_lengths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_passed_status_without_result_is_incomplete() -> None:
    assert resolved_case_state({"state": "passed"}, result_exists=False) == "incomplete"
    assert resolved_case_state({"state": "passed"}, result_exists=True) == "passed"


def _matrix_rows(name: str) -> list[tuple[str, str, str]]:
    matrix = PROJECT_ROOT / "cpp_validation" / "configs" / "matrices" / name
    return [
        tuple(line.split())
        for line in matrix.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def test_mrv2_performance_matrix_has_cpp_off_eager_controls():
    expected = [
        ("mrv2", "0", "eager"),
        ("mrv2", "1", "eager"),
        ("mrv2", "1", "graph"),
    ]

    assert _matrix_rows("performance_mrv2.tsv") == expected
    full_rows = _matrix_rows("performance.tsv")
    assert all(
        (*row, dataset) in full_rows
        for row in expected
        for dataset in ("fixed", "variable")
    )


def test_full_matrix_runs_in_requested_cross_dataset_order():
    assert _matrix_rows("performance.tsv") == [
        ("mrv2", "0", "eager", "variable"),
        ("mrv1", "1", "eager", "variable"),
        ("mrv2", "1", "eager", "variable"),
        ("mrv2", "0", "eager", "fixed"),
        ("mrv1", "1", "eager", "fixed"),
        ("mrv2", "1", "eager", "fixed"),
        ("mrv2", "1", "graph", "fixed"),
        ("mrv1", "0", "eager", "variable"),
        ("mrv1", "0", "eager", "fixed"),
        ("mrv2", "1", "graph", "variable"),
    ]


def test_performance_load_uses_requested_pressure_without_changing_lengths():
    suite = json.loads(
        (
            PROJECT_ROOT
            / "cpp_validation"
            / "configs"
            / "suites"
            / "performance.json"
        ).read_text(encoding="utf-8")
    )

    assert suite["max_output_tokens"] == 1
    assert suite["cpp"]["smooth_factor"] == 0.8
    assert suite["manual_warmup"] == {
        "enabled": "auto",
        "dataset_mode": "generated",
        "request_count": 30,
        "seed_offset": 1,
    }
    assert suite["fixed"]["input_tokens"] == 131072
    assert suite["fixed"]["request_count"] == 5
    assert suite["fixed"]["concurrency"] == 1
    assert suite["fixed"]["request_rate"] == 0
    assert suite["fixed"]["max_num_batched_tokens"] == 32768
    assert suite["fixed"]["prefix_cache"]["enabled"] is False
    assert suite["variable"]["min_input_tokens"] == 4096
    assert suite["variable"]["max_input_tokens"] == 65536
    assert suite["variable"]["mean_input_tokens"] == 32768
    assert suite["variable"]["request_count"] == 64
    assert suite["variable"]["concurrency"] == 4
    assert suite["variable"]["request_rate"] == 0
    assert suite["variable"]["max_num_batched_tokens"] == 20480
    assert suite["variable"]["prefix_cache"] == {
        "enabled": True,
        "dataset_type": "prefix_cache",
        "repeat_rate": "90%",
        "prefix_test": True,
    }


def test_shell_loader_exposes_dataset_specific_tuning():
    common = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "common.sh"
    loader = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "config.sh"
    model = (
        PROJECT_ROOT
        / "cpp_validation"
        / "configs"
        / "models"
        / "deepseek_v4_flash.json"
    )
    suite = (
        PROJECT_ROOT / "cpp_validation" / "configs" / "suites" / "performance.json"
    )
    command = (
        f"source {common}; source {loader}; "
        f"cpp_load_performance_config_defaults {model} {suite}; "
        "printf '%s\\n' \"${CPP_PERF_SUITE_DEFAULTS[@]}\""
    )
    completed = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    values = completed.stdout.splitlines()

    assert len(values) == 24
    assert values[3:7] == ["5", "1", "0", "32768"]
    assert values[10:14] == ["64", "4", "0", "20480"]
    assert values[16:] == ["auto", "generated", "1", "0.8", "0", "1", "90%", "1"]


def test_summary_compares_mrv2_cpp_on_with_same_runner_cpp_off():
    def result(dataset: str, cpp_mode: str, ttft: float, throughput: float) -> dict:
        return {
            "case_id": f"mrv2_{cpp_mode}_{dataset}",
            "dataset": dataset,
            "runner": "mrv2",
            "cpp_mode": cpp_mode,
            "execution_mode": "eager",
            "average_ttft_ms": ttft,
            "input_throughput_tokens_per_second": throughput,
            "input_throughput_per_card_tokens_per_second": throughput / 8,
        }

    cases = []
    for dataset in ("fixed", "variable"):
        cases.extend(
            [
                result(dataset, "static", 100, 800),
                result(dataset, "dynamic", 90, 880),
            ]
        )

    comparisons = build_comparisons(cases)

    assert [item["name"] for item in comparisons] == [
        "cpp_mrv2_vs_mrv2_static",
        "cpp_mrv2_vs_mrv2_static",
    ]
    assert [item["dataset"] for item in comparisons] == ["fixed", "variable"]
    assert all(item["ttft_improvement_percent"] == 10 for item in comparisons)
    assert all(item["input_throughput_change_percent"] == 10 for item in comparisons)


def test_summary_never_compares_different_data_generators():
    baseline = {
        "case_id": "script_static",
        "dataset": "fixed",
        "data_generator": "script",
        "runner": "mrv2",
        "cpp_mode": "static",
        "execution_mode": "eager",
        "average_ttft_ms": 100,
        "input_throughput_tokens_per_second": 800,
        "input_throughput_per_card_tokens_per_second": 100,
    }
    current = {
        **baseline,
        "case_id": "aisbench_dynamic",
        "data_generator": "aisbench",
        "cpp_mode": "dynamic",
    }

    assert build_comparisons([baseline, current]) == []


def test_variable_dataset_contract():
    lengths = variable_input_lengths()

    assert len(lengths) == VARIABLE_REQUEST_COUNT == REQUEST_COUNT
    assert min(lengths) == VARIABLE_MIN_TOKENS
    assert max(lengths) == VARIABLE_MAX_TOKENS
    assert sum(lengths) / len(lengths) == VARIABLE_MEAN_TOKENS
    assert lengths == variable_input_lengths()


def test_fixed_dataset_contract():
    lengths = performance_input_lengths("fixed")

    assert lengths == [FIXED_INPUT_TOKENS] * FIXED_REQUEST_COUNT


def _write_aisbench_result(root, dataset="fixed"):
    output = root / "performances" / "cpp-vllm-api"
    output.mkdir(parents=True)
    lengths = performance_input_lengths(dataset)
    request_count = len(lengths)
    benchmark_duration_ms = sum(lengths) / 800 * 1000
    common = {
        "Benchmark Duration": {"Total": f"{benchmark_duration_ms} ms"},
        "Input Token Throughput": {"Total": "800 token/s"},
        "Success Requests": {"Total": request_count},
        "Failed Requests": {"Total": 0},
        "Total Input Tokens": {"Total": sum(lengths)},
        "Total Generated Tokens": {"Total": request_count},
    }
    (output / f"cpp_{dataset}.json").write_text(json.dumps(common), encoding="utf-8")
    with (output / f"cpp_{dataset}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Performance Parameters",
                "Stage",
                "Average",
                "Min",
                "Max",
                "P50",
                "P75",
                "P90",
                "P95",
                "P99",
                "N",
            ]
        )
        writer.writerow(
            [
                "TTFT",
                "Total",
                "123 ms",
                "",
                "",
                "120 ms",
                "",
                "130 ms",
                "140 ms",
                "",
                request_count,
            ]
        )


def _trace_line(payload):
    return f"INFO [CPP_EXECUTION_MODE_TRACE] {json.dumps(payload)}\n"


def test_graph_case_validates_eager_profile_and_graph_probe(tmp_path):
    aisbench = tmp_path / "aisbench"
    _write_aisbench_result(aisbench)
    server_log = tmp_path / "server.log"
    lines = [
        _trace_line(
            {
                "event": "startup_profile_warmup_execution_mode",
                "runner": "mrv2",
                "configured_cudagraph_mode": "FULL_DECODE_ONLY",
                "actual_execution_mode": "NONE",
                "need_eager": True,
                "pp_rank": 0,
                "tp_rank": 0,
            }
        )
    ]
    lines.extend(
        [
        _trace_line(
            {
                "event": "startup_profile_execution_mode",
                "runner": "mrv2",
                "sample_index": index,
                "configured_cudagraph_mode": "FULL_DECODE_ONLY",
                "actual_execution_mode": "NONE",
                "need_eager": True,
                "pp_rank": 0,
                "tp_rank": 0,
            }
        )
        for index in range(64)
        ]
    )
    lines.append(
        _trace_line(
            {
                "event": "inference_execution_mode_observed",
                "runner": "mrv2",
                "configured_cudagraph_mode": "FULL_DECODE_ONLY",
                "actual_execution_mode": "FULL",
                "need_eager": False,
                "pp_rank": 0,
                "tp_rank": 0,
            }
        )
    )
    server_log.write_text("".join(lines), encoding="utf-8")
    graph_probe = tmp_path / "graph_probe.json"
    graph_probe.write_text(
        json.dumps(
            {
                "request_count": 1,
                "expected_output_tokens": 2,
                "records": [{"prompt_tokens": 128, "completion_tokens": 2}],
                "observed_execution_modes": ["FULL"],
                "observed_graph": True,
            }
        ),
        encoding="utf-8",
    )

    result, errors = validate_case(
        aisbench_root=aisbench,
        server_log=server_log,
        dataset="fixed",
        runner="mrv2",
        dynamic=True,
        execution_mode="graph",
        pp_size=2,
        tp_size=4,
        graph_probe=graph_probe,
    )

    assert not errors
    assert result["average_ttft_ms"] == 123
    assert result["ttft_p50_ms"] == 120
    assert result["ttft_p90_ms"] == 130
    assert result["ttft_p95_ms"] == 140
    assert result["measurement_wall_time_seconds"] == 819.2
    assert result["input_throughput_tokens_per_second"] == 800
    assert result["input_throughput_per_card_tokens_per_second"] == 100
    assert result["configured_cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert result["startup_profile_sample_count"] == 64
    assert result["startup_profile_warmup_count"] == 1
    assert result["startup_profile_all_eager"] is True
    assert result["inference_observed_modes"] == ["FULL"]
    assert result["cpp_graph_isolation_pass"] is True


def test_graph_case_rejects_silent_eager_fallback(tmp_path):
    aisbench = tmp_path / "aisbench"
    _write_aisbench_result(aisbench)
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "".join(
            _trace_line(
                {
                    "event": "startup_profile_execution_mode",
                    "runner": "mrv2",
                    "sample_index": index,
                    "configured_cudagraph_mode": "NONE",
                    "actual_execution_mode": "NONE",
                    "need_eager": True,
                    "pp_rank": 0,
                    "tp_rank": 0,
                }
            )
            for index in range(64)
        ),
        encoding="utf-8",
    )
    graph_probe = tmp_path / "graph_probe.json"
    graph_probe.write_text(
        json.dumps(
            {
                "request_count": 1,
                "expected_output_tokens": 2,
                "records": [{"prompt_tokens": 128, "completion_tokens": 2}],
                "observed_execution_modes": [],
                "observed_graph": False,
            }
        ),
        encoding="utf-8",
    )

    result, errors = validate_case(
        aisbench_root=aisbench,
        server_log=server_log,
        dataset="fixed",
        runner="mrv2",
        dynamic=True,
        execution_mode="graph",
        pp_size=2,
        tp_size=4,
        graph_probe=graph_probe,
    )

    assert errors
    assert result["passed"] is False
    assert result["cpp_graph_isolation_pass"] is False


def test_variable_case_requires_and_records_prefix_prime(tmp_path):
    aisbench = tmp_path / "aisbench"
    _write_aisbench_result(aisbench, dataset="variable")
    server_log = tmp_path / "server.log"
    server_log.write_text("", encoding="utf-8")
    lengths = performance_input_lengths("variable")
    prefix_lengths = [round(length * 0.9) for length in lengths]
    cacheable_prefix_lengths = [length // 32 * 32 for length in prefix_lengths]
    dataset_metadata = tmp_path / "dataset.json"
    dataset_metadata.write_text(
        json.dumps(
            {
                "backend": "aisbench",
                "input_token_lengths": lengths,
                "prefix_cache": {
                    "enabled": True,
                    "prefix_test": True,
                    "mean_planned_prefix_hit_ratio": sum(
                        prefix / length
                        for prefix, length in zip(prefix_lengths, lengths, strict=True)
                    )
                    / len(lengths),
                    "mean_cacheable_prefix_hit_ratio": sum(
                        prefix / length
                        for prefix, length in zip(
                            cacheable_prefix_lengths, lengths, strict=True
                        )
                    )
                    / len(lengths),
                    "planned_prefix_token_lengths": prefix_lengths,
                    "cacheable_prefix_token_lengths": cacheable_prefix_lengths,
                },
            }
        ),
        encoding="utf-8",
    )
    prefix_prime = tmp_path / "prefix_prime.json"
    prefix_prime.write_text(
        json.dumps(
            {
                "mode": "prefix-prime",
                "request_count": 1,
                "expected_prefix_tokens": max(cacheable_prefix_lengths),
                "records": [
                    {
                        "prompt_tokens": max(cacheable_prefix_lengths),
                        "completion_tokens": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result, errors = validate_case(
        aisbench_root=aisbench,
        server_log=server_log,
        dataset="variable",
        runner="mrv2",
        dynamic=False,
        execution_mode="eager",
        pp_size=2,
        tp_size=4,
        graph_probe=None,
        data_generator="aisbench",
        dataset_metadata=dataset_metadata,
        prefix_prime=prefix_prime,
    )

    assert not errors
    assert result["prefix_prime"]["expected_prefix_tokens"] == max(
        cacheable_prefix_lengths
    )
