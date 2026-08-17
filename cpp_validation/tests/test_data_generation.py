from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

from cpp_validation.workloads import data_generation
from cpp_validation.workloads.aisbench_auto_tools import (
    build_command,
    validate_aisbench_run,
)
from cpp_validation.workloads.performance.dataset import load_generated_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_suite_defaults_select_aisbench_generation():
    for name in ("functional.json", "performance.json"):
        path = PROJECT_ROOT / "cpp_validation" / "configs" / "suites" / name
        suite = json.loads(path.read_text(encoding="utf-8"))
        assert suite["data_generator"] == "aisbench"


def test_aisbench_backend_preserves_requested_length_plan(monkeypatch, tmp_path):
    lengths = [4096, 32768, 65536]
    prompts = ["first", "second", "third"]
    observed = {}

    def fake_generate(**kwargs):
        observed.update(kwargs)
        return prompts

    monkeypatch.setattr(
        data_generation, "generate_with_aisbench_auto_tools", fake_generate
    )
    monkeypatch.setattr(data_generation, "tool_revision", lambda _: "revision")
    monkeypatch.setattr(
        data_generation,
        "repair_aisbench_prompt_lengths",
        lambda model_path, actual_prompts, actual_lengths: (actual_prompts, []),
    )
    monkeypatch.setattr(
        data_generation,
        "validate_prompts",
        lambda model_path, actual_prompts, actual_lengths: actual_lengths,
    )

    records, metadata = data_generation.generate_records(
        backend="aisbench",
        model_path="/model",
        lengths=lengths,
        output_tokens=1,
        seed=811,
        tool_root=tmp_path,
    )

    assert observed["lengths"] == lengths
    assert [record["question"] for record in records] == prompts
    assert [record["expected_input_tokens"] for record in records] == lengths
    assert metadata["backend"] == "aisbench"
    assert metadata["aisbench_auto_tools_revision"] == "revision"
    assert metadata["total_input_tokens"] == sum(lengths)


def test_script_backend_remains_available(monkeypatch):
    monkeypatch.setattr(
        data_generation,
        "generate_with_script",
        lambda **kwargs: ["legacy"] * len(kwargs["lengths"]),
    )
    monkeypatch.setattr(
        data_generation,
        "validate_prompts",
        lambda model_path, prompts, lengths: lengths,
    )

    records, metadata = data_generation.generate_records(
        backend="script",
        model_path="/model",
        lengths=[16, 32],
        output_tokens=1,
        seed=811,
    )

    assert len(records) == 2
    assert metadata["backend"] == "script"
    assert metadata["aisbench_auto_tools_revision"] is None


def test_high_prefix_hit_preserves_variable_lengths(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(token_ids)

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    prompts, prefix_lengths = data_generation.apply_shared_prefix(
        "/model", ["a" * 10, "b" * 20, "c" * 30], [10, 20, 30], 0.9
    )
    actual_prefix_lengths = data_generation.validate_shared_prefixes(
        "/model", prompts, prefix_lengths
    )

    assert [len(prompt) for prompt in prompts] == [10, 20, 30]
    assert prefix_lengths == [9, 18, 27]
    assert actual_prefix_lengths == prefix_lengths
    assert prompts[0][:9] == prompts[1][:9] == prompts[2][:9]
    assert data_generation.parse_prefix_repeat_rate("90%") == 0.9


def test_generated_jsonl_loader_checks_required_fields(tmp_path):
    path = tmp_path / "dataset.jsonl"
    record = {
        "question": "prompt",
        "answer": "",
        "max_out_len": 1,
        "expected_input_tokens": 16,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert load_generated_records(path) == [record]


def test_manual_warmup_command_matches_requested_contract():
    command = build_command(
        input_len=131072,
        output_len=1,
        data_num=5,
        concurrency=1,
        request_rate="0",
    )

    assert command == [
        sys.executable,
        "aisbench_test.py",
        "--input_len",
        "131072",
        "--output_len",
        "1",
        "--data_num",
        "5",
        "--concurrency",
        "1",
        "--request_rate",
        "0",
    ]


def test_reference_variable_and_prefix_commands_are_supported():
    common = {
        "input_len": 32768,
        "output_len": 300,
        "data_num": 48,
        "concurrency": 12,
        "request_rate": "0",
        "length_mean": 32768,
        "length_std": 49152,
        "length_min": 8192,
        "length_max": 131072,
    }

    normal = build_command(**common)
    prefix = build_command(
        **common,
        dataset_type="prefix_cache",
        repeat_rate="90%",
        prefix_test=True,
    )

    assert normal[-8:] == [
        "--length_mean",
        "32768",
        "--length_std",
        "49152",
        "--length_min",
        "8192",
        "--length_max",
        "131072",
    ]
    assert "--dataset_type" in prefix
    assert "prefix_cache" in prefix
    assert "--repeat_rate" in prefix
    assert "90%" in prefix
    assert "--prefix_test" in prefix


def test_performance_manual_warmup_defaults():
    path = (
        PROJECT_ROOT
        / "cpp_validation"
        / "configs"
        / "suites"
        / "performance.json"
    )
    warmup = json.loads(path.read_text(encoding="utf-8"))["manual_warmup"]

    assert warmup == {
        "enabled": True,
        "input_tokens": 131072,
        "output_tokens": 1,
        "request_count": 5,
        "concurrency": 1,
        "request_rate": 0,
    }


def test_manual_warmup_rejects_upstream_pipeline_false_success(tmp_path):
    details = tmp_path / "outputs" / "run" / "performances" / "model"
    details.mkdir(parents=True)
    (details / "gsm8k_details.jsonl").write_text(
        json.dumps(
            {
                "success": False,
                "error_info": "connection refused",
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evidence, errors = validate_aisbench_run(
        tmp_path,
        expected_requests=1,
        expected_output_tokens=1,
    )

    assert evidence["failed_requests"] == 1
    assert errors == [
        "AISBench warmup failed requests=1: ['connection refused']"
    ]
