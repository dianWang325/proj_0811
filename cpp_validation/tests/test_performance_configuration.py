from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from cpp_validation.configuration.performance import (
    CONFIG_ERROR_EXIT_CODE,
    Conflict,
    case_conflicts,
    global_conflicts,
    resolve_model_profile,
    resolve_suite_contract,
    write_conflicts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    PROJECT_ROOT
    / "cpp_validation"
    / "configs"
    / "models"
    / "deepseek_v4_flash.json"
)
SUITE_CONFIG = (
    PROJECT_ROOT / "cpp_validation" / "configs" / "suites" / "performance.json"
)


def _model() -> dict:
    return resolve_model_profile(json.loads(MODEL_CONFIG.read_text(encoding="utf-8")))


def _suite() -> dict:
    return resolve_suite_contract(json.loads(SUITE_CONFIG.read_text(encoding="utf-8")))


def test_model_v2_resolves_model_owned_fields() -> None:
    model = _model()

    assert model["model_id"] == "deepseek-v4-flash-w4a8"
    assert model["model_family"] == "deepseek"
    assert model["tokenizer_path"] == model["model_path"]
    assert model["tokenizer_mode"] == "deepseek_v4"
    assert model["quantization"] == "ascend"
    assert model["safetensors_load_strategy"] == "auto"


def test_legacy_model_fields_remain_a_migration_fallback() -> None:
    model = resolve_model_profile(
        {
            "model_path": "/models/legacy",
            "fallback_model_path": "/models/legacy-backup",
            "served_model_name": "legacy-model",
            "max_model_len": 8192,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 2,
        }
    )

    assert model["schema_version"] == 1
    assert model["model_id"] == "legacy-model"
    assert model["legacy_max_model_len"] == 8192
    assert model["legacy_pipeline_parallel_size"] == 1
    assert model["legacy_tensor_parallel_size"] == 2


def test_invalid_v2_model_id_is_rejected() -> None:
    data = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    data["model_id"] = "DeepSeek V4"

    with pytest.raises(ValueError, match="invalid model_id"):
        resolve_model_profile(data)


def test_global_preflight_checks_only_global_contract(tmp_path: Path) -> None:
    model = _model()
    suite = _suite()
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}\n", encoding="utf-8")

    conflicts = global_conflicts(
        model,
        suite,
        model_path=str(model_root),
        tokenizer_path=str(model_root),
        tokenizer_trust_remote_code=True,
        max_model_len=132000,
        pipeline_parallel_size=2,
        tensor_parallel_size=4,
        npu_devices=list(range(8)),
        quantization="ascend",
        safetensors_load_strategy="auto",
        load_tokenizer=False,
    )

    assert conflicts == []


def test_case_preflight_reports_model_specific_conflicts() -> None:
    model = _model()
    model["model_kind"] = "dense"
    model["supported_execution_modes"] = ["eager"]
    model["supports_prefix_cache"] = False

    conflicts = case_conflicts(
        model,
        execution_mode="graph",
        prefix_cache_enabled=True,
        enable_expert_parallel=True,
        api_mode="completions",
        prompt_mode="raw",
        tokenizer_mode="deepseek_v4",
    )

    assert {conflict.key for conflict in conflicts} == {
        "execution_mode",
        "prefix_cache",
        "expert_parallel",
    }


def test_conflict_log_is_named_for_model_and_key(tmp_path: Path) -> None:
    model = _model()
    outputs = write_conflicts(
        [
            Conflict(
                "execution_mode",
                ["eager"],
                "graph",
                "graph is unsupported",
                "select eager manually",
            )
        ],
        artifact_root=tmp_path,
        model=model,
        model_config=MODEL_CONFIG,
        suite_config=SUITE_CONFIG,
        effective={"scope": "case"},
        project_root=PROJECT_ROOT,
    )

    assert CONFIG_ERROR_EXIT_CODE == 78
    assert len(outputs) == 1
    assert outputs[0].name.startswith(
        "deepseek-v4-flash-w4a8__execution_mode__"
    )
    payload = json.loads(outputs[0].read_text(encoding="utf-8").split("\n", 1)[1])
    assert payload["status"] == "blocked"
    assert payload["conflict"]["resolution"] == "select eager manually"


def test_run_id_inserts_model_once_after_perf_matrix() -> None:
    artifacts = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "artifacts.sh"
    command = (
        f"source {artifacts}; "
        "cpp_run_id_with_model "
        "server112_perf_matrix_20260818_150556_need_timing "
        "deepseek-v4-flash-w4a8; "
        "cpp_run_id_with_model "
        "server112_perf_matrix_deepseek-v4-flash-w4a8_20260818_150556_need_timing "
        "deepseek-v4-flash-w4a8"
    )

    completed = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )

    expected = (
        "server112_perf_matrix_deepseek-v4-flash-w4a8_"
        "20260818_150556_need_timing"
    )
    assert completed.stdout.splitlines() == [expected, expected]


def test_functional_run_metadata_remains_schema_v1(tmp_path: Path) -> None:
    common = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "common.sh"
    artifacts = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "artifacts.sh"
    command = (
        f"PROJECT_ROOT={PROJECT_ROOT}; CPP_ARTIFACT_ROOT={tmp_path}; "
        "MODEL_PATH=/models/functional; MODEL_NAME=functional; NPU_DEVICES=8,9; "
        f"source {common}; source {artifacts}; cpp_initialize_run functional"
    )

    subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    run_json = next((tmp_path / "runs").glob("*/run.json"))
    metadata = json.loads(run_json.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert "base_run_id" not in metadata


def test_automatic_performance_run_id_and_schema_v2(tmp_path: Path) -> None:
    common = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "common.sh"
    artifacts = PROJECT_ROOT / "cpp_validation" / "scripts" / "lib" / "artifacts.sh"
    environment = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CPP_ARTIFACT_ROOT": str(tmp_path),
        "MODEL_CONFIG": str(MODEL_CONFIG),
        "SUITE_CONFIG": str(SUITE_CONFIG),
        "MODEL_PATH": "/models/deepseek",
        "MODEL_NAME": "deepseek-v4-flash",
        "MODEL_ID": "deepseek-v4-flash-w4a8",
        "MODEL_FAMILY": "deepseek",
        "TOKENIZER_PATH": "/tokenizers/deepseek",
        "TOKENIZER_MODE": "deepseek_v4",
        "TOKENIZER_TRUST_REMOTE_CODE": "1",
        "QUANTIZATION": "ascend",
        "GPU_MEMORY_UTILIZATION": "0.9",
        "KV_CACHE_MEMORY": "",
        "ENABLE_EXPERT_PARALLEL": "0",
        "SAFETENSORS_LOAD_STRATEGY": "auto",
        "MAX_MODEL_LEN": "132000",
        "PIPELINE_PARALLEL_SIZE": "2",
        "TENSOR_PARALLEL_SIZE": "4",
        "NPU_DEVICES": "8,9,10,11,12,13,14,15",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f"source {common}; source {artifacts}; cpp_initialize_run performance-matrix",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    latest = (tmp_path / "LATEST").read_text(encoding="utf-8").strip()
    assert latest.startswith("perf_matrix_deepseek-v4-flash-w4a8_")
    assert latest.count("deepseek-v4-flash-w4a8") == 1
    run_dir = tmp_path / "runs" / latest
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["run_id"] == latest
    assert (run_dir / "effective_model_config.json").is_file()
    assert (run_dir / "effective_test_config.json").is_file()
