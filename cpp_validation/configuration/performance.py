#!/usr/bin/env python3
"""Resolve performance model profiles and fail fast on incompatible tests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_ERROR_EXIT_CODE = 78
MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class Conflict:
    key: str
    model_value: Any
    test_value: Any
    rule: str
    resolution: str


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON configuration {source}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a JSON object: {source}")
    return data


def slugify(value: str) -> str:
    rendered = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
    rendered = re.sub(r"-+", "-", rendered).strip("-._")
    return rendered or "unknown-model"


def resolve_model_profile(data: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(data.get("schema_version", 1))
    model_path = str(data.get("model_path", ""))
    raw_model_id = str(
        data.get("model_id") or data.get("served_model_name") or Path(model_path).name
    )
    if schema_version >= 2 and not MODEL_ID_PATTERN.fullmatch(raw_model_id):
        raise ValueError(f"invalid model_id: {raw_model_id}")
    model_id = raw_model_id if schema_version >= 2 else slugify(raw_model_id)
    tokenizer = data.get("tokenizer") or {}
    if not isinstance(tokenizer, dict):
        raise ValueError("model tokenizer configuration must be an object")
    weight_loading = data.get("weight_loading") or {}
    constraints = data.get("constraints") or {}
    if not isinstance(weight_loading, dict) or not isinstance(constraints, dict):
        raise ValueError("model weight_loading and constraints must be objects")
    max_context = constraints.get("max_context_tokens", data.get("max_model_len"))
    if not model_path or not data.get("served_model_name") or max_context is None:
        raise ValueError("model configuration requires model_path, served_model_name, and max context")
    return {
        "schema_version": schema_version,
        "model_id": model_id,
        "model_family": str(data.get("model_family", "unknown")),
        "model_path": model_path,
        "fallback_model_path": str(data.get("fallback_model_path", "")),
        "served_model_name": str(data["served_model_name"]),
        "tokenizer_path": str(tokenizer.get("path") or model_path),
        "tokenizer_mode": str(tokenizer.get("mode", "deepseek_v4" if schema_version == 1 else "auto")),
        "tokenizer_trust_remote_code": bool(tokenizer.get("trust_remote_code", True)),
        "quantization": data.get("quantization", "ascend"),
        "safetensors_load_strategy": str(weight_loading.get("safetensors_load_strategy", "auto")),
        "allowed_safetensors_load_strategies": list(
            weight_loading.get("allowed_safetensors_load_strategies", ["auto"])
        ),
        "max_context_tokens": int(max_context),
        "model_kind": str(constraints.get("model_kind", "unknown")),
        "supported_execution_modes": list(constraints.get("supported_execution_modes", ["eager", "graph"])),
        "supports_prefix_cache": bool(constraints.get("supports_prefix_cache", True)),
        "supported_api_modes": list(constraints.get("supported_api_modes", ["completions"])),
        "supported_prompt_modes": list(constraints.get("supported_prompt_modes", ["raw"])),
        "supported_tokenizer_modes": list(
            constraints.get("supported_tokenizer_modes", [str(tokenizer.get("mode", "auto"))])
        ),
        "allowed_pipeline_parallel_sizes": [int(value) for value in constraints.get("allowed_pipeline_parallel_sizes", [])],
        "allowed_tensor_parallel_sizes": [int(value) for value in constraints.get("allowed_tensor_parallel_sizes", [])],
        "legacy_max_model_len": data.get("max_model_len"),
        "legacy_pipeline_parallel_size": data.get("pipeline_parallel_size"),
        "legacy_tensor_parallel_size": data.get("tensor_parallel_size"),
    }


def resolve_suite_contract(data: dict[str, Any]) -> dict[str, Any]:
    serving = data.get("serving") or {}
    fixed = data["fixed"]
    variable = data["variable"]
    return {
        "max_output_tokens": int(data["max_output_tokens"]),
        "max_model_len": int(serving.get("max_model_len", 132000)),
        "pipeline_parallel_size": int(serving.get("pipeline_parallel_size", 2)),
        "tensor_parallel_size": int(serving.get("tensor_parallel_size", 4)),
        "api_mode": str(serving.get("api_mode", "completions")),
        "prompt_mode": str(serving.get("prompt_mode", "raw")),
        "fixed_max_input_tokens": int(fixed["input_tokens"]),
        "variable_max_input_tokens": int(variable["max_input_tokens"]),
    }


def global_conflicts(
    model: dict[str, Any],
    suite: dict[str, Any],
    *,
    model_path: str,
    tokenizer_path: str,
    tokenizer_trust_remote_code: bool,
    max_model_len: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    npu_devices: list[int],
    quantization: str | None,
    safetensors_load_strategy: str,
    dataset: str | None = None,
    load_tokenizer: bool = True,
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    model_root = Path(model_path)
    if not (model_root / "config.json").is_file():
        conflicts.append(Conflict(
            "model_path", model["model_path"], model_path,
            "the selected model directory must contain a readable config.json",
            "set CPP_MODEL_PATH to the intended model directory or repair the selected model profile",
        ))
    tokenizer_root = Path(tokenizer_path)
    if not tokenizer_root.exists():
        conflicts.append(Conflict(
            "tokenizer_path", model["tokenizer_path"], tokenizer_path,
            "the selected tokenizer path must exist",
            "repair tokenizer.path in the model profile or explicitly set CPP_TOKENIZER_PATH",
        ))
    elif load_tokenizer:
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=tokenizer_trust_remote_code,
            )
        except Exception as error:  # noqa: BLE001 - diagnostic boundary
            conflicts.append(Conflict(
                "tokenizer_readable", model["tokenizer_mode"], tokenizer_path,
                f"the tokenizer must load locally before dataset generation: {error}",
                "repair the tokenizer files or select the correct model profile; do not change test settings automatically",
            ))

    if max_model_len > model["max_context_tokens"]:
        conflicts.append(Conflict(
            "max_model_len", model["max_context_tokens"], max_model_len,
            "test max_model_len cannot exceed the model context capability",
            "manually lower CPP_MAX_MODEL_LEN or select a model with sufficient context",
        ))
    dataset_limits = {
        "fixed": suite["fixed_max_input_tokens"],
        "variable": suite["variable_max_input_tokens"],
    }
    selected = dataset_limits.items() if dataset is None else [(dataset, dataset_limits[dataset])]
    for dataset_name, max_input in selected:
        required = max_input + suite["max_output_tokens"]
        if required > max_model_len or required > model["max_context_tokens"]:
            conflicts.append(Conflict(
                f"{dataset_name}_context", model["max_context_tokens"], required,
                "dataset input plus output must fit both test max_model_len and model context",
                "manually select a compatible test suite or model; do not truncate prompts automatically",
            ))

    device_count = len(npu_devices)
    expected_devices = pipeline_parallel_size * tensor_parallel_size
    if device_count != expected_devices:
        conflicts.append(Conflict(
            "parallel_devices", expected_devices, device_count,
            "NPU device count must equal pipeline_parallel_size multiplied by tensor_parallel_size",
            "manually align CPP_NPU_DEVICES, CPP_PIPELINE_PARALLEL_SIZE, and CPP_TENSOR_PARALLEL_SIZE",
        ))
    allowed_pp = model["allowed_pipeline_parallel_sizes"]
    allowed_tp = model["allowed_tensor_parallel_sizes"]
    if allowed_pp and pipeline_parallel_size not in allowed_pp:
        conflicts.append(Conflict(
            "pipeline_parallel_size", allowed_pp, pipeline_parallel_size,
            "pipeline parallel size is outside the model-declared compatibility set",
            "choose a declared PP size or update the model capability after manual validation",
        ))
    if allowed_tp and tensor_parallel_size not in allowed_tp:
        conflicts.append(Conflict(
            "tensor_parallel_size", allowed_tp, tensor_parallel_size,
            "tensor parallel size is outside the model-declared compatibility set",
            "choose a declared TP size or update the model capability after manual validation",
        ))
    if quantization != model["quantization"]:
        conflicts.append(Conflict(
            "quantization", model["quantization"], quantization,
            "effective quantization must match the selected model profile",
            "manually select the matching model profile or correct CPP_QUANTIZATION",
        ))
    allowed_loaders = model["allowed_safetensors_load_strategies"]
    if allowed_loaders and safetensors_load_strategy not in allowed_loaders:
        conflicts.append(Conflict(
            "safetensors_load_strategy", allowed_loaders, safetensors_load_strategy,
            "weight loading strategy is not declared compatible with the selected model",
            "manually select an allowed CPP_SAFETENSORS_LOAD_STRATEGY",
        ))
    return conflicts


def case_conflicts(
    model: dict[str, Any],
    *,
    execution_mode: str,
    prefix_cache_enabled: bool,
    enable_expert_parallel: bool,
    api_mode: str,
    prompt_mode: str,
    tokenizer_mode: str,
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    if api_mode != "completions":
        conflicts.append(Conflict(
            "api_mode", "completions", api_mode,
            "the current performance client implements only the completions API",
            "manually select completions or add and validate an API adapter before retrying",
        ))
    if prompt_mode != "raw":
        conflicts.append(Conflict(
            "prompt_mode", "raw", prompt_mode,
            "the current performance dataset implements only raw prompts",
            "manually select raw prompts or add and validate a prompt adapter before retrying",
        ))
    if execution_mode not in model["supported_execution_modes"]:
        conflicts.append(Conflict(
            "execution_mode", model["supported_execution_modes"], execution_mode,
            "execution mode is not declared compatible with the selected model",
            "manually select a supported matrix row or validate and update the model capability",
        ))
    if prefix_cache_enabled and not model["supports_prefix_cache"]:
        conflicts.append(Conflict(
            "prefix_cache", False, True,
            "prefix cache is not declared compatible with the selected model",
            "manually disable CPP_PREFIX_CACHE_ENABLED or select a compatible model",
        ))
    if enable_expert_parallel and model["model_kind"] == "dense":
        conflicts.append(Conflict(
            "expert_parallel", "dense", True,
            "expert parallel cannot be enabled for a model declared dense",
            "manually disable CPP_ENABLE_EXPERT_PARALLEL or correct the model kind",
        ))
    if api_mode == "completions" and api_mode not in model["supported_api_modes"]:
        conflicts.append(Conflict(
            "api_mode", model["supported_api_modes"], api_mode,
            "API mode is not declared compatible with the selected model",
            "manually select a supported test API or update the model capability after validation",
        ))
    if prompt_mode == "raw" and prompt_mode not in model["supported_prompt_modes"]:
        conflicts.append(Conflict(
            "prompt_mode", model["supported_prompt_modes"], prompt_mode,
            "prompt mode is not declared compatible with the selected model",
            "manually select a supported prompt mode or add a tested prompt adapter",
        ))
    if tokenizer_mode not in model["supported_tokenizer_modes"]:
        conflicts.append(Conflict(
            "tokenizer_mode", model["supported_tokenizer_modes"], tokenizer_mode,
            "tokenizer mode is not declared compatible with the selected model",
            "manually correct CPP_TOKENIZER_MODE or the model tokenizer profile",
        ))
    return conflicts


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_conflicts(
    conflicts: list[Conflict],
    *,
    artifact_root: Path,
    model: dict[str, Any],
    model_config: Path,
    suite_config: Path,
    effective: dict[str, Any],
    project_root: Path,
) -> list[Path]:
    conflict_root = artifact_root / "conflicts"
    conflict_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outputs = []
    for conflict in conflicts:
        key = slugify(conflict.key)
        output = conflict_root / f"{model['model_id']}__{key}__{timestamp}.log"
        payload = {
            "status": "blocked",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_id": model["model_id"],
            "model_config": str(model_config.resolve()),
            "test_config": str(suite_config.resolve()),
            "conflict": asdict(conflict),
            "effective_config": effective,
            "environment_overrides": {
                key: value for key, value in os.environ.items() if key.startswith("CPP_")
            },
            "project_revision": git_revision(project_root),
            "vllm_revision": git_revision(project_root / "deps" / "vllm"),
            "vllm_ascend_revision": git_revision(project_root / "deps" / "vllm-ascend"),
        }
        output.write_text(
            "CPP_MODEL_CONFIG_CONFLICT\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        outputs.append(output)
        print(f"CPP_MODEL_CONFIG_CONFLICT key={conflict.key} file={output}", file=sys.stderr)
    return outputs


def parse_bool(value: str) -> bool:
    rendered = value.strip().lower()
    if rendered in {"1", "true", "yes"}:
        return True
    if rendered in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def parse_quantization(value: str) -> str | None:
    return None if value.strip().lower() in {"", "none", "null"} else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("global", "case"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-mode", required=True)
    parser.add_argument("--tokenizer-trust-remote-code", type=parse_bool, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--npu-devices", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--safetensors-load-strategy", required=True)
    parser.add_argument("--dataset", choices=("fixed", "variable"))
    parser.add_argument("--execution-mode", choices=("eager", "graph"))
    parser.add_argument("--prefix-cache-enabled", type=parse_bool)
    parser.add_argument("--enable-expert-parallel", type=parse_bool)
    parser.add_argument("--api-mode")
    parser.add_argument("--prompt-mode")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        model = resolve_model_profile(load_json(args.model_config))
        suite = resolve_suite_contract(load_json(args.suite_config))
    except ValueError as error:
        model_id = slugify(args.model_config.stem)
        model = {"model_id": model_id}
        conflicts = [Conflict(
            "configuration", str(args.model_config), str(args.suite_config),
            str(error), "manually repair the configuration files before retrying",
        )]
    else:
        common = dict(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            tokenizer_trust_remote_code=args.tokenizer_trust_remote_code,
            max_model_len=args.max_model_len,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            npu_devices=[int(value) for value in args.npu_devices.split(",")],
            quantization=parse_quantization(args.quantization),
            safetensors_load_strategy=args.safetensors_load_strategy,
        )
        if args.scope == "global":
            conflicts = global_conflicts(model, suite, dataset=args.dataset, **common)
        else:
            required = {
                "execution_mode": args.execution_mode,
                "prefix_cache_enabled": args.prefix_cache_enabled,
                "enable_expert_parallel": args.enable_expert_parallel,
                "api_mode": args.api_mode,
                "prompt_mode": args.prompt_mode,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise SystemExit(f"case preflight missing arguments: {', '.join(missing)}")
            conflicts = case_conflicts(
                model,
                execution_mode=args.execution_mode,
                prefix_cache_enabled=args.prefix_cache_enabled,
                enable_expert_parallel=args.enable_expert_parallel,
                api_mode=args.api_mode,
                prompt_mode=args.prompt_mode,
                tokenizer_mode=args.tokenizer_mode,
            )
        effective = {**common, "tokenizer_mode": args.tokenizer_mode, "scope": args.scope}
    if conflicts:
        effective = locals().get("effective", {"scope": args.scope})
        write_conflicts(
            conflicts,
            artifact_root=args.artifact_root,
            model=model,
            model_config=args.model_config,
            suite_config=args.suite_config,
            effective=effective,
            project_root=args.project_root,
        )
        return CONFIG_ERROR_EXIT_CODE
    print(f"CPP_MODEL_CONFIG_PREFLIGHT_PASS scope={args.scope} model={model['model_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
