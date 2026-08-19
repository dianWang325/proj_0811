#!/usr/bin/env bash

cpp_load_config_defaults() {
    local model_config="$1" suite_config="$2"
    [[ -r "${model_config}" ]] || cpp_fail "model config is not readable: ${model_config}"
    [[ -r "${suite_config}" ]] || cpp_fail "suite config is not readable: ${suite_config}"

    mapfile -t CPP_MODEL_DEFAULTS < <(python3 - "${model_config}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("model_path", "served_model_name", "max_model_len", "pipeline_parallel_size"):
    print(data[key])
PY
    )
    [[ "${#CPP_MODEL_DEFAULTS[@]}" -eq 4 ]] || cpp_fail "invalid model config: ${model_config}"

    mapfile -t CPP_SUITE_DEFAULTS < <(python3 - "${suite_config}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["request_mode"])
print(",".join(str(value) for value in data["token_targets"]))
for key in ("max_output_tokens", "max_num_batched_tokens", "max_num_seqs"):
    print(data[key])
print(data.get("data_generator", "aisbench"))
PY
    )
    [[ "${#CPP_SUITE_DEFAULTS[@]}" -eq 6 ]] || cpp_fail "invalid suite config: ${suite_config}"
}

cpp_load_performance_config_defaults() {
    local model_config="$1" suite_config="$2"
    [[ -r "${model_config}" ]] || cpp_fail "model config is not readable: ${model_config}"
    [[ -r "${suite_config}" ]] || cpp_fail "suite config is not readable: ${suite_config}"

    mapfile -t CPP_PERF_MODEL_DEFAULTS < <(python3 - "${model_config}" "${suite_config}" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))
suite = json.load(open(sys.argv[2], encoding="utf-8"))
serving = suite.get("serving", {})
schema_version = int(model.get("schema_version", 1))
if schema_version == 1:
    print(
        f"CPP_CONFIG_DEPRECATED legacy performance model config: {sys.argv[1]}",
        file=sys.stderr,
    )
model_path = model["model_path"]
model_id = model.get("model_id", model.get("served_model_name"))
tokenizer = model.get("tokenizer", {})
weight_loading = model.get("weight_loading", {})
values = (
    model_path,
    model.get("fallback_model_path", ""),
    model["served_model_name"],
    serving.get("max_model_len", model.get("max_model_len", 132000)),
    serving.get("pipeline_parallel_size", model.get("pipeline_parallel_size", 2)),
    serving.get("tensor_parallel_size", model.get("tensor_parallel_size", 1)),
    model_id,
    model.get("model_family", "unknown"),
    tokenizer.get("path") or "",
    tokenizer.get("mode", "deepseek_v4" if schema_version == 1 else "auto"),
    1 if tokenizer.get("trust_remote_code", True) else 0,
    "" if model.get("quantization", "ascend") is None else model.get("quantization", "ascend"),
    weight_loading.get("safetensors_load_strategy", "auto"),
)
for value in values:
    print(value)
PY
    )
    [[ "${#CPP_PERF_MODEL_DEFAULTS[@]}" -eq 13 ]] || \
        cpp_fail "invalid performance model config: ${model_config}"

    mapfile -t CPP_PERF_SUITE_DEFAULTS < <(python3 - "${suite_config}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
fixed = data["fixed"]
variable = data["variable"]
cpp = data.get("cpp", {})
manual = data.get("manual_warmup", {})
fixed_prefix = fixed.get("prefix_cache", {})
variable_prefix = variable.get("prefix_cache", {})
print(manual.get("request_count", 30))
print(data["max_output_tokens"])
print(fixed["input_tokens"])
print(fixed["request_count"])
print(fixed["concurrency"])
print(fixed.get("request_rate", 0))
print(fixed["max_num_batched_tokens"])
print(variable["min_input_tokens"])
print(variable["max_input_tokens"])
print(variable["mean_input_tokens"])
print(variable["request_count"])
print(variable["concurrency"])
print(variable.get("request_rate", 0))
print(variable["max_num_batched_tokens"])
print(variable["seed"])
print(data.get("data_generator", "aisbench"))
enabled = manual.get("enabled", "auto")
print(enabled.lower() if isinstance(enabled, str) else (1 if enabled else 0))
print(manual.get("dataset_mode", "generated"))
print(manual.get("seed_offset", 1))
print(cpp.get("smooth_factor", 1.0))
print(1 if fixed_prefix.get("enabled", False) else 0)
print(1 if variable_prefix.get("enabled", False) else 0)
print(variable_prefix.get("repeat_rate", ""))
print(1 if variable_prefix.get("prefix_test", False) else 0)
serving = data.get("serving", {})
print(serving.get("max_model_len", 132000))
print(serving.get("pipeline_parallel_size", 2))
print(serving.get("tensor_parallel_size", 4))
print(serving.get("api_mode", "completions"))
print(serving.get("prompt_mode", "raw"))
PY
    )
    [[ "${#CPP_PERF_SUITE_DEFAULTS[@]}" -eq 29 ]] || \
        cpp_fail "invalid performance suite config: ${suite_config}"
}
