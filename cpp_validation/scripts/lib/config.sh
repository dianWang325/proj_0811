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

    mapfile -t CPP_PERF_MODEL_DEFAULTS < <(python3 - "${model_config}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "model_path", "fallback_model_path", "served_model_name",
    "max_model_len", "pipeline_parallel_size", "tensor_parallel_size",
):
    print(data[key])
PY
    )
    [[ "${#CPP_PERF_MODEL_DEFAULTS[@]}" -eq 6 ]] || \
        cpp_fail "invalid performance model config: ${model_config}"

    mapfile -t CPP_PERF_SUITE_DEFAULTS < <(python3 - "${suite_config}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
fixed = data["fixed"]
variable = data["variable"]
cpp = data.get("cpp", {})
calibration = cpp.get("online_calibration", {})
fixed_prefix = fixed.get("prefix_cache", {})
variable_prefix = variable.get("prefix_cache", {})
print(data["warmup_count"])
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
manual = data.get("manual_warmup", {})
print(1 if manual.get("enabled", True) else 0)
print(manual.get("input_tokens", 131072))
print(manual.get("output_tokens", 1))
print(manual.get("request_count", 5))
print(manual.get("concurrency", 1))
print(manual.get("request_rate", 0))
print(cpp.get("smooth_factor", 1.0))
print(1 if calibration.get("same_distribution_first", False) else 0)
print(calibration.get("post_manual_rewarm_count", 0))
print(1 if fixed_prefix.get("enabled", False) else 0)
print(1 if variable_prefix.get("enabled", False) else 0)
print(variable_prefix.get("repeat_rate", ""))
print(1 if variable_prefix.get("prefix_test", False) else 0)
PY
    )
    [[ "${#CPP_PERF_SUITE_DEFAULTS[@]}" -eq 29 ]] || \
        cpp_fail "invalid performance suite config: ${suite_config}"
}
