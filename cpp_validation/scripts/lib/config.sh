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
print(data["request_count"])
print(data["warmup_count"])
print(data["max_output_tokens"])
print(data["max_num_batched_tokens"])
print(data["fixed"]["input_tokens"])
print(data["fixed"]["concurrency"])
print(data["variable"]["min_input_tokens"])
print(data["variable"]["max_input_tokens"])
print(data["variable"]["mean_input_tokens"])
print(data["variable"]["concurrency"])
print(data["variable"]["seed"])
print(data.get("data_generator", "aisbench"))
manual = data.get("manual_warmup", {})
print(1 if manual.get("enabled", True) else 0)
print(manual.get("input_tokens", 131072))
print(manual.get("output_tokens", 1))
print(manual.get("request_count", 5))
print(manual.get("concurrency", 1))
print(manual.get("request_rate", 0))
print(data["fixed"].get("request_rate", 0))
print(data["variable"].get("request_rate", 0))
PY
    )
    [[ "${#CPP_PERF_SUITE_DEFAULTS[@]}" -eq 20 ]] || \
        cpp_fail "invalid performance suite config: ${suite_config}"
}
