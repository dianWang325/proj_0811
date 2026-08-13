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
PY
    )
    [[ "${#CPP_SUITE_DEFAULTS[@]}" -eq 5 ]] || cpp_fail "invalid suite config: ${suite_config}"
}
