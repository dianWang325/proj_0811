#!/usr/bin/env bash

cpp_new_run_id() {
    local suite="$1" revision short_revision
    revision="$(cpp_git_revision "${PROJECT_ROOT}")"
    short_revision="${revision:0:7}"
    [[ "${short_revision}" != "unknown" ]] || short_revision="nogit"
    if [[ "${suite}" == performance* ]]; then
        local suite_id="perf_case"
        [[ "${suite}" == "performance-matrix" ]] && suite_id="perf_matrix"
        printf '%s_%s_%s_%s\n' "${suite_id}" "$(date +%Y%m%dT%H%M%S%z)" \
            "${short_revision}" "$$"
        return
    fi
    printf '%s_%s_%s_%s\n' "$(date +%Y%m%dT%H%M%S%z)" "${short_revision}" "${suite}" "$$"
}

cpp_run_id_with_model() {
    local base_run_id="$1" model_id="$2" marker prefix suffix
    [[ -n "${model_id}" ]] || cpp_fail "performance MODEL_ID must not be empty"
    if [[ "${base_run_id}" == *"${model_id}"* ]]; then
        printf '%s\n' "${base_run_id}"
        return
    fi
    for marker in perf_matrix performance_matrix performance-matrix performance; do
        if [[ "${base_run_id}" == *"${marker}"* ]]; then
            prefix="${base_run_id%%${marker}*}"
            suffix="${base_run_id#*${marker}}"
            printf '%s%s_%s%s\n' "${prefix}" "${marker}" "${model_id}" "${suffix}"
            return
        fi
    done
    printf '%s_%s\n' "${base_run_id}" "${model_id}"
}

cpp_initialize_run() {
    local suite="$1" requested_run_dir="${CPP_RUN_DIR:-}" run_schema_version=1
    mkdir -p "${CPP_ARTIFACT_ROOT}/runs"
    CPP_RUN_BASE_ID="${CPP_RUN_ID:-$(cpp_new_run_id "${suite}")}"
    CPP_RUN_ID="${CPP_RUN_BASE_ID}"
    if [[ "${suite}" == performance* ]]; then
        run_schema_version=2
        CPP_RUN_ID="$(cpp_run_id_with_model "${CPP_RUN_BASE_ID}" "${MODEL_ID:-}")"
    fi
    if [[ -n "${requested_run_dir}" ]]; then
        CPP_RUN_DIR="$(dirname "${requested_run_dir}")/${CPP_RUN_ID}"
    else
        CPP_RUN_DIR="${CPP_ARTIFACT_ROOT}/runs/${CPP_RUN_ID}"
    fi
    export CPP_RUN_BASE_ID CPP_RUN_ID CPP_RUN_DIR
    [[ ! -e "${CPP_RUN_DIR}" ]] || cpp_fail "run directory already exists: ${CPP_RUN_DIR}"
    mkdir -p "${CPP_RUN_DIR}/cases" "${CPP_RUN_DIR}/datasets" "${CPP_RUN_DIR}/reports"

    python3 - "${CPP_RUN_DIR}/run.json" "${CPP_RUN_ID}" "${CPP_RUN_BASE_ID}" \
        "${suite}" "${run_schema_version}" \
        "$(cpp_utc_timestamp)" "$(hostname 2>/dev/null || uname -n)" \
        "${MODEL_PATH}" "${MODEL_NAME}" \
        "$(cpp_git_revision "${PROJECT_ROOT}")" \
        "$(cpp_git_revision "${PROJECT_ROOT}/deps/vllm")" \
        "$(cpp_git_revision "${PROJECT_ROOT}/deps/vllm-ascend")" \
        "${NPU_DEVICES}" "${MODEL_ID:-}" "${MODEL_FAMILY:-}" \
        "${TOKENIZER_PATH:-}" "${TOKENIZER_MODE:-}" \
        "${TOKENIZER_TRUST_REMOTE_CODE:-}" "${QUANTIZATION:-}" \
        "${GPU_MEMORY_UTILIZATION:-}" "${KV_CACHE_MEMORY:-}" \
        "${ENABLE_EXPERT_PARALLEL:-}" "${SAFETENSORS_LOAD_STRATEGY:-}" <<'PY'
import json
import platform
import sys
from pathlib import Path

(
    output, run_id, base_run_id, suite, schema_version, started_at, host,
    model_path, model_name,
    project_revision, vllm_revision, vllm_ascend_revision, devices,
    model_id, model_family, tokenizer_path, tokenizer_mode,
    tokenizer_trust_remote_code, quantization, gpu_memory_utilization,
    kv_cache_memory, expert_parallel, safetensors_load_strategy,
) = sys.argv[1:]
data = {
    "schema_version": int(schema_version),
    "run_id": run_id,
    "suite": suite,
    "started_at": started_at,
    "host": host,
    "model_path": model_path,
    "model_name": model_name,
    "project_revision": project_revision,
    "vllm_revision": vllm_revision,
    "vllm_ascend_revision": vllm_ascend_revision,
    "python_version": platform.python_version(),
    "npu_devices": [int(value) for value in devices.split(",")],
}
if int(schema_version) >= 2:
    data["base_run_id"] = base_run_id
if model_id:
    data.update({
        "model_id": model_id,
        "model_family": model_family,
        "tokenizer": {
            "path": tokenizer_path,
            "mode": tokenizer_mode,
            "trust_remote_code": tokenizer_trust_remote_code == "1",
        },
        "quantization": quantization or None,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "kv_cache_memory": kv_cache_memory or None,
        "expert_parallel": expert_parallel == "1",
        "safetensors_load_strategy": safetensors_load_strategy,
    })
Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
    if [[ "${suite}" == performance* ]]; then
        python3 - "${MODEL_CONFIG}" "${SUITE_CONFIG}" \
            "${CPP_RUN_DIR}/effective_model_config.json" \
            "${CPP_RUN_DIR}/effective_test_config.json" \
            "${MODEL_ID}" "${MODEL_FAMILY}" "${MODEL_PATH}" "${MODEL_NAME}" \
            "${TOKENIZER_PATH}" "${TOKENIZER_MODE}" "${TOKENIZER_TRUST_REMOTE_CODE}" \
            "${QUANTIZATION}" "${SAFETENSORS_LOAD_STRATEGY}" \
            "${MAX_MODEL_LEN}" "${PIPELINE_PARALLEL_SIZE}" \
            "${TENSOR_PARALLEL_SIZE}" "${GPU_MEMORY_UTILIZATION}" \
            "${KV_CACHE_MEMORY}" "${ENABLE_EXPERT_PARALLEL}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    model_source, suite_source, model_output, test_output, model_id,
    model_family, model_path, model_name, tokenizer_path, tokenizer_mode,
    tokenizer_trust, quantization, load_strategy, max_model_len, pp_size,
    tp_size, gpu_memory, kv_cache, expert_parallel,
) = sys.argv[1:]
model_raw = json.loads(Path(model_source).read_text(encoding="utf-8"))
suite_raw = json.loads(Path(suite_source).read_text(encoding="utf-8"))
overrides = {key: value for key, value in os.environ.items() if key.startswith("CPP_")}
model_data = {
    "schema_version": 2,
    "source": str(Path(model_source).resolve()),
    "source_config": model_raw,
    "effective": {
        "model_id": model_id,
        "model_family": model_family,
        "model_path": model_path,
        "served_model_name": model_name,
        "tokenizer": {
            "path": tokenizer_path,
            "mode": tokenizer_mode,
            "trust_remote_code": tokenizer_trust == "1",
        },
        "quantization": quantization or None,
        "safetensors_load_strategy": load_strategy,
    },
    "environment_overrides": overrides,
}
test_data = {
    "schema_version": 2,
    "source": str(Path(suite_source).resolve()),
    "source_config": suite_raw,
    "effective": {
        "max_model_len": int(max_model_len),
        "pipeline_parallel_size": int(pp_size),
        "tensor_parallel_size": int(tp_size),
        "gpu_memory_utilization": float(gpu_memory),
        "kv_cache_memory": kv_cache or None,
        "expert_parallel": expert_parallel == "1",
    },
    "environment_overrides": overrides,
}
Path(model_output).write_text(json.dumps(model_data, indent=2) + "\n", encoding="utf-8")
Path(test_output).write_text(json.dumps(test_data, indent=2) + "\n", encoding="utf-8")
PY
    fi
    printf '%s\n' "${CPP_RUN_ID}" >"${CPP_ARTIFACT_ROOT}/LATEST"
    echo "CPP_RUN_ID_RESOLVED base=${CPP_RUN_BASE_ID} effective=${CPP_RUN_ID}"
}

cpp_initialize_case() {
    local case_dir="$1" case_id="$2" runner="$3" dynamic="$4"
    local execution_mode="$5" request_mode="$6" token_targets="$7"
    local request_repeats="$8" expected_request_count="$9" max_fit_chunk="${10}"
    local data_generator="${11}"
    [[ ! -e "${case_dir}" ]] || cpp_fail "case directory already exists: ${case_dir}"
    mkdir -p "${case_dir}/logs" "${case_dir}/raw" "${case_dir}/compiler" "${case_dir}/results"
    python3 - "${case_dir}/case.json" "${case_id}" "${runner}" "${dynamic}" \
        "${execution_mode}" "${request_mode}" "${token_targets}" "${SERVER_PORT}" \
        "${NPU_DEVICES}" "${HCCL_PORT_RANGE}" "${MAX_OUTPUT_TOKENS}" \
        "${PIPELINE_PARALLEL_SIZE}" "${MAX_MODEL_LEN}" \
        "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" \
        "${request_repeats}" "${expected_request_count}" "${max_fit_chunk}" \
        "${data_generator}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, case_id, runner, dynamic, execution_mode, request_mode,
    targets, port, devices, hccl_range, max_output_tokens, pipeline_parallel_size,
    max_model_len, max_num_batched_tokens, max_num_seqs, request_repeats,
    expected_request_count, max_fit_chunk, data_generator,
) = sys.argv[1:]
data = {
    "schema_version": 1,
    "case_id": case_id,
    "suite": "functional",
    "runner": runner,
    "cpp_mode": "dynamic" if dynamic == "1" else "static",
    "dynamic": dynamic == "1",
    "execution_mode": execution_mode,
    "request_mode": request_mode,
    "token_targets": [int(value) for value in targets.split(",")],
    "max_output_tokens": int(max_output_tokens),
    "server_port": int(port),
    "npu_devices": [int(value) for value in devices.split(",")],
    "hccl_port_range": hccl_range,
    "pipeline_parallel_size": int(pipeline_parallel_size),
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "max_num_seqs": int(max_num_seqs),
    "request_repeats": int(request_repeats),
    "expected_request_count": int(expected_request_count),
    "max_fit_chunk": int(max_fit_chunk),
    "data_generator": data_generator,
}
Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

cpp_initialize_performance_case() {
    local case_dir="$1" case_id="$2"
    [[ ! -e "${case_dir}" ]] || cpp_fail "case directory already exists: ${case_dir}"
    mkdir -p \
        "${case_dir}/logs" \
        "${case_dir}/raw" \
        "${case_dir}/compiler" \
        "${case_dir}/results"
    python3 - "${case_dir}/case.json" "${case_id}" "${RUNNER}" \
        "${DYNAMIC}" "${EXECUTION_MODE}" "${PERF_DATASET}" \
        "${SERVER_PORT}" "${NPU_DEVICES}" "${HCCL_PORT_RANGE}" \
        "${PIPELINE_PARALLEL_SIZE}" "${TENSOR_PARALLEL_SIZE}" \
        "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" \
        "${REQUEST_COUNT}" "${WARMUP_COUNT}" "${CONCURRENCY}" \
        "${REQUEST_RATE}" "${MAX_OUTPUT_TOKENS}" "${DATA_GENERATOR}" \
        "${MANUAL_WARMUP_ENABLED}" "${MANUAL_WARMUP_ENABLED_CONFIG}" \
        "${MANUAL_WARMUP_DATASET_MODE}" "${MANUAL_WARMUP_SEED}" \
        "${PREFIX_CACHE_ENABLED}" "${PREFIX_REPEAT_RATE}" "${PREFIX_TEST}" \
        "${CPP_SMOOTH_FACTOR}" "${NEED_TIMING}" "${ASYNC_SCHEDULING}" \
        "${MATRIX_ROUND}" "${MATRIX_POSITION}" "${MODEL_ID}" \
        "${MODEL_FAMILY}" "${TOKENIZER_PATH}" "${TOKENIZER_MODE}" \
        "${TOKENIZER_TRUST_REMOTE_CODE}" "${QUANTIZATION}" \
        "${GPU_MEMORY_UTILIZATION}" "${KV_CACHE_MEMORY}" \
        "${ENABLE_EXPERT_PARALLEL}" "${SAFETENSORS_LOAD_STRATEGY}" \
        "${API_MODE}" "${PROMPT_MODE}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, case_id, runner, dynamic, execution_mode, dataset, port,
    devices, hccl_range, pp_size, tp_size, max_model_len,
    max_num_batched_tokens, request_count, warmup_count, concurrency,
    request_rate, max_output_tokens, data_generator, manual_warmup_enabled,
    manual_warmup_enabled_config, manual_warmup_dataset_mode,
    manual_warmup_seed, prefix_cache_enabled, prefix_repeat_rate,
    prefix_test, smooth_factor, need_timing, async_scheduling,
    matrix_round, matrix_position,
    model_id, model_family, tokenizer_path, tokenizer_mode,
    tokenizer_trust_remote_code, quantization, gpu_memory_utilization,
    kv_cache_memory, expert_parallel, safetensors_load_strategy,
    api_mode, prompt_mode,
) = sys.argv[1:]
data = {
    "schema_version": 2,
    "case_id": case_id,
    "suite": "performance",
    "runner": runner,
    "cpp_mode": "dynamic" if dynamic == "1" else "static",
    "dynamic": dynamic == "1",
    "execution_mode": execution_mode,
    "async_scheduling": async_scheduling == "1",
    "configured_cudagraph_mode": (
        "FULL_DECODE_ONLY" if execution_mode == "graph" else "NONE"
    ),
    "dataset": dataset,
    "server_port": int(port),
    "npu_devices": [int(value) for value in devices.split(",")],
    "hccl_port_range": hccl_range,
    "pipeline_parallel_size": int(pp_size),
    "tensor_parallel_size": int(tp_size),
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "request_count": int(request_count),
    "warmup_count": int(warmup_count),
    "concurrency": int(concurrency),
    "request_rate": float(request_rate),
    "max_output_tokens": int(max_output_tokens),
    "data_generator": data_generator,
    "model_id": model_id,
    "model_family": model_family,
    "tokenizer": {
        "path": tokenizer_path,
        "mode": tokenizer_mode,
        "trust_remote_code": tokenizer_trust_remote_code == "1",
    },
    "quantization": quantization or None,
    "gpu_memory_utilization": float(gpu_memory_utilization),
    "kv_cache_memory": kv_cache_memory or None,
    "expert_parallel": expert_parallel == "1",
    "safetensors_load_strategy": safetensors_load_strategy,
    "api_mode": api_mode,
    "prompt_mode": prompt_mode,
    "manual_warmup": {
        "enabled": manual_warmup_enabled == "1",
        "configured_enabled": manual_warmup_enabled_config,
        "dataset_mode": manual_warmup_dataset_mode,
        "dataset": dataset,
        "seed": int(manual_warmup_seed),
        "request_count": int(warmup_count),
        "concurrency": int(concurrency),
        "request_rate": float(request_rate),
        "output_tokens": int(max_output_tokens),
    },
    "prefix_cache": {
        "enabled": prefix_cache_enabled == "1",
        "repeat_rate": prefix_repeat_rate or None,
        "prefix_test": prefix_test == "1",
    },
    "cpp_tuning": {
        "smooth_factor": float(smooth_factor),
        "need_timing": need_timing == "true",
    },
}
if matrix_round:
    data["matrix"] = {
        "round": int(matrix_round),
        "position": int(matrix_position),
    }
Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

cpp_write_case_status() {
    local output="$1" state="$2" exit_code="$3" stage="$4" started_at="$5"
    python3 - "${output}" "${state}" "${exit_code}" "${stage}" "${started_at}" \
        "$(cpp_utc_timestamp)" <<'PY'
import json
import sys
from pathlib import Path

output, state, exit_code, stage, started_at, finished_at = sys.argv[1:]
data = {
    "schema_version": 1,
    "state": state,
    "exit_code": int(exit_code),
    "last_stage": stage,
    "started_at": started_at,
    "finished_at": finished_at,
}
Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}
