#!/usr/bin/env bash

cpp_new_run_id() {
    local suite="$1" revision short_revision
    revision="$(cpp_git_revision "${PROJECT_ROOT}")"
    short_revision="${revision:0:7}"
    [[ "${short_revision}" != "unknown" ]] || short_revision="nogit"
    printf '%s_%s_%s_%s\n' "$(date +%Y%m%dT%H%M%S%z)" "${short_revision}" "${suite}" "$$"
}

cpp_initialize_run() {
    local suite="$1"
    mkdir -p "${CPP_ARTIFACT_ROOT}/runs"
    CPP_RUN_ID="${CPP_RUN_ID:-$(cpp_new_run_id "${suite}")}"
    CPP_RUN_DIR="${CPP_RUN_DIR:-${CPP_ARTIFACT_ROOT}/runs/${CPP_RUN_ID}}"
    export CPP_RUN_ID CPP_RUN_DIR
    [[ ! -e "${CPP_RUN_DIR}" ]] || cpp_fail "run directory already exists: ${CPP_RUN_DIR}"
    mkdir -p "${CPP_RUN_DIR}/cases" "${CPP_RUN_DIR}/reports"

    python3 - "${CPP_RUN_DIR}/run.json" "${CPP_RUN_ID}" "${suite}" \
        "$(cpp_utc_timestamp)" "$(hostname)" "${MODEL_PATH}" "${MODEL_NAME}" \
        "$(cpp_git_revision "${PROJECT_ROOT}")" \
        "$(cpp_git_revision "${PROJECT_ROOT}/deps/vllm")" \
        "$(cpp_git_revision "${PROJECT_ROOT}/deps/vllm-ascend")" \
        "${NPU_DEVICES}" <<'PY'
import json
import platform
import sys
from pathlib import Path

(
    output, run_id, suite, started_at, host, model_path, model_name,
    project_revision, vllm_revision, vllm_ascend_revision, devices,
) = sys.argv[1:]
data = {
    "schema_version": 1,
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
Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
    printf '%s\n' "${CPP_RUN_ID}" >"${CPP_ARTIFACT_ROOT}/LATEST"
}

cpp_initialize_case() {
    local case_dir="$1" case_id="$2" runner="$3" dynamic="$4"
    local execution_mode="$5" request_mode="$6" token_targets="$7"
    local request_repeats="$8" expected_request_count="$9" max_fit_chunk="${10}"
    [[ ! -e "${case_dir}" ]] || cpp_fail "case directory already exists: ${case_dir}"
    mkdir -p "${case_dir}/logs" "${case_dir}/raw" "${case_dir}/compiler" "${case_dir}/results"
    python3 - "${case_dir}/case.json" "${case_id}" "${runner}" "${dynamic}" \
        "${execution_mode}" "${request_mode}" "${token_targets}" "${SERVER_PORT}" \
        "${NPU_DEVICES}" "${HCCL_PORT_RANGE}" "${MAX_OUTPUT_TOKENS}" \
        "${PIPELINE_PARALLEL_SIZE}" "${MAX_MODEL_LEN}" \
        "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" \
        "${request_repeats}" "${expected_request_count}" "${max_fit_chunk}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, case_id, runner, dynamic, execution_mode, request_mode,
    targets, port, devices, hccl_range, max_output_tokens, pipeline_parallel_size,
    max_model_len, max_num_batched_tokens, max_num_seqs, request_repeats,
    expected_request_count, max_fit_chunk,
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
