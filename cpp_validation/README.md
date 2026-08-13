# CPP validation

This directory contains source code and configuration for CPP-specific
validation. Runtime artifacts are deliberately kept outside the source tree in
`../artifacts/cpp`.

## Test layers

- `workloads/functional`: request generation for functional validation.
- `workloads/performance`: reserved for repeatable latency and throughput runs.
- `workloads/accuracy`: reserved for dataset-driven accuracy generation.
- `validators`: suite-specific pass/fail checks and structured result creation.
- `analysis`: run-level reporting and future cross-run comparison tools.
- `scripts/lib`: reusable server lifecycle and artifact-management stages.

The shared lifecycle is: create run, create case, start server, wait for health,
run workload, collect compiler artifacts, validate, write status, and stop the
server. Suite-specific request and analysis logic should not be added to the
shared server lifecycle.

## Running tests

Run one MRv2 dynamic functional case:

```bash
CPP_NPU_DEVICES=8,9 ./proj_0811/cpp_validation/bin/cpp-test quick
```

Run the functional compatibility matrix:

```bash
CPP_NPU_DEVICES=8,9 ./proj_0811/cpp_validation/bin/cpp-test matrix
```

Run one explicitly configured case:

```bash
CPP_RUNNER=mrv1 \
CPP_DYNAMIC=1 \
CPP_EXECUTION_MODE=graph \
CPP_REQUEST_MODE=both \
CPP_NPU_DEVICES=8,9 \
./proj_0811/cpp_validation/bin/cpp-test case
```

Useful overrides include `CPP_ARTIFACT_ROOT`, `CPP_MODEL_PATH`,
`CPP_MODEL_NAME`, `CPP_PORT`, `CPP_TOKEN_TARGETS`, `CPP_REQUEST_TIMEOUT`,
`CPP_REQUEST_REPEATS`, `CPP_MAX_FIT_CHUNK`, `CPP_STARTUP_TIMEOUT`, and
`CPP_HCCL_PORT_RANGE`. For dynamic CPP cases, the runner rejects a workload
unless its expected request count is strictly greater than `CPP_MAX_FIT_CHUNK`.
The expected count is token targets × request repeats × request phases (two
phases when `CPP_REQUEST_MODE=both`).

The default model and functional workload values come from `configs/models`
and `configs/suites`; the matrix rows come from `configs/matrices`. Override
the selected files with `CPP_MODEL_CONFIG`, `CPP_SUITE_CONFIG`, and
`CPP_MATRIX_FILE`. Explicit `CPP_*` values take precedence over config files.

## Artifact contract

Each invocation creates an immutable run directory:

```text
artifacts/cpp/runs/<run-id>/
├── run.json
├── cases/<case-id>/
│   ├── case.json
│   ├── status.json
│   ├── logs/server.log
│   ├── logs/client.log
│   ├── raw/requests.json
│   ├── raw/cpp_trace.jsonl
│   ├── compiler/fusion_result.json
│   └── results/functional_result.json
└── reports/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

`logs` and `raw` are source evidence and should not be edited after a run.
`results` contains machine-readable validation output. `reports` contains
run-level summaries and may gain versioned re-analysis subdirectories later.
`artifacts/cpp/LATEST` contains the most recently created run ID for
convenience; it is not a durable identifier.

Historical outputs migrated from `smoke_test` are under
`artifacts/cpp/legacy`. Files whose original case cannot be proven remain under
`legacy/root-level` instead of being assigned speculatively.
