# CPP validation

This directory contains source code and configuration for CPP-specific
validation. Runtime artifacts are deliberately kept outside the source tree in
`../artifacts/cpp`.

## Test layers

- `workloads/data_generation.py`: shared `aisbench`/`script` dataset backend.
- `workloads/aisbench_auto_tools.py`: isolated upstream-tool manual warmup.
- `workloads/functional`: functional request submission using either dataset path.
- `workloads/performance`: deterministic AISBench latency and throughput runs.
- `workloads/accuracy`: reserved for dataset-driven accuracy generation.
- `validators`: suite-specific pass/fail checks and structured result creation.
- `analysis`: run-level reporting and future cross-run comparison tools.
- `scripts/lib`: reusable server lifecycle and artifact-management stages.

The shared lifecycle is: create run, create case, generate and validate the
dataset, start the server, wait for health, run untimed preparation, run the
measured workload, validate, write status, and stop the server. Dataset JSONL
and generation provenance are immutable case artifacts, so warmup and measured
requests consume the same verified prompts.

## Dataset backends and AISBench auto tools

Both functional and performance suites accept `CPP_DATA_GENERATOR=aisbench` or
`CPP_DATA_GENERATOR=script`. The default is `aisbench`. This backend calls the
generator from `rayn-zzz/aisbench_auto_tools_prefix`, then checks every prompt
with the selected model tokenizer before writing `raw/dataset.jsonl` and
`raw/dataset_metadata.json`. The `script` backend retains the original
exact-token prompt generator and is useful for comparison or fallback.

Install and verify the pinned upstream revision in the container with:

```bash
./proj_0811/cpp_validation/scripts/install_aisbench_auto_tools.sh
```

The default installation is
`/home/w00985415/tools/aisbench_auto_tools_prefix`; override it with
`CPP_AISBENCH_AUTO_TOOLS_ROOT`. Each manual warmup receives a private copy of
the tool configuration and output directory, avoiding cross-case `config.py`,
dataset, and log reuse.

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

## Performance matrix

Run one performance case on PP=2/TP=4 (eight NPUs):

```bash
CPP_RUNNER=mrv2 \
CPP_DYNAMIC=1 \
CPP_EXECUTION_MODE=eager \
CPP_PERF_DATASET=fixed \
CPP_NPU_DEVICES=8,9,10,11,12,13,14,15 \
./proj_0811/cpp_validation/bin/cpp-test perf-case
```

Run the complete five-configuration by two-dataset matrix:

```bash
CPP_NPU_DEVICES=8,9,10,11,12,13,14,15 \
./proj_0811/cpp_validation/bin/cpp-test perf-matrix
```

The fixed dataset contains 5 requests of exactly 131072 input tokens at
concurrency 1 and uses `max_num_batched_tokens=32768`. The variable dataset
contains 64 requests spanning 4096 to 65536 tokens with an exact mean of 32768
at concurrency 4 and uses `max_num_batched_tokens=20480`. Both use request
rate 0 and generate one output token. Variable prompts share a deterministic,
nested 90% token prefix generated from the selected backend; prefix caching is
enabled only for that dataset. Generation metadata records and validates the
planned prefix-hit ratio.

After the service becomes healthy, the fixed AISBench hardware warmup remains
five requests and is excluded from performance metrics:

```bash
python3 aisbench_test.py --input_len 131072 --output_len 1 --data_num 5 --concurrency 1 --request_rate 0
```

For CPP-enabled cases, 30 untimed requests from the target dataset now run
before the fixed manual warmup, so the bounded online timing history is fitted
to the measured distribution instead of being consumed by 128K-only traffic.
Five same-distribution requests are sent again after the manual warmup to
restore the variable prefix-cache state, followed by one untimed request for
the longest shared prefix (the `--prefix_test` equivalent). Static cases keep the conventional
manual-warmup then 30-request warmup order. CPP uses `smooth_factor=0.8`.
Override the manual stage with `CPP_MANUAL_WARMUP_*`, or disable it explicitly with
`CPP_MANUAL_WARMUP_ENABLED=0` for diagnostic comparisons. The normal matrix
keeps it enabled.

The pre-tuning configuration is preserved in
`configs/snapshots/performance_pre_tuning_20260817.json`; do not overwrite this
file when tuning later runs.

The matrix includes both `mrv2_cpp0_eager` and `mrv2_cpp1_eager` for the
fixed and variable datasets. Reports compare these pairs directly as
`cpp_mrv2_vs_mrv2_static`, isolating CPP overhead or benefit without changing
the model runner or execution mode.

Execution order is MRv2 CPP off/on, MRv1 CPP off/on, then MRv2 CPP Graph; each
configuration runs fixed before variable. Reports include TTFT average, P50,
P90, and P95 together with aggregate and per-card input throughput.

The `mrv2_cpp1_graph` cases request `FULL_DECODE_ONLY`. Before the measured
one-token workload, an untimed two-token probe proves that normal inference
actually reaches graph replay. Bounded execution-mode records separately prove
that all 64 CPP startup profiling samples used `CUDAGraphMode.NONE` with
`need_eager=true`. A backend fallback to eager therefore fails validation
instead of being reported as a graph result.

The default model is `/mnt/a800_weight/DeepSeek-V4-Flash-w4a8`. If its startup
log proves a W4A8-specific incompatibility, rerun the entire matrix with
`CPP_MODEL_PATH=/mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp`; never mix model
paths within one comparison run.

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
