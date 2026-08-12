# vLLM Ascend PP=2 smoke test

This test serves `/mnt/a800_weight/Qwen3-30B-A3B-W8A8` on physical NPU IDs
12 and 13 (card group 6) with tensor parallelism 1 and pipeline parallelism 2.

Run all commands from `/home/w00985415` inside container `wd_test0811`:

```bash
./proj_0811/smoke_test/start_server.sh
./proj_0811/smoke_test/smoke_test.py
./proj_0811/smoke_test/stop_server.sh
```

The start script refuses to run if card group 6 has an NPU process, if the
chosen port is in use, or if an earlier recorded test process is alive. The
stop script only signals the isolated process group after verifying its PID,
command line, and model path.

## CPP long-context functional validation

Run one MRv2 + dynamic CPP case with exact 10K, 20K, and 40K-token requests:

```bash
CPP_NPU_DEVICES=8,9 ./proj_0811/smoke_test/run_mrv2_cpp_smoke.sh
```

The default request mode is `both`: the client first sends the three lengths
sequentially to verify online calibration, then sends them concurrently to
exercise multi-request scheduling. The service keeps `max_num_seqs=8`; startup
profiling independently forces a single dummy request.

Run the MRv1/MRv2, static/dynamic, eager/graph compatibility matrix with:

```bash
CPP_NPU_DEVICES=8,9 ./proj_0811/smoke_test/run_cpp_regression_matrix.sh
```

Useful overrides include `CPP_PORT`, `CPP_REQUEST_MODE`, `CPP_TOKEN_TARGETS`,
`CPP_HCCL_PORT_RANGE`, and `CPP_LOG_FILE`. Each case validates the API results,
server errors, startup execution mode, dynamic chunk changes, history fitting,
and the worker timing-disable handshake.
