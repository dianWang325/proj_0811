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

CPP-specific functional, performance, and accuracy validation lives in
`../cpp_validation`. Runtime outputs are stored separately under
`../artifacts/cpp`.
