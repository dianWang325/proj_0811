# Environment setup report

- Run ID: `20260826-181501`
- Generated: `2026-08-26 18:19:02+0800`
- Project: `/home/w00985415/proj_0811`
- Container: `wd_test0811`
- Image: `sha256:ad94d749cc48fd1c29a0f3f926fe78a0cf11686c73f5ef273aa65bc6cd7834e5`
- vLLM commit: `cdc4824a21eaa986d4d1fee90a7e6465c9f706e6`
- vLLM Ascend branch: `feat/modelrunner-v2-cpp-core`
- Recorded incidents: `2`

## Installed distributions

- vllm: `0.26.1rc1.dev785+gcdc4824a2.empty`
- vllm-ascend: `0.19.1rc2.dev1651+g8e3fc0594`
- ais-bench-benchmark: `3.1.20260522`
- numpy: `1.26.4`
- scipy: `1.13.1`
- transformers: `5.14.1`
- huggingface-hub: `1.26.0`
- opencv-python-headless: `4.11.0.86`

## Verification

All mandatory source, revision, version, import, AISBench, and scoped GitHub checks passed. `pip check` was recorded for diagnosis but was not a pass/fail gate.

## Logs

Raw logs and incident details are stored under `deps/.bootstrap/runs/20260826-181501/` and are excluded from Git through `/deps/`.
