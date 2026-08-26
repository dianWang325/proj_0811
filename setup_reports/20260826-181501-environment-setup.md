# Environment setup report

- Run ID: `20260826-181501`
- Generated: `2026-08-26 18:19:02+0800`
- Finalized: `2026-08-26 18:19:11+0800`
- Server: `server-27` (`80.5.9.126`)
- Project: `/home/w00985415/proj_0811`
- Container: `wd_test0811`
- Image: `sha256:ad94d749cc48fd1c29a0f3f926fe78a0cf11686c73f5ef273aa65bc6cd7834e5`
- vLLM commit: `cdc4824a21eaa986d4d1fee90a7e6465c9f706e6`
- vLLM Ascend branch: `feat/modelrunner-v2-cpp-core`
- vLLM Ascend commit: `8e3fc05941d1b4c7c42f150a2bfc3ba42cf20fe2`
- Recorded incidents: `2` (all resolved)

## Installed distributions

- torch: `2.10.0`
- torch-npu: `2.10.0.post4`
- triton-ascend: `3.2.2`
- vllm: `0.26.1rc1.dev785+gcdc4824a2.empty`
- vllm-ascend: `0.19.1rc2.dev1651+g8e3fc0594`
- ais-bench-benchmark: `3.1.20260522`
- numpy: `1.26.4`
- scipy: `1.13.1`
- transformers: `5.14.1`
- huggingface-hub: `1.26.0`
- opencv-python-headless: `4.11.0.86`
- fastapi: `0.136.3`

## Verification

All mandatory source revisions, editable import locations, branch-specific Torch/Triton requirements, common exact pins, AISBench, and host/container scoped GitHub checks passed.

`pip check` remains diagnostic. The selected vLLM commit requires FastAPI `>=0.133,<0.137`, while this vLLM Ascend branch declares `fastapi<0.124`; these constraints cannot be satisfied simultaneously. The established common pins also intentionally differ from newer vLLM metadata for `huggingface-hub` and OpenCV. Base-image profiler packages report additional missing or mismatched optional dependencies.

## Resolved incidents

1. The initial branch-requirement parser assumed the wrong `pyproject.toml` table; it now reads `[build-system].requires` and enforces Torch/Triton requirements.
2. A foreground SSH timeout interrupted the logging pipe after compatibility installation; the same run resumed and executed only verification/report stages.

## Logs

Raw logs and completed incident details are stored under `deps/.bootstrap/runs/20260826-181501/` and are excluded from Git through `/deps/`.
