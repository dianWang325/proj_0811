# Environment setup report

- Run ID: `20260826-180729`
- Generated: `2026-08-26 18:13:12+0800`
- Status: superseded by corrective run `20260826-181501`
- Project: `/home/w00985415/proj_0811`
- Container: `wd_test0811`
- Image: `sha256:ad94d749cc48fd1c29a0f3f926fe78a0cf11686c73f5ef273aa65bc6cd7834e5`
- vLLM commit: `cdc4824a21eaa986d4d1fee90a7e6465c9f706e6`
- vLLM Ascend branch: `feat/modelrunner-v2-cpp-core`
- Recorded incidents: `1` (resolved)

## Result

The source rebuild and five common compatibility pins passed. A later audit found that this first verification did not enforce the branch-specific `torch-npu` and `triton-ascend` requirements. Run `20260826-181501` corrected those packages and is the authoritative final report.

## Logs

Raw logs and the completed report-stage incident are stored under `deps/.bootstrap/runs/20260826-180729/`.
