# CPP × Model Runner V2 适配开发说明

## 1. 文档范围

本文说明 CPP（Profiling-based Dynamic Chunk）适配 Model Runner V2（MRv2）所涉及的单元测试、结构化埋点和新增文件。代码基线为：

- 功能代码：`/home/w00985415/proj_0811/deps/vllm-ascend`
- 开发分支：`feat/modelrunner-v2-cpp-adaptation`
- CPP 验证工具：`/home/w00985415/proj_0811/cpp_validation`
- CPP 运行产物：`/home/w00985415/proj_0811/artifacts/cpp`

文中行号以本文生成时的当前工作区为准；代码继续修改后应同步更新。

## 2. 单元测试分层

### 2.1 新增单元测试文件

| 文件 | 主要用例 | 作用 |
|---|---|---|
| `tests/ut/worker/test_model_runner_v2.py` | `test_dummy_run_is_inherited_from_upstream_model_runner`、`test_dispatch_wrapper_captures_final_execution_mode`、执行计时及 AsyncOutput 用例 | 验证 MRv2 的 `_dummy_run` 继承关系、`NPUModelRunner.execute_model → super().execute_model` 调用、最终 execution mode 捕获，以及实测耗时回传和停止计时。 |
| `tests/ut/core/test_profiling_chunk_trace.py` | `test_online_calibration_completion_is_traced_and_propagated`、`test_trace_disabled_emits_no_cpp_events` | 验证逐轮记录、在线校准完成信号和 batch 级执行时间语义，同时保证默认关闭 trace 时不输出 CPP 事件。 |

### 2.2 既有文件中新增的测试模块

| 文件与行号 | 新增测试模块 | 作用 |
|---|---|---|
| `tests/ut/worker/a2/test_worker_v2.py:59`、`:104` | `profile_prefill_latency` 正常/异常恢复用例 | 验证 MRv2 启动 Profiling 临时设置 `max_num_reqs=1`、关闭已捕获 graph、调用 `_dummy_run`，且成功或异常时均恢复 Runner 状态。 |
| `tests/ut/core/test_profiling_chunk.py:83`、`:424` | trace 配置和预测/实调 chunk 语义用例 | 验证 `trace_enabled` 可配置，以及 `predicted_chunk_size` 保留原始预测、`actual_scheduled_chunk_size` 反映预算截断后的真实下发值。 |

### 2.3 除新增测试文件外必须执行的既有单元测试文件

| 文件 | 作用 |
|---|---|
| `tests/ut/core/test_profiling_chunk.py` | 覆盖启动采样拟合、历史模型拟合、动态 chunk 预测、调度预算和执行结果更新，是 CPP 核心回归集。 |
| `tests/ut/worker/a2/test_worker_v2.py` | 覆盖 NPUWorker 的 MRv2 执行和 PP 行为，并验证 CPP Profiling 临时状态管理。 |
| `tests/ut/test_ascend_config.py` | 覆盖 `profiling_chunk_config` 的默认值、嵌套配置和旧配置兼容，防止新增 `trace_enabled` 破坏配置解析。 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | 对 MRv1 NPUModelRunner 做回归，确保加入 execution mode 和停止计时埋点后不影响既有执行路径。 |
| `tests/ut/test_platform.py` | 对平台级 Worker、Model Runner 和调度配置选择做回归，防止 MRv2 适配影响 MRv1/MRv2 装配逻辑。 |

推荐一次性执行：

```bash
cd /home/w00985415/proj_0811/deps/vllm-ascend
pytest -q \
  tests/ut/worker/test_model_runner_v2.py \
  tests/ut/core/test_profiling_chunk_trace.py \
  tests/ut/worker/a2/test_worker_v2.py \
  tests/ut/core/test_profiling_chunk.py \
  tests/ut/test_ascend_config.py \
  tests/ut/worker/a2/test_model_runner_v1.py \
  tests/ut/test_platform.py
```

前四个聚焦文件当前已执行通过，共 `35 passed`；后三个属于提交前应补跑的扩大回归集。

## 3. CPP 埋点与日志

### 3.1 开关和格式

- 开关定义：`vllm_ascend/ascend_config.py:709-712`。
- 配置项：`profiling_chunk_config.trace_enabled`，默认 `false`。
- 统一输出函数：`vllm_ascend/core/profiling_chunk_trace.py:25-31`。
- 日志格式：INFO 级别的单行 `[CPP_TRACE] {JSON}`，可按 `event` 字段解析。

### 3.2 事件清单

| 事件 | 触发阶段 | 文件与行号 | 主要说明 |
|---|---|---|---|
| `startup_profile_sample` | 每个启动 Profiling 样本结束 | `vllm_ascend/worker/worker.py:904-925` | 输出 runner、PP rank、token 数、dummy/execute_model 调用链、`need_eager`、最终 `execution_mode` 和耗时。 |
| `startup_profile_completed` | 64 个启动样本拟合完成 | `vllm_ascend/core/scheduler_profiling_chunk.py:203-215` | 输出样本数、`is_ready`、历史模型状态、目标延迟和初始模型系数。 |
| `scheduler_iteration` | 一轮模型执行完成并回传耗时 | 调度上下文采集：`vllm_ascend/core/scheduler_profiling_chunk.py:258-270`、`:428-442`、`:700-711`；事件输出：`vllm_ascend/patch/platform/patch_profiling_chunk.py:160-173` | 将请求级预测/实际下发 chunk 与本轮 batch 实测耗时关联，输出完整在线校准状态。 |
| `history_predictor_updated` | 正式请求样本触发重新拟合 | `vllm_ascend/patch/platform/patch_profiling_chunk.py:145-158` | 输出拟合样本数、历史就绪/完成状态和更新后的二次模型系数。 |
| `online_calibration_completed` | 首块目标耗时和历史拟合均完成 | `vllm_ascend/patch/platform/patch_profiling_chunk.py:66-84`、`:175-182` | 输出 `history_fitted=true` 和 `disable_profiling_timing=true`；同一 Scheduler 生命周期只记录一次。 |
| `worker_profiling_timing_disabled` | Worker 收到停止计时信号 | MRv1：`vllm_ascend/worker/model_runner_v1.py:1785-1803`；MRv2：`vllm_ascend/worker/v2/model_runner.py:241-252` | 证明跨进程标志已到达 Worker，后续不再执行无意义的设备同步计时。 |

### 3.3 关键调用链和状态定位

- MRv2 启动 Profiling：`vllm_ascend/worker/worker.py:872-891` 临时设置单请求和 `_graphs_captured=False`，随后调用继承自上游 `GPUModelRunner` 的 `_dummy_run`。
- MRv2 执行入口：`vllm_ascend/worker/v2/model_runner.py:256-283` 中 NPU `execute_model` 是重写方法；它捕获 dispatch 后的最终 `BatchExecutionDescriptor.cg_mode`，再于 `:275` 调用 `super().execute_model(...)`。
- MRv1 execution mode：`vllm_ascend/worker/model_runner_v1.py:3294-3300` 在 CPP dummy run 中记录最终 CUDAGraph mode。
- Scheduler → Worker 停止计时：`vllm_ascend/patch/platform/patch_profiling_chunk.py:211-234` 将 `disable_profiling_timing` 附加到后续 `SchedulerOutput`。

### 3.4 `scheduler_iteration` 字段语义

| 字段 | 含义 |
|---|---|
| `iteration`、`req_id` | Scheduler 轮次及请求标识。 |
| `num_computed_tokens` / `hist_seq_len` | 本轮调度前已完成的请求 Token 数。 |
| `remaining_prefill_tokens` | 本轮调度前尚未完成的 Prefill Token 数。 |
| `target_latency_ms` | CPP 当前目标延迟。 |
| `predicted_chunk_size` | Predictor 输出的原始对齐结果，未经过剩余 Token 和本轮预算截断。 |
| `actual_scheduled_chunk_size` | 经过剩余 Token、调度预算等约束后真正下发的 Token 数。 |
| `predicted_latency_ms` | Predictor 对真实下发 chunk 的预计耗时。 |
| `actual_execution_time_ms` / `batch_execution_time_ms` | 本轮整个 batch 的实测耗时；两者当前相同。 |
| `execution_time_scope` | 固定为 `batch`，明确实测时间不是单请求独占耗时；并发时同一轮请求共享该值。 |
| `with_history_ready`、`history_fitted`、`fit_sample_count` | 在线历史样本及模型拟合进度。 |
| `predictor_updated` | 本轮是否重新拟合 Predictor。 |
| `disable_profiling_timing` | 是否已满足条件并要求 Worker 停止同步计时。 |

## 4. 文件清单

### 4.1 新增生产代码

| 文件 | 简洁说明 |
|---|---|
| `deps/vllm-ascend/vllm_ascend/core/profiling_chunk_trace.py` | CPP 结构化 JSON 日志的统一输出入口。 |

### 4.2 新增单元测试

| 文件 | 简洁说明 |
|---|---|
| `deps/vllm-ascend/tests/ut/worker/test_model_runner_v2.py` | MRv2 CPP 调用链、execution mode、执行计时和 AsyncOutput 回传测试。 |
| `deps/vllm-ascend/tests/ut/core/test_profiling_chunk_trace.py` | 在线校准状态、停止计时信号、batch 耗时语义及 trace 默认关闭测试。 |

### 4.3 新增功能验证工具（非单元测试）

| 文件 | 简洁说明 |
|---|---|
| `cpp_validation/workloads/functional/long_context.py` | 使用本地 tokenizer 构造并校验 10K/20K/40K 输入，支持顺序、并发或两者组合请求。 |
| `cpp_validation/validators/functional/cpp_trace.py` | 解析 `[CPP_TRACE]`，自动检查启动调用链、execution mode、动态 chunk 和在线校准生命周期，并写入结构化结果。 |
| `cpp_validation/scripts/run_case.sh` | 统一启动并验收单个 MRv1/MRv2、Static/Dynamic、Eager/Graph-enabled 功能组合。 |
| `cpp_validation/scripts/run_matrix.sh` | 编排配置文件中定义的功能回归矩阵。 |
| `cpp_validation/scripts/run_quick.sh` | MRv2 Dynamic CPP 冒烟验证的快捷入口。 |
| `cpp_validation/bin/cpp-test` | CPP 快速、单用例、矩阵和报告生成的统一命令入口。 |

### 4.4 关键修改文件（非新增文件）

| 分类 | 文件 | 简洁说明 |
|---|---|---|
| 配置 | `vllm_ascend/ascend_config.py` | 新增默认关闭的 `trace_enabled`。 |
| 调度 | `vllm_ascend/core/scheduler_profiling_chunk.py` | 生成启动完成事件，并保存 Predictor 原始结果与真实调度结果。 |
| 校准 | `vllm_ascend/patch/platform/patch_profiling_chunk.py` | 回收 batch 实测时间、驱动在线拟合、输出逐轮事件并跨进程下发停止计时信号。 |
| Worker | `vllm_ascend/worker/worker.py` | 适配 MRv2 启动 Profiling 临时状态并输出每个启动样本。 |
| MRv2 | `vllm_ascend/worker/v2/model_runner.py` | 重写执行入口中的计时与日志逻辑，捕获最终 execution mode，并委托上游执行实现。 |
| MRv1 | `vllm_ascend/worker/model_runner_v1.py` | 保持 MRv1 兼容，补充 execution mode 与停止计时事件。 |
| 用户文档 | `docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md` | 说明 `trace_enabled` 配置和日志用途。 |
| 既有测试 | `tests/ut/core/test_profiling_chunk.py`、`tests/ut/worker/a2/test_worker_v2.py` | 增补配置、调度记录语义和 MRv2 Profiling 状态恢复测试。 |

## 5. 验收关系

1. `startup_profile_sample` 证明启动 Profiling 进入 `_dummy_run` 和 NPU/上游 `execute_model`，并给出最终 execution mode。
2. `startup_profile_completed` 证明启动样本拟合完成，CPP Predictor 可参与调度。
3. `scheduler_iteration` 将“预测 chunk → 实际下发 chunk → batch 实测时间”关联为同一轮记录。
4. `history_predictor_updated` 证明正式请求的实测数据被用于在线更新 Predictor。
5. `online_calibration_completed` 和 `worker_profiling_timing_disabled` 依次证明校准结束及 Worker 真正停止计时。

上述链路同时由单元测试验证控制流和字段语义，由 `cpp_validation` 验证真实服务、多进程和 NPU 执行结果；两类测试互补，不能互相替代。
