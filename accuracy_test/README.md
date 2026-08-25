# DeepSeek V4 Flash 精度测试

本目录用于在 `server-112` 的 `wd_test0811` 容器中，对性能测试同款
`DeepSeek-V4-Flash-w4a8` 权重执行 GSM8K 和 GPQA 精度测试。

脚本不会安装、升级或降级任何包，也不会自动下载数据集或自动切换到备用权重。

## 固定测试范围

- 模型：`/mnt/weight/DeepSeek-V4-Flash-w4a8`
- 服务名：`deepseek-v4-flash`
- 量化：Ascend W4A8
- 卡数：8张物理 NPU，默认 `8,9,10,11,12,13,14,15`
- 并行：PP=2、TP=4
- 数据集：
  - `gsm8k_gen_0_shot_cot_chat_prompt`
  - `gpqa_gen_0_shot_cot_chat_prompt`

默认使用 8192 token 上下文、4096 token 最大输出和4并发，适用于当前两个精度数据集。
这些参数可通过环境变量覆盖，但模型路径、数据集版本和生成参数在对比实验中必须保持一致。

## 1. 进入工作容器

在 `server-112` 宿主机执行：

~~~bash
docker exec -it -w /home/w00985415/proj_0811/accuracy_test wd_test0811 bash
~~~

后续命令均在容器内执行。

## 2. 前置条件

数据集必须放在：

~~~text
/home/w00985415/proj_0811/predict/ais_bench/datasets/
├── gsm8k/test.jsonl
└── gpqa/
    ├── gpqa_diamond.csv
    └── license.txt
~~~

运行只读检查：

~~~bash
./scripts/00_check_env.sh
~~~

该检查不会安装包。若出现 `DEPENDENCY_ERROR`，请由环境维护者处理后重新检查。
当前已知容器曾存在 NumPy 2.3.5 与 SciPy 1.13.1 不兼容的问题，AISBench 运行前
NumPy 必须调整为 `<2.3`。脚本不会代为调整。

模型目录在当前容器中配置为 `/mnt/weight/DeepSeek-V4-Flash-w4a8`。如果该挂载不可访问，环境检查会在15秒后失败；
应先修复挂载，不能静默换用备用权重。确实需要显式改用另一份模型时，可导出：

~~~bash
export ACCURACY_MODEL_PATH=/实际/模型路径
export ACCURACY_TOKENIZER_PATH=/实际/模型路径
~~~

## 3. 启动服务

CPP 与 Model Runner 是两个独立开关。支持以下四种组合：

~~~bash
# MRv2 + 开启 CPP（默认）
./scripts/10_start_service.sh --runner mrv2 --cpp on

# MRv2 + 关闭 CPP
./scripts/10_start_service.sh --runner mrv2 --cpp off

# MRv1 + 开启 CPP
./scripts/10_start_service.sh --runner mrv1 --cpp on

# MRv1 + 关闭 CPP
./scripts/10_start_service.sh --runner mrv1 --cpp off
~~~

只查看最终命令而不启动：

~~~bash
./scripts/10_start_service.sh --runner mrv2 --cpp on --dry-run
~~~

默认端口为 `18080`。启动脚本会检查：

- 8张 NPU 是否与 PP×TP 匹配；
- NPU 是否已有进程；
- 端口是否被占用；
- 模型 `config.json` 是否可访问；
- 是否已有本精度测试启动的服务。

服务在后台启动，日志位于 `logs/service/`。等待模型加载并验证 Chat API：

~~~bash
./scripts/11_check_service.sh --wait 3600 --probe
~~~

## 4. 冒烟测试

服务健康后，分别取 GSM8K 和 GPQA 的前10条：

~~~bash
./scripts/20_smoke_test.sh
~~~

冒烟测试使用 AISBench debug 模式，仅用于验证请求、回答提取和评分链路。

## 5. 正式测试

运行前100条：

~~~bash
./scripts/30_run_accuracy.sh --dataset gsm8k --num-prompts 100
./scripts/30_run_accuracy.sh --dataset gpqa --num-prompts 100
~~~

一次运行两个数据集的前100条：

~~~bash
./scripts/30_run_accuracy.sh --dataset all --num-prompts 100
~~~

执行全量测试：

~~~bash
./scripts/30_run_accuracy.sh --dataset gsm8k --full
./scripts/30_run_accuracy.sh --dataset gpqa --full
~~~

脚本始终添加 `--dump-eval-details`，便于区分模型答错、答案抽取失败和请求失败。

## 6. 停止服务

~~~bash
./scripts/12_stop_service.sh
~~~

停止脚本只处理 `state/service.pid` 记录、且命令行同时匹配本实验模型的进程组。
身份不匹配时会拒绝操作，也不会发送 `SIGKILL`。

## 7. 产物位置

~~~text
logs/service/       vLLM 服务日志
logs/benchmark/     AISBench 控制台日志
logs/environment/   每次服务和测试的参数、版本记录
results/<run-id>/   AISBench predictions、results 和 summary
reports/generated/  自动生成的 Markdown 报告
state/              当前服务 PID 与运行参数
tmp/compiler/       vLLM 编译器临时产物
~~~

每次运行 ID 包含时间、MR版本、CPP状态、数据集和样本数量，例如：

~~~text
20260824_150000_mrv2_cpp1_gsm8k_n100
~~~

## 8. 常用覆盖项

覆盖值应在运行脚本前导出：

~~~bash
export ACCURACY_PORT=18081
export ACCURACY_NPU_DEVICES=0,1,2,3,4,5,6,7
export ACCURACY_MAX_MODEL_LEN=8192
export ACCURACY_MAX_OUT_LEN=4096
export ACCURACY_BATCH_SIZE=4
~~~

使用不同卡号前必须确认8张卡均属于当前任务，且没有其他服务占用。
