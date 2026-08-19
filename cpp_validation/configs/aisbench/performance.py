"""AISBench configuration for one CPP performance case."""

# ruff: noqa: C408

from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.datasets import MATHEvaluator, math_postprocess_v2
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer

from cpp_validation.workloads.performance.dataset import CPPPerformanceDataset

dataset_name = "@@CPP_DATASET@@"
model_path = "@@CPP_MODEL_PATH@@"
model_name = "@@CPP_MODEL_NAME@@"
port = "@@CPP_PORT@@"
concurrency = "@@CPP_CONCURRENCY@@"
request_rate = "@@CPP_REQUEST_RATE@@"
dataset_path = "@@CPP_DATASET_PATH@@"
suite_config = "@@CPP_SUITE_CONFIG@@"

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr="cpp-vllm-api",
        path=model_path,
        model=model_name,
        stream=True,
        request_rate=request_rate,
        retry=1,
        host_ip="127.0.0.1",
        host_port=port,
        max_out_len=1,
        batch_size=concurrency,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        ),
    )
]

reader_cfg = dict(
    input_columns=["question", "max_out_len"],
    output_column="answer",
)
infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version="v2"),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

datasets = [
    dict(
        abbr=f"cpp_{dataset_name}",
        type=CPPPerformanceDataset,
        config=dict(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_path=model_path,
            suite_config=suite_config,
        ),
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]

summarizer = dict(
    attr="performance",
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=DefaultPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "P50", "P75", "P90", "P95", "P99"],
    ),
)
