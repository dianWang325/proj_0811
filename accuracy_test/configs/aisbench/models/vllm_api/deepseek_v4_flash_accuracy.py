import os

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (
    extract_non_reasoning_content,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="deepseek-v4-flash-accuracy",
        path=os.environ.get(
            "ACCURACY_TOKENIZER_PATH",
            "/mnt/weight/DeepSeek-V4-Flash-w4a8",
        ),
        model=os.environ.get("ACCURACY_SERVED_MODEL_NAME", "deepseek-v4-flash"),
        stream=False,
        request_rate=_env_float("ACCURACY_REQUEST_RATE", 0),
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip=os.environ.get("ACCURACY_HOST", "127.0.0.1"),
        host_port=_env_int("ACCURACY_PORT", 18080),
        url="",
        max_out_len=_env_int("ACCURACY_MAX_OUT_LEN", 4096),
        batch_size=_env_int("ACCURACY_BATCH_SIZE", 4),
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=_env_float("ACCURACY_TEMPERATURE", 0),
            repetition_penalty=_env_float("ACCURACY_REPETITION_PENALTY", 1),
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
