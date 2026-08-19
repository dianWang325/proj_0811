"""Deterministic exact-token datasets for CPP performance measurements."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET
from datasets import Dataset
from transformers import AutoTokenizer

DEFAULT_SUITE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "suites" / "performance.json"


def tokenizer_trust_remote_code() -> bool:
    return os.environ.get("CPP_EFFECTIVE_TOKENIZER_TRUST_REMOTE_CODE", "1") == "1"


def load_performance_suite(suite_config: str | Path | None = None) -> dict:
    path = Path(suite_config) if suite_config else DEFAULT_SUITE_CONFIG
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("suite") != "performance":
        raise ValueError(f"not a performance suite configuration: {path}")
    return data


_DEFAULT_SUITE = load_performance_suite()
FIXED_REQUEST_COUNT = int(_DEFAULT_SUITE["fixed"]["request_count"])
VARIABLE_REQUEST_COUNT = int(_DEFAULT_SUITE["variable"]["request_count"])
# Kept as the variable-dataset count for callers that imported the original
# single-count constant. New code should use the dataset-specific constants.
REQUEST_COUNT = VARIABLE_REQUEST_COUNT
FIXED_INPUT_TOKENS = int(_DEFAULT_SUITE["fixed"]["input_tokens"])
VARIABLE_MIN_TOKENS = int(_DEFAULT_SUITE["variable"]["min_input_tokens"])
VARIABLE_MAX_TOKENS = int(_DEFAULT_SUITE["variable"]["max_input_tokens"])
VARIABLE_MEAN_TOKENS = int(_DEFAULT_SUITE["variable"]["mean_input_tokens"])
VARIABLE_SEED = int(_DEFAULT_SUITE["variable"]["seed"])


def variable_input_lengths(
    seed: int | None = None,
    suite_config: str | Path | None = None,
) -> list[int]:
    """Build the configured deterministic variable-length distribution."""

    suite = load_performance_suite(suite_config)
    variable = suite["variable"]
    minimum = int(variable["min_input_tokens"])
    maximum = int(variable["max_input_tokens"])
    mean = int(variable["mean_input_tokens"])
    count = int(variable["request_count"])
    step = int(variable.get("step_tokens", 4096))
    selected_seed = int(variable["seed"] if seed is None else seed)
    if minimum <= 0 or maximum < minimum or step <= 0 or (maximum - minimum) % step:
        raise ValueError("invalid variable performance token range")
    buckets = list(range(minimum, maximum + 1, step))
    lengths = [buckets[index % len(buckets)] for index in range(count)]
    target_sum = count * mean
    delta = sum(lengths) - target_sum
    if delta % step:
        raise ValueError("variable performance mean cannot be represented by step_tokens")

    # Adjust the evenly distributed bucket plan while preserving at least one
    # minimum and maximum sample. For the default suite this exactly retains
    # the historical two 64K plus one 12K replacement plan.
    while delta > 0:
        candidates = sorted(
            (
                (value - minimum, index)
                for index, value in enumerate(lengths)
                if value > minimum and not (value == maximum and lengths.count(maximum) == 1)
            ),
            reverse=True,
        )
        reduction, index = next(
            ((amount, position) for amount, position in candidates if amount <= delta),
            (0, -1),
        )
        if not reduction:
            raise ValueError("cannot construct configured variable token mean")
        lengths[index] -= reduction
        delta -= reduction
    while delta < 0:
        candidates = sorted(
            (
                (maximum - value, index)
                for index, value in enumerate(lengths)
                if value < maximum and not (value == minimum and lengths.count(minimum) == 1)
            ),
            reverse=True,
        )
        increase, index = next(
            ((amount, position) for amount, position in candidates if amount <= -delta),
            (0, -1),
        )
        if not increase:
            raise ValueError("cannot construct configured variable token mean")
        lengths[index] += increase
        delta += increase

    random.Random(selected_seed).shuffle(lengths)
    assert len(lengths) == count
    assert min(lengths) == minimum
    assert max(lengths) == maximum
    assert sum(lengths) == count * mean
    return lengths


def performance_input_lengths(
    dataset_name: str,
    suite_config: str | Path | None = None,
) -> list[int]:
    suite = load_performance_suite(suite_config)
    if dataset_name == "fixed":
        fixed = suite["fixed"]
        return [int(fixed["input_tokens"])] * int(fixed["request_count"])
    if dataset_name == "variable":
        return variable_input_lengths(suite_config=suite_config)
    raise ValueError(f"unsupported CPP performance dataset: {dataset_name}")


def load_generated_records(dataset_path: str | Path) -> list[dict]:
    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"generated dataset is not readable: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"generated dataset is empty: {path}")
    required = {"question", "answer", "max_out_len", "expected_input_tokens"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"generated dataset row {index} is missing {sorted(missing)}")
    return records


class ExactPromptFactory:
    """Build prompts whose tokenizer round trip preserves the requested size."""

    def __init__(self, model_path: str, variant: int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=tokenizer_trust_remote_code(),
        )
        self.token_id = self._find_round_trip_token(variant)
        self._prompt_cache: dict[int, str] = {}

    def _find_round_trip_token(self, variant: int) -> int:
        stable_ids = []
        for text in (" hello", " world", " test", " A", " B"):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = token_ids[0]
            probe = self.tokenizer.decode([token_id] * 16, skip_special_tokens=False)
            if self.tokenizer.encode(probe, add_special_tokens=False) == [token_id] * 16:
                if token_id not in stable_ids:
                    stable_ids.append(token_id)
        if stable_ids:
            return stable_ids[variant % len(stable_ids)]
        raise RuntimeError("could not find a tokenizer token with a stable repeated round trip")

    def build(self, token_count: int) -> str:
        if token_count in self._prompt_cache:
            return self._prompt_cache[token_count]
        expected_ids = [self.token_id] * token_count
        prompt = self.tokenizer.decode(expected_ids, skip_special_tokens=False)
        actual_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"tokenizer round trip changed requested length {token_count} "
                f"to {len(actual_ids)}"
            )
        self._prompt_cache[token_count] = prompt
        return prompt


@LOAD_DATASET.register_module()
class CPPPerformanceDataset(BaseDataset):
    """AISBench dataset that exposes exact-token prompts and one-token outputs."""

    def load(self, config, **kwargs):
        dataset_name = config["dataset_name"]
        dataset_path = config.get("dataset_path")
        suite_config = config.get("suite_config")
        if dataset_path:
            records = load_generated_records(dataset_path)
            expected_lengths = performance_input_lengths(dataset_name, suite_config)
            actual_lengths = [record["expected_input_tokens"] for record in records]
            if actual_lengths != expected_lengths:
                raise ValueError(
                    f"generated {dataset_name} length plan does not match the "
                    "performance suite contract"
                )
            return Dataset.from_list(records)
        model_path = config.get("model_path") or kwargs.get("model_path")
        if not model_path:
            raise ValueError("CPPPerformanceDataset requires model_path")
        factory = ExactPromptFactory(model_path)
        records = [
            {
                "question": factory.build(token_count),
                "answer": "",
                "max_out_len": 1,
                "expected_input_tokens": token_count,
            }
            for token_count in performance_input_lengths(dataset_name, suite_config)
        ]
        return Dataset.from_list(records)
