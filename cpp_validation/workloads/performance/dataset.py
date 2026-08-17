"""Deterministic exact-token datasets for CPP performance measurements."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET
from datasets import Dataset
from transformers import AutoTokenizer

FIXED_REQUEST_COUNT = 5
VARIABLE_REQUEST_COUNT = 64
# Kept as the variable-dataset count for callers that imported the original
# single-count constant. New code should use the dataset-specific constants.
REQUEST_COUNT = VARIABLE_REQUEST_COUNT
FIXED_INPUT_TOKENS = 131072
VARIABLE_MIN_TOKENS = 4096
VARIABLE_MAX_TOKENS = 65536
VARIABLE_MEAN_TOKENS = 32768
VARIABLE_SEED = 811


def variable_input_lengths(seed: int = VARIABLE_SEED) -> list[int]:
    """Return 64 reproducible lengths spanning 4K-64K with exact 32K mean."""

    lengths = []
    for token_count in range(VARIABLE_MIN_TOKENS, VARIABLE_MAX_TOKENS + 1, 4096):
        lengths.extend([token_count] * 4)

    # Four repeats of every 4K bucket have a 34K mean. Replace two 64K
    # samples and one 12K sample with 4K samples to reduce the sum by 128K.
    for value in (VARIABLE_MAX_TOKENS, VARIABLE_MAX_TOKENS, 12288):
        lengths.remove(value)
        lengths.append(VARIABLE_MIN_TOKENS)

    random.Random(seed).shuffle(lengths)
    assert len(lengths) == VARIABLE_REQUEST_COUNT
    assert min(lengths) == VARIABLE_MIN_TOKENS
    assert max(lengths) == VARIABLE_MAX_TOKENS
    assert sum(lengths) == VARIABLE_REQUEST_COUNT * VARIABLE_MEAN_TOKENS
    return lengths


def performance_input_lengths(dataset_name: str) -> list[int]:
    if dataset_name == "fixed":
        return [FIXED_INPUT_TOKENS] * FIXED_REQUEST_COUNT
    if dataset_name == "variable":
        return variable_input_lengths()
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

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.token_id = self._find_round_trip_token()
        self._prompt_cache: dict[int, str] = {}

    def _find_round_trip_token(self) -> int:
        for text in (" hello", " world", " test", " A", " B"):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = token_ids[0]
            probe = self.tokenizer.decode([token_id] * 16, skip_special_tokens=False)
            if self.tokenizer.encode(probe, add_special_tokens=False) == [token_id] * 16:
                return token_id
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
        if dataset_path:
            records = load_generated_records(dataset_path)
            expected_lengths = performance_input_lengths(dataset_name)
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
            for token_count in performance_input_lengths(dataset_name)
        ]
        return Dataset.from_list(records)
