#!/usr/bin/env python3
"""Generate functional and performance datasets through a selected backend."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

DEFAULT_BACKEND = "aisbench"
SUPPORTED_BACKENDS = ("aisbench", "script")
DEFAULT_AISBENCH_AUTO_TOOLS_ROOT = Path(
    os.environ.get(
        "CPP_AISBENCH_AUTO_TOOLS_ROOT",
        "/home/w00985415/tools/aisbench_auto_tools_prefix",
    )
)


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def tool_revision(tool_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(tool_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_upstream_create_dataset(tool_root: Path) -> Callable:
    required = ("aisbench_test.py", "generate_dataset.py", "GSM8K.jsonl")
    missing = [name for name in required if not (tool_root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"invalid aisbench_auto_tools_prefix installation at {tool_root}: "
            f"missing={missing}"
        )

    root_string = str(tool_root)
    sys.path.insert(0, root_string)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module("generate_dataset")
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if sys.path[0] == root_string:
            sys.path.pop(0)
    return module.create_dataset


def generate_with_aisbench_auto_tools(
    *,
    model_path: str,
    lengths: list[int],
    seed: int,
    tool_root: Path,
) -> list[str]:
    """Use the upstream prompt generator while preserving an exact length plan."""

    create_dataset = load_upstream_create_dataset(tool_root)
    positions: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        positions[length].append(index)

    prompts: list[str | None] = [None] * len(lengths)
    with working_directory(tool_root):
        for length in sorted(positions):
            # Upstream uses the process-global random module. Seed every bucket
            # so a repeated run does not depend on which suites ran before it.
            random.seed(seed + length)
            generated = create_dataset(
                model_path,
                length,
                len(positions[length]),
                0,
            )
            if generated is None or len(generated) != len(positions[length]):
                raise RuntimeError(
                    "aisbench_auto_tools_prefix did not generate the requested "
                    f"records for input length {length}"
                )
            for index, prompt in zip(positions[length], generated, strict=True):
                prompts[index] = prompt

    if any(prompt is None for prompt in prompts):
        raise RuntimeError("aisbench_auto_tools_prefix left dataset rows ungenerated")
    return [prompt for prompt in prompts if prompt is not None]


def generate_with_script(*, model_path: str, lengths: list[int]) -> list[str]:
    """Retain the original exact repeated-token generator as the fallback path."""

    from cpp_validation.workloads.performance.dataset import ExactPromptFactory

    factory = ExactPromptFactory(model_path)
    return [factory.build(length) for length in lengths]


def validate_prompts(model_path: str, prompts: list[str], lengths: list[int]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    actual_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    if actual_lengths != lengths:
        mismatches = [
            {"index": index, "expected": expected, "actual": actual}
            for index, (expected, actual) in enumerate(
                zip(lengths, actual_lengths, strict=True)
            )
            if expected != actual
        ]
        raise RuntimeError(f"generated prompt token lengths changed: {mismatches[:5]}")
    return actual_lengths


def repair_aisbench_prompt_lengths(
    model_path: str, prompts: list[str], lengths: list[int]
) -> tuple[list[str], list[dict]]:
    """Repair rare upstream decode/encode off-by-one rows without changing the plan."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    filler = None
    for candidate in (" hello", " world", " test", " A", " B"):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        probe = tokenizer.decode(token_ids * 16, skip_special_tokens=False)
        if tokenizer.encode(probe, add_special_tokens=False) == token_ids * 16:
            filler = candidate
            break
    if filler is None:
        raise RuntimeError("could not find a stable one-token repair filler")

    repaired = list(prompts)
    repairs = []
    for index, (prompt, target) in enumerate(zip(repaired, lengths, strict=True)):
        original_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        current_length = original_length
        for _ in range(8):
            if current_length == target:
                break
            if current_length < target:
                prompt += filler * (target - current_length)
            else:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:target]
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            current_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if current_length != target:
            raise RuntimeError(
                "could not repair aisbench_auto_tools_prefix token round trip: "
                f"index={index}, target={target}, actual={current_length}"
            )
        if original_length != current_length:
            repairs.append(
                {
                    "index": index,
                    "before": original_length,
                    "after": current_length,
                    "target": target,
                    "strategy": "stable_token_suffix_or_truncate",
                }
            )
        repaired[index] = prompt
    return repaired, repairs


def generate_records(
    *,
    backend: str,
    model_path: str,
    lengths: list[int],
    output_tokens: int,
    seed: int,
    tool_root: Path = DEFAULT_AISBENCH_AUTO_TOOLS_ROOT,
) -> tuple[list[dict], dict]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported data generator: {backend}")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("input token lengths must be non-empty and positive")
    if output_tokens <= 0:
        raise ValueError("output token length must be positive")

    if backend == "aisbench":
        prompts = generate_with_aisbench_auto_tools(
            model_path=model_path,
            lengths=lengths,
            seed=seed,
            tool_root=tool_root,
        )
        prompts, token_length_repairs = repair_aisbench_prompt_lengths(
            model_path, prompts, lengths
        )
        backend_revision = tool_revision(tool_root)
    else:
        prompts = generate_with_script(model_path=model_path, lengths=lengths)
        token_length_repairs = []
        backend_revision = None

    actual_lengths = validate_prompts(model_path, prompts, lengths)
    records = [
        {
            "index": index,
            "question": prompt,
            "answer": "",
            "max_out_len": output_tokens,
            "expected_input_tokens": length,
        }
        for index, (prompt, length) in enumerate(
            zip(prompts, actual_lengths, strict=True)
        )
    ]
    metadata = {
        "schema_version": 1,
        "backend": backend,
        "record_count": len(records),
        "seed": seed,
        "model_path": model_path,
        "output_tokens": output_tokens,
        "input_token_lengths": actual_lengths,
        "min_input_tokens": min(actual_lengths),
        "max_input_tokens": max(actual_lengths),
        "mean_input_tokens": sum(actual_lengths) / len(actual_lengths),
        "total_input_tokens": sum(actual_lengths),
        "token_length_repairs": token_length_repairs,
        "aisbench_auto_tools_root": str(tool_root) if backend == "aisbench" else None,
        "aisbench_auto_tools_revision": backend_revision,
    }
    return records, metadata


def parse_lengths(args: argparse.Namespace) -> list[int]:
    if args.performance_dataset:
        from cpp_validation.workloads.performance.dataset import (
            performance_input_lengths,
        )

        return performance_input_lengths(args.performance_dataset)
    if not args.lengths:
        raise ValueError("either --performance-dataset or --lengths is required")
    return [int(value) for value in args.lengths.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--performance-dataset", choices=("fixed", "variable"))
    parser.add_argument("--lengths", help="comma-separated functional input lengths")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument(
        "--aisbench-auto-tools-root",
        type=Path,
        default=DEFAULT_AISBENCH_AUTO_TOOLS_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    if args.performance_dataset and args.lengths:
        parser.error("--performance-dataset and --lengths are mutually exclusive")
    try:
        lengths = parse_lengths(args)
        records, metadata = generate_records(
            backend=args.backend,
            model_path=args.model_path,
            lengths=lengths,
            output_tokens=args.output_tokens,
            seed=args.seed,
            tool_root=args.aisbench_auto_tools_root,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "CPP_DATASET_GENERATED "
        f"backend={args.backend} records={len(records)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
