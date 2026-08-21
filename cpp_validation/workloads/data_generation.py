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


def tokenizer_trust_remote_code() -> bool:
    return os.environ.get("CPP_EFFECTIVE_TOKENIZER_TRUST_REMOTE_CODE", "1") == "1"


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


def generate_with_script(*, model_path: str, lengths: list[int], seed: int) -> list[str]:
    """Retain the original exact repeated-token generator as the fallback path."""

    from cpp_validation.workloads.performance.dataset import ExactPromptFactory

    factory = ExactPromptFactory(model_path, variant=seed)
    return [factory.build(length) for length in lengths]


def parse_prefix_repeat_rate(value: str | float | None) -> float | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if rendered.endswith("%"):
        ratio = float(rendered[:-1]) / 100.0
    else:
        ratio = float(rendered)
    if not 0 < ratio < 1:
        raise ValueError(f"prefix repeat rate must be in (0, 1): {value}")
    return ratio


def apply_shared_prefix(
    model_path: str,
    prompts: list[str],
    lengths: list[int],
    repeat_rate: float,
) -> tuple[list[str], list[int]]:
    """Create nested common prefixes without changing the exact length plan.

    The prompt bodies still come from the selected generator. Every row takes
    the requested fraction from one common token stream and its remaining
    suffix from its original generated prompt. Variable-length rows therefore
    share the shorter prefix with all longer rows, matching the prefix-cache
    semantics of aisbench_auto_tools_prefix while retaining our deterministic
    4K--64K, exact-mean contract.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code(),
    )
    prefix_lengths = [int(round(length * repeat_rate)) for length in lengths]
    max_prefix_length = max(prefix_lengths)
    anchor_index = max(range(len(lengths)), key=lengths.__getitem__)
    anchor_ids = tokenizer.encode(prompts[anchor_index], add_special_tokens=False)
    if len(anchor_ids) < max_prefix_length:
        raise RuntimeError("prefix anchor is shorter than the requested common prefix")
    common_ids = anchor_ids[:max_prefix_length]

    prefixed = []
    for prompt, length, prefix_length in zip(
        prompts, lengths, prefix_lengths, strict=True
    ):
        suffix_length = length - prefix_length
        source_ids = tokenizer.encode(prompt, add_special_tokens=False)
        suffix_ids = source_ids[-suffix_length:] if suffix_length else []
        combined_ids = common_ids[:prefix_length] + suffix_ids
        prefixed.append(tokenizer.decode(combined_ids, skip_special_tokens=False))
    return prefixed, prefix_lengths


def validate_shared_prefixes(
    model_path: str,
    prompts: list[str],
    prefix_lengths: list[int],
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code(),
    )
    encoded = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    anchor_index = max(range(len(prefix_lengths)), key=prefix_lengths.__getitem__)
    common_ids = encoded[anchor_index][: prefix_lengths[anchor_index]]
    actual_prefix_lengths = []
    excessive_drift = []
    for index, (token_ids, prefix_length) in enumerate(
        zip(encoded, prefix_lengths, strict=True)
    ):
        actual = 0
        for actual, (token_id, common_id) in enumerate(
            zip(token_ids[:prefix_length], common_ids[:prefix_length], strict=True),
            start=1,
        ):
            if token_id != common_id:
                actual -= 1
                break
        else:
            actual = prefix_length
        actual_prefix_lengths.append(actual)
        # Decode/encode can retokenize the final text boundary. Keep this
        # bounded to less than one cache block; larger drift indicates that
        # the rows no longer represent the intended shared-prefix workload.
        if prefix_length - actual >= 32:
            excessive_drift.append(
                {"index": index, "planned": prefix_length, "actual": actual}
            )
    if excessive_drift:
        raise RuntimeError(
            "generated prompts lost at least one cache block of shared prefix: "
            f"{excessive_drift[:5]}"
        )
    return actual_prefix_lengths


def validate_prompts(model_path: str, prompts: list[str], lengths: list[int]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code(),
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


def validate_dataset_isolation(
    model_path: str,
    prompts: list[str],
    other_path: Path,
    prefix_lengths: list[int] | None,
) -> None:
    """Reject generated warmup content that could prime measurement cache keys."""

    if not other_path.is_file():
        raise ValueError(f"disjoint comparison dataset is not readable: {other_path}")
    other_records = [
        json.loads(line)
        for line in other_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    other_prompts = [record["question"] for record in other_records]
    overlap = set(prompts).intersection(other_prompts)
    if overlap:
        raise RuntimeError(
            f"generated warmup dataset overlaps measurement prompts: count={len(overlap)}"
        )
    if prefix_lengths is None:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code(),
    )
    anchor_index = max(range(len(prefix_lengths)), key=prefix_lengths.__getitem__)
    prefix_length = prefix_lengths[anchor_index]
    current_prefix = tokenizer.encode(
        prompts[anchor_index], add_special_tokens=False
    )[:prefix_length]
    for other_prompt in other_prompts:
        other_ids = tokenizer.encode(other_prompt, add_special_tokens=False)
        comparison_length = min(prefix_length, len(other_ids))
        if comparison_length < 32:
            continue
        if current_prefix[:comparison_length] == other_ids[:comparison_length]:
            raise RuntimeError(
                "generated warmup dataset reuses a measurement prefix"
            )


def repair_aisbench_prompt_lengths(
    model_path: str, prompts: list[str], lengths: list[int]
) -> tuple[list[str], list[dict]]:
    """Repair rare upstream decode/encode off-by-one rows without changing the plan."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code(),
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
    prefix_repeat_rate: str | float | None = None,
    prefix_test: bool = False,
    disjoint_from: Path | None = None,
) -> tuple[list[dict], dict]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported data generator: {backend}")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("input token lengths must be non-empty and positive")
    if output_tokens <= 0:
        raise ValueError("output token length must be positive")

    repeat_rate = parse_prefix_repeat_rate(prefix_repeat_rate)
    if prefix_test and repeat_rate is None:
        raise ValueError("prefix test requires a prefix repeat rate")

    if backend == "aisbench":
        prompts = generate_with_aisbench_auto_tools(
            model_path=model_path,
            lengths=lengths,
            seed=seed,
            tool_root=tool_root,
        )
        backend_revision = tool_revision(tool_root)
    else:
        prompts = generate_with_script(model_path=model_path, lengths=lengths, seed=seed)
        backend_revision = None

    prefix_lengths = None
    if repeat_rate is not None:
        prompts, prefix_lengths = apply_shared_prefix(
            model_path, prompts, lengths, repeat_rate
        )
    if backend == "aisbench" or repeat_rate is not None:
        prompts, token_length_repairs = repair_aisbench_prompt_lengths(
            model_path, prompts, lengths
        )
    else:
        token_length_repairs = []
    actual_prefix_lengths = None
    if prefix_lengths is not None:
        actual_prefix_lengths = validate_shared_prefixes(
            model_path, prompts, prefix_lengths
        )

    actual_lengths = validate_prompts(model_path, prompts, lengths)
    if disjoint_from is not None:
        validate_dataset_isolation(
            model_path, prompts, disjoint_from, prefix_lengths
        )
    records = [
        {
            "index": index,
            "question": prompt,
            "answer": "",
            "max_out_len": output_tokens,
            "expected_input_tokens": length,
        }
        for index, (prompt, length) in enumerate(zip(prompts, actual_lengths, strict=True))
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
        "disjoint_from": str(disjoint_from) if disjoint_from is not None else None,
        "aisbench_auto_tools_root": str(tool_root) if backend == "aisbench" else None,
        "aisbench_auto_tools_revision": backend_revision,
        "prefix_cache": {
            "enabled": repeat_rate is not None,
            "dataset_type": "prefix_cache" if repeat_rate is not None else "normal",
            "repeat_rate": repeat_rate,
            "prefix_test": prefix_test,
            "planned_prefix_token_lengths": prefix_lengths,
            "actual_shared_prefix_token_lengths": actual_prefix_lengths,
            "cacheable_prefix_token_lengths": (
                [length // 32 * 32 for length in actual_prefix_lengths]
                if actual_prefix_lengths is not None
                else None
            ),
            "mean_planned_prefix_hit_ratio": (
                sum(prefix / length for prefix, length in zip(
                    prefix_lengths, actual_lengths, strict=True
                )) / len(actual_lengths)
                if prefix_lengths is not None
                else 0.0
            ),
            "mean_actual_prefix_hit_ratio": (
                sum(prefix / length for prefix, length in zip(
                    actual_prefix_lengths, actual_lengths, strict=True
                )) / len(actual_lengths)
                if actual_prefix_lengths is not None
                else 0.0
            ),
            "mean_cacheable_prefix_hit_ratio": (
                sum((prefix // 32 * 32) / length for prefix, length in zip(
                    actual_prefix_lengths, actual_lengths, strict=True
                )) / len(actual_lengths)
                if actual_prefix_lengths is not None
                else 0.0
            ),
        },
    }
    return records, metadata


def parse_lengths(args: argparse.Namespace) -> list[int]:
    if args.performance_dataset:
        from cpp_validation.workloads.performance.dataset import (
            performance_input_lengths,
        )

        return performance_input_lengths(args.performance_dataset, args.suite_config)
    if not args.lengths:
        raise ValueError("either --performance-dataset or --lengths is required")
    return [int(value) for value in args.lengths.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--performance-dataset", choices=("fixed", "variable"))
    parser.add_argument("--suite-config", type=Path)
    parser.add_argument("--lengths", help="comma-separated functional input lengths")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--prefix-repeat-rate")
    parser.add_argument("--prefix-test", action="store_true")
    parser.add_argument("--disjoint-from", type=Path)
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
            prefix_repeat_rate=args.prefix_repeat_rate,
            prefix_test=args.prefix_test,
            disjoint_from=args.disjoint_from,
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
