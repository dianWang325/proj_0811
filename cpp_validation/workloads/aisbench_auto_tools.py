#!/usr/bin/env python3
"""Run aisbench_auto_tools_prefix in an isolated, reproducible workspace."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cpp_validation.workloads.data_generation import (
    DEFAULT_AISBENCH_AUTO_TOOLS_ROOT,
    tool_revision,
)


def find_aisbench_work_path() -> Path:
    import ais_bench

    return Path(ais_bench.__file__).resolve().parent.parent


def prepare_workspace(source: Path, workspace: Path) -> None:
    required = ("aisbench_test.py", "default_api.py", "GSM8K.jsonl")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(
            f"invalid aisbench_auto_tools_prefix installation at {source}: "
            f"missing={missing}"
        )
    if workspace.exists():
        raise RuntimeError(f"isolated AISBench workspace already exists: {workspace}")
    shutil.copytree(
        source,
        workspace,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "outputs", "aisbench.log", "aisbench_all.log",
            "aisbench_result.csv", "picked_ids.txt", "temp_api.py",
        ),
    )


def render_config(
    *,
    workspace: Path,
    dataset_dir: Path,
    work_path: Path,
    model_path: str,
    model_name: str,
    host: str,
    port: int,
    output_dir: Path,
) -> None:
    values = {
        "DATASET_PATH": str(dataset_dir),
        "WORK_PATH": str(work_path),
        "MODEL_NAME": model_name,
        "MODEL_PATH": model_path,
        "HOST_IP": host,
        "HOST_PORT": str(port),
        "API_KEY": "",
        "DEFAULT_PERFORMANCE_TEST": "default_perf",
        "OUTPUT_DIR": str(output_dir),
        "POD_INFO": [],
    }
    lines = [f"{key} = {value!r}" for key, value in values.items()]
    (workspace / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command(
    *,
    input_len: int,
    output_len: int,
    data_num: int,
    concurrency: int,
    request_rate: str,
    dataset_type: str = "normal",
    repeat_rate: str | None = None,
    prefix_test: bool = False,
    length_mean: int | None = None,
    length_std: float | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "aisbench_test.py",
        "--input_len",
        str(input_len),
        "--output_len",
        str(output_len),
        "--data_num",
        str(data_num),
        "--concurrency",
        str(concurrency),
        "--request_rate",
        request_rate,
    ]
    if dataset_type != "normal":
        command.extend(["--dataset_type", dataset_type])
    if repeat_rate is not None:
        command.extend(["--repeat_rate", repeat_rate])
    if prefix_test:
        command.append("--prefix_test")
    optional_lengths = (
        ("--length_mean", length_mean),
        ("--length_std", length_std),
        ("--length_min", length_min),
        ("--length_max", length_max),
    )
    for flag, value in optional_lengths:
        if value is not None:
            rendered = (
                str(int(value))
                if isinstance(value, float) and value.is_integer()
                else str(value)
            )
            command.extend([flag, rendered])
    return command


def validate_aisbench_run(
    workspace: Path, *, expected_requests: int, expected_output_tokens: int
) -> tuple[dict, list[str]]:
    detail_paths = sorted((workspace / "outputs").rglob("*_details.jsonl"))
    rows = []
    for path in detail_paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    errors = []
    if len(rows) != expected_requests:
        errors.append(
            f"AISBench warmup detail rows={len(rows)}, expected={expected_requests}"
        )
    failed = [row for row in rows if row.get("success") is not True]
    if failed:
        reasons = sorted(
            {str(row.get("error_info") or "unknown failure") for row in failed}
        )
        errors.append(
            f"AISBench warmup failed requests={len(failed)}: {reasons[:3]}"
        )
    bad_outputs = [
        row
        for row in rows
        if row.get("success") is True
        and int(row.get("output_tokens") or 0) != expected_output_tokens
    ]
    if bad_outputs:
        errors.append(
            "AISBench warmup output-token mismatch for "
            f"{len(bad_outputs)} successful requests"
        )
    evidence = {
        "detail_files": [str(path) for path in detail_paths],
        "request_count": len(rows),
        "success_requests": len(rows) - len(failed),
        "failed_requests": len(failed),
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
    }
    return evidence, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool-root", type=Path, default=DEFAULT_AISBENCH_AUTO_TOOLS_ROOT
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=131072)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--data-num", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-rate", default="0")
    parser.add_argument(
        "--dataset-type", choices=("normal", "prefix_cache"), default="normal"
    )
    parser.add_argument("--repeat-rate")
    parser.add_argument("--prefix-test", action="store_true")
    parser.add_argument("--length-mean", type=int)
    parser.add_argument("--length-std", type=float)
    parser.add_argument("--length-min", type=int)
    parser.add_argument("--length-max", type=int)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    positive = {
        "input_len": args.input_len,
        "output_len": args.output_len,
        "data_num": args.data_num,
        "concurrency": args.concurrency,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        parser.error(f"warmup values must be positive: {invalid}")
    if (args.length_mean is None) != (args.length_std is None):
        parser.error("--length-mean and --length-std must be provided together")
    if (args.length_min is None) != (args.length_max is None):
        parser.error("--length-min and --length-max must be provided together")
    if args.dataset_type == "prefix_cache" and args.repeat_rate is None:
        parser.error("prefix_cache requires --repeat-rate")

    source = args.tool_root.resolve()
    workspace = args.workspace.resolve()
    dataset_dir = workspace / "datasets"
    output_dir = workspace / "outputs"
    prepare_workspace(source, workspace)
    dataset_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    render_config(
        workspace=workspace,
        dataset_dir=dataset_dir,
        work_path=find_aisbench_work_path(),
        model_path=args.model_path,
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        output_dir=output_dir,
    )
    command = build_command(
        input_len=args.input_len,
        output_len=args.output_len,
        data_num=args.data_num,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        dataset_type=args.dataset_type,
        repeat_rate=args.repeat_rate,
        prefix_test=args.prefix_test,
        length_mean=args.length_mean,
        length_std=args.length_std,
        length_min=args.length_min,
        length_max=args.length_max,
    )
    display_command = "python3 " + " ".join(
        shlex.quote(value) for value in command[1:]
    )
    metadata = {
        "schema_version": 1,
        "phase": "manual_warmup",
        "tool_root": str(source),
        "tool_revision": tool_revision(source),
        "workspace": str(workspace),
        "command": display_command,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "data_num": args.data_num,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "dataset_type": args.dataset_type,
        "repeat_rate": args.repeat_rate,
        "prefix_test": args.prefix_test,
        "length_mean": args.length_mean,
        "length_std": args.length_std,
        "length_min": args.length_min,
        "length_max": args.length_max,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "exit_code": None,
        "aisbench_evidence": None,
        "validation_errors": [],
    }
    print(f"CPP_MANUAL_WARMUP_COMMAND {display_command}", flush=True)

    exit_code = 0
    if not args.dry_run:
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parents[2]
        sitecustomize = (
            project_root
            / "cpp_validation"
            / "workloads"
            / "performance"
            / "aisbench_sitecustomize"
        )
        pythonpath = [str(sitecustomize), str(project_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["CPP_AISBENCH_MEMORY_GUARD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        completed = subprocess.run(command, cwd=workspace, env=env, check=False)
        exit_code = completed.returncode
        evidence, validation_errors = validate_aisbench_run(
            workspace,
            expected_requests=args.data_num,
            expected_output_tokens=args.output_len,
        )
        metadata["aisbench_evidence"] = evidence
        metadata["validation_errors"] = validation_errors
        if validation_errors and exit_code == 0:
            # The upstream shell pipeline ends in tee, so its process status can
            # be zero even when AISBench itself failed. Treat request evidence
            # as authoritative.
            exit_code = 1

    metadata["exit_code"] = exit_code
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if exit_code:
        print(f"CPP_MANUAL_WARMUP_FAIL exit_code={exit_code}", file=sys.stderr)
        return exit_code
    print(
        f"CPP_MANUAL_WARMUP_PASS requests={args.data_num} workspace={workspace}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
