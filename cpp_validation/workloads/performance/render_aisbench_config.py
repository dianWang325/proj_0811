#!/usr/bin/env python3
"""Render an MMEngine-compatible AISBench config with literal values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def escaped_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fixed", "variable"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    args = parser.parse_args()

    content = args.template.read_text(encoding="utf-8")
    replacements = {
        "@@CPP_DATASET@@": escaped_string(args.dataset),
        "@@CPP_MODEL_PATH@@": escaped_string(args.model_path),
        "@@CPP_MODEL_NAME@@": escaped_string(args.model_name),
        '"@@CPP_PORT@@"': str(args.port),
        '"@@CPP_CONCURRENCY@@"': str(args.concurrency),
        '"@@CPP_REQUEST_RATE@@"': repr(args.request_rate),
        "@@CPP_DATASET_PATH@@": escaped_string(str(args.dataset_path.resolve())),
    }
    for marker, value in replacements.items():
        if marker not in content:
            raise RuntimeError(f"AISBench template marker is missing: {marker}")
        content = content.replace(marker, value)
    remaining = [marker for marker in replacements if marker in content]
    if remaining:
        raise RuntimeError(f"AISBench template markers remain: {remaining}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"CPP_AISBENCH_CONFIG_RENDERED output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
