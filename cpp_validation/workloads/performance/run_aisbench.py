"""Run AISBench with an available-memory guard suitable for shared servers."""

from __future__ import annotations

import os
from pathlib import Path

import psutil
from ais_bench.benchmark.tasks import openicl_api_infer
from ais_bench.benchmark.tasks import utils as task_utils

MIN_AVAILABLE_BYTES = 8 * 1024**3
MAX_USED_PERCENT = 95
ORIGINAL_MEMORY_CHECK = task_utils.check_virtual_memory_usage


def check_available_memory(dataset_bytes: int, threshold_percent: int = 80) -> None:
    """Keep a hard free-memory floor, then call AISBench with a 95% ceiling.

    AISBench 3.1 hard-codes an 80% used-memory ceiling even when hundreds of
    GiB remain available. The benchmark datasets here are small, so an
    available-memory floor is the relevant safety constraint on shared hosts.
    """

    memory = psutil.virtual_memory()
    required = max(MIN_AVAILABLE_BYTES, dataset_bytes * 2)
    if memory.available < required:
        raise RuntimeError(
            f"insufficient available memory for AISBench: "
            f"available={memory.available}, required={required}"
        )
    ORIGINAL_MEMORY_CHECK(
        dataset_bytes=dataset_bytes,
        threshold_percent=MAX_USED_PERCENT,
    )


def main() -> int:
    sitecustomize_dir = Path(__file__).with_name("aisbench_sitecustomize")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["CPP_AISBENCH_MEMORY_GUARD"] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(sitecustomize_dir),
            *([existing_pythonpath] if existing_pythonpath else []),
        ]
    )
    task_utils.check_virtual_memory_usage = check_available_memory
    openicl_api_infer.check_virtual_memory_usage = check_available_memory
    from ais_bench.benchmark.cli.main import main as aisbench_main

    return int(aisbench_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
