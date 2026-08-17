"""Propagate the CPP AISBench memory guard into spawned Python workers."""

from __future__ import annotations

import os

if os.environ.get("CPP_AISBENCH_MEMORY_GUARD") == "1":
    from ais_bench.benchmark.tasks import openicl_api_infer
    from ais_bench.benchmark.tasks import utils as task_utils

    from cpp_validation.workloads.performance.run_aisbench import (
        check_available_memory,
    )

    task_utils.check_virtual_memory_usage = check_available_memory
    openicl_api_infer.check_virtual_memory_usage = check_available_memory
