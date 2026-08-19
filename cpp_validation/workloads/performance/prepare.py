#!/usr/bin/env python3
"""Run untimed CPP warmup requests or the graph-mode decode probe."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cpp_validation.workloads.performance.dataset import (
    ExactPromptFactory,
    load_generated_records,
    performance_input_lengths,
    tokenizer_trust_remote_code,
)

TRACE_PATTERN = re.compile(r"\[CPP_EXECUTION_MODE_TRACE\]\s+(\{.*\})")


def read_trace_events_since(log_path: Path, offset: int) -> list[dict]:
    with log_path.open("rb") as handle:
        handle.seek(offset)
        content = handle.read().decode("utf-8", errors="replace")
    events = []
    for line in content.splitlines():
        match = TRACE_PATTERN.search(line)
        if match:
            events.append(json.loads(match.group(1)))
    return events


def stream_completion(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    expected_input_tokens: int,
    output_tokens: int,
    timeout: int,
) -> dict:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    usage = None
    chunks = 0
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            content = line.removeprefix("data:").strip()
            if not content or content == "[DONE]":
                continue
            event = json.loads(content)
            if event.get("usage"):
                usage = event["usage"]
            if event.get("choices"):
                chunks += 1
    if usage is None:
        raise RuntimeError("streaming response did not include token usage")
    if usage.get("prompt_tokens") != expected_input_tokens:
        raise RuntimeError(
            f"prompt token mismatch: expected={expected_input_tokens}, "
            f"actual={usage.get('prompt_tokens')}"
        )
    if usage.get("completion_tokens") != output_tokens:
        raise RuntimeError(
            f"completion token mismatch: expected={output_tokens}, "
            f"actual={usage.get('completion_tokens')}"
        )
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed_seconds": time.perf_counter() - started,
        "stream_chunks": chunks,
    }


def run_requests(
    *,
    samples: list[tuple[str, int]],
    concurrency: int,
    base_url: str,
    model_name: str,
    output_tokens: int,
    timeout: int,
) -> list[dict]:
    def submit(index: int, prompt: str, token_count: int) -> dict:
        result = stream_completion(
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            expected_input_tokens=token_count,
            output_tokens=output_tokens,
            timeout=timeout,
        )
        return {"index": index, "target_input_tokens": token_count, **result}

    records = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(submit, index, prompt, token_count): index
            for index, (prompt, token_count) in enumerate(samples)
        }
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda record: record["index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("warmup", "prefix-prime", "graph-probe"), required=True
    )
    parser.add_argument("--dataset-mode", choices=("generated", "reuse"))
    parser.add_argument("--dataset", choices=("fixed", "variable"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "warmup" and args.dataset_mode is None:
        parser.error("warmup mode requires --dataset-mode")

    prefix_tokens = None
    if args.mode == "graph-probe":
        if args.server_log is None or not args.server_log.is_file():
            parser.error("graph-probe mode requires a readable --server-log")
        factory = ExactPromptFactory(args.model_path)
        samples = [(factory.build(128), 128)]
        concurrency = 1
        output_tokens = 2
    elif args.mode == "prefix-prime":
        if args.dataset_path is None or args.dataset_metadata is None:
            parser.error("prefix-prime requires dataset and metadata paths")
        generated = load_generated_records(args.dataset_path)
        metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
        prefix_config = metadata.get("prefix_cache", {})
        prefix_lengths = prefix_config.get("cacheable_prefix_token_lengths") or \
            prefix_config.get("planned_prefix_token_lengths")
        if not prefix_lengths or len(prefix_lengths) != len(generated):
            parser.error("prefix-prime metadata has no valid prefix length plan")
        index = max(range(len(prefix_lengths)), key=prefix_lengths.__getitem__)
        prefix_tokens = int(prefix_lengths[index])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=tokenizer_trust_remote_code(),
        )
        full_ids = tokenizer.encode(
            generated[index]["question"], add_special_tokens=False
        )
        expected_ids = full_ids[:prefix_tokens]
        prompt = tokenizer.decode(expected_ids, skip_special_tokens=False)
        actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise RuntimeError("prefix-prime tokenizer round trip changed the prefix")
        samples = [(prompt, prefix_tokens)]
        concurrency = 1
        output_tokens = 1
    else:
        if args.count <= 0:
            parser.error("--count must be positive")
        if args.dataset_path:
            generated = load_generated_records(args.dataset_path)
            source_samples = [
                (record["question"], int(record["expected_input_tokens"]))
                for record in generated
            ]
        else:
            source_lengths = performance_input_lengths(args.dataset)
            factory = ExactPromptFactory(args.model_path)
            source_samples = [
                (factory.build(token_count), token_count)
                for token_count in source_lengths
            ]
        samples = [
            source_samples[index % len(source_samples)] for index in range(args.count)
        ]
        concurrency = args.concurrency
        output_tokens = 1

    server_log_offset = (
        args.server_log.stat().st_size if args.mode == "graph-probe" else None
    )
    records = run_requests(
        samples=samples,
        concurrency=concurrency,
        base_url=f"http://127.0.0.1:{args.port}",
        model_name=args.model_name,
        output_tokens=output_tokens,
        timeout=args.timeout,
    )
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "dataset": args.dataset,
        "request_count": len(records),
        "concurrency": concurrency,
        "expected_output_tokens": output_tokens,
        "records": records,
    }
    if args.mode == "warmup":
        result["dataset_mode"] = args.dataset_mode
        result["dataset_path"] = str(args.dataset_path.resolve())
        if args.dataset_metadata is not None:
            metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
            result["dataset_seed"] = metadata.get("seed")
    if args.mode == "prefix-prime":
        result["expected_prefix_tokens"] = prefix_tokens
    if args.mode == "graph-probe":
        trace_events = read_trace_events_since(args.server_log, server_log_offset)
        probe_execution_events = [
            event
            for event in trace_events
            if event.get("event") == "inference_execution_mode_observed"
            and event.get("runner") == "mrv2"
            and event.get("pp_rank") == 0
            and event.get("tp_rank") == 0
        ]
        graph_events = [
            event
            for event in probe_execution_events
            if event.get("configured_cudagraph_mode") == "FULL_DECODE_ONLY"
            and event.get("actual_execution_mode") != "NONE"
            and event.get("need_eager") is False
        ]
        result["server_log_start_offset"] = server_log_offset
        result["execution_mode_events"] = probe_execution_events
        result["observed_execution_modes"] = sorted(
            {
                event.get("actual_execution_mode")
                for event in probe_execution_events
                if event.get("actual_execution_mode") is not None
            }
        )
        result["observed_graph"] = bool(graph_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.mode == "graph-probe" and not result["observed_graph"]:
        raise RuntimeError(
            "graph probe completed but did not observe a non-NONE execution mode"
        )
    print(
        f"CPP_PERFORMANCE_PREP_PASS mode={args.mode} "
        f"requests={len(records)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
