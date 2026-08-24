#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight AISBench accuracy report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main() -> int:
    args = parse_args()
    summaries = sorted(args.results_dir.rglob("summary_*.md"))
    csv_files = sorted(args.results_dir.rglob("summary_*.csv"))
    predictions = sorted(args.results_dir.rglob("predictions/**/*.json"))
    evaluated = sorted(args.results_dir.rglob("results/**/*.json"))

    lines = [
        f"# 精度测试报告：{args.run_id}",
        "",
        "## 运行信息",
        "",
        "~~~text",
        read_text(args.metadata) if args.metadata.is_file() else "metadata unavailable",
        "~~~",
        "",
        "## AISBench 汇总",
        "",
    ]
    if summaries:
        for summary in summaries:
            lines.extend((f"### {summary.name}", "", read_text(summary), ""))
    else:
        lines.extend(("未找到 Markdown 汇总文件。", ""))

    lines.extend((
        "## 结果文件",
        "",
        f"- AISBench 工作目录：`{args.results_dir}`",
        f"- CSV 汇总文件数：{len(csv_files)}",
        f"- 原始预测文件数：{len(predictions)}",
        f"- 逐题评估文件数：{len(evaluated)}",
        "",
    ))
    if csv_files:
        lines.append("### CSV 汇总")
        lines.append("")
        lines.extend(f"- `{path}`" for path in csv_files)
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT_WRITTEN {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
