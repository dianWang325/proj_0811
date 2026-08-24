#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/30_run_accuracy.sh" --dataset all --num-prompts 10 --debug "$@"
