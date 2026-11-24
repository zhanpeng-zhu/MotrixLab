#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

run_marker=$(mktemp)
trap 'rm -f "$run_marker"' EXIT

source .venv/bin/activate
mxlab train \
  task=go2-walk-flat/rslrl.ppo \
  num_envs=4096 \
  seed=15 \
  algo.max_iterations=2000 \
  checkpoint.interval=100

run_dir=$(
  find runs/go2-walk-flat/rslrl/torch/ppo \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -newer "$run_marker" \
    -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)
test -n "$run_dir"

python tools/go2/posttrain_range_audit.py \
  "$run_dir" \
  go2-ground-frame-friction150-only \
  --start 800 \
  --end 1999 \
  --step 100 \
  --jobs 3 \
  --zero-hold \
  --speed-friction-matrix
