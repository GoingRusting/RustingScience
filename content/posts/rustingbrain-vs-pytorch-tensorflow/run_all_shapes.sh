#!/usr/bin/env bash
# Runs each (batch, seq) shape as its own process for tf_bench_300m.py and
# torch_bench_300m.py. TF's XLA allocator hard-crashes the whole process
# (SIGABRT, not a catchable OOM) on shapes it can't fit -- per-shape
# subprocesses mean one crash doesn't cost the rest of the sweep.
set -uo pipefail
cd "$(dirname "$0")/.."

SHAPES=("1 512" "1 1024" "4 512" "4 1024" "8 512")

echo "=== TensorFlow ==="
LD_LIBRARY_PATH=$(find .venv-tf/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH
for shape in "${SHAPES[@]}"; do
  read -r batch seq <<< "$shape"
  BENCH_BATCH=$batch BENCH_SEQ=$seq .venv-tf/bin/python scripts/tf_bench_300m.py 2>&1 \
    | grep -v "^I0000\|^WARNING\|^W0000\|^E0000\|To enable" \
    | (grep -E "^\s*[0-9]+\s+[0-9]+\s" || echo "$batch $seq  FAILED (OOM/crash)")
done

echo
echo "=== PyTorch ==="
for shape in "${SHAPES[@]}"; do
  read -r batch seq <<< "$shape"
  BENCH_BATCH=$batch BENCH_SEQ=$seq .venv-torch/bin/python scripts/torch_bench_300m.py 2>&1 \
    | (grep -E "^\s*[0-9]+\s+[0-9]+\s" || echo "$batch $seq  FAILED (OOM/crash)")
done

echo
echo "=== RustingBrain ==="
for shape in "${SHAPES[@]}"; do
  read -r batch seq <<< "$shape"
  echo "--- batch=$batch seq=$seq ---"
  ./target/release/bench --gpu --mixed-precision \
    --d-model 768 --n-layers 16 --n-heads 12 --n-kv-heads 4 --head-dim 64 \
    --d-ff 1408 --seq-len "$seq" --batch-size "$batch" \
    --steps 10 --gpu-memory-budget-mib 9000 \
    || echo "$batch $seq  FAILED (OOM/crash)"
done
