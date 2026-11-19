#!/bin/bash
#SBATCH --job-name=wo-bench-regression
#SBATCH --partition=preempted
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=wo-bench-%j.log

# Run the full regression suite on the Biohub HPC, from a checkout with `wo` on PATH (see benchmarks/README.md):
#
#   WAVEORDER_BENCH_OUTPUT=/path/to/runs sbatch benchmarks/run_regression.sbatch
#   sbatch benchmarks/run_regression.sbatch -o /path/to/runs
#
# Arguments are passed on to `wo bm run`. Runs land in -o, else in $WAVEORDER_BENCH_OUTPUT; one of the two is
# required.

set -euo pipefail

if [[ -z "${WAVEORDER_BENCH_OUTPUT:-}" && " $* " != *" -o "* && " $* " != *" --output-dir "* ]]; then
    echo "set WAVEORDER_BENCH_OUTPUT or pass -o OUTPUT_DIR" >&2
    exit 2
fi

wo bm run --scope all "$@"
