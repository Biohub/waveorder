#!/bin/bash
# Two nodes x 1 GPU of the preempted partition, DDP over NCCL; with MODE=requeue, one forced requeue at step 22
# (mid-epoch 5) on the first attempt. Run it twice and compare: an uninterrupted run and a requeued one must end
# with identical weights, weight average, optimizer, scheduler and per-step batches.
#
#   REPO=/path/to/waveorder ROOT=/hpc/mydata/$USER/benchmarks/train-multinode    # ROOT on a shared filesystem
#   $REPO/.venv/bin/python $REPO/benchmarks/train/multinode/make_data.py $ROOT/data  # once, on a login node
#   cd $ROOT
#   sbatch --export=ALL,MODE=straight,REPO=$REPO $REPO/benchmarks/train/multinode/two_node_requeue.sbatch
#   sbatch --export=ALL,MODE=requeue,REPO=$REPO $REPO/benchmarks/train/multinode/two_node_requeue.sbatch
#   $REPO/.venv/bin/python $REPO/benchmarks/train/multinode/compare_runs.py $ROOT/straight $ROOT/requeue
#
# The log of a job is $ROOT/slurm-wo-train-2node-JOBID.out (the submit directory); a requeued job appends to it.
#SBATCH --job-name=wo-train-2node
#SBATCH --partition=preempted
#SBATCH --account=czb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # = GPUs per node; Lightning's SLURMEnvironment reads it
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=4            # 2 loader workers and the rank
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue                    # preemption and scontrol requeue put the job back under the same id
#SBATCH --signal=SIGUSR1@90          # the time limit: Lightning writes hpc_ckpt_N.ckpt and requeues itself
#SBATCH --open-mode=append           # keep the first attempt's log after the requeue
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${REPO:?pass REPO, the waveorder checkout, with --export}" "${MODE:?pass MODE=straight or MODE=requeue}"
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}
HERE=$REPO/benchmarks/train/multinode
export DATA=$ROOT/data RUN=$ROOT/$MODE                      # one run folder per mode, the same on every attempt
export PYTHONPATH=$HERE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
mkdir -p "$RUN/checkpoints"
envsubst < "$HERE/slurm.yml" > "$RUN/slurm.resolved.yml"
EXTRA=()
if [ "$MODE" = requeue ]; then
  EXTRA=(--trainer.callbacks+ callbacks.RequeueOnce --trainer.callbacks.at_step 22)
fi
# Resume from the newest checkpoint a stopped attempt left; none on the first attempt.
LAST=$(ls -t "$RUN"/checkpoints/last.ckpt "$RUN"/checkpoints/on_exception.ckpt "$RUN"/hpc_ckpt_*.ckpt 2>/dev/null \
  | head -1 || true)
echo "attempt ${SLURM_RESTART_COUNT:-0} on $SLURM_JOB_NODELIST, resuming from: ${LAST:-nothing}"
srun --kill-on-bad-exit=1 --cpu-bind=none "$REPO/.venv/bin/wo" distill fit \
  --config "$REPO/benchmarks/train/base.yml" --config "$RUN/slurm.resolved.yml" \
  --trainer.num_nodes "$SLURM_NNODES" --trainer.devices "$SLURM_NTASKS_PER_NODE" \
  ${LAST:+--ckpt_path "$LAST"} "${EXTRA[@]}"
