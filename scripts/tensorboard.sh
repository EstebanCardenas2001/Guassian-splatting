#!/usr/bin/env bash
# Start TensorBoard over outputs/ in a detached tmux session.
# From your laptop:  ssh -L 6006:localhost:6006 <vm>  then open http://localhost:6006
# (VS Code Remote-SSH forwards the port automatically.)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=${1:-6006}
tmux has-session -t tb 2>/dev/null && { echo "tmux session 'tb' already running"; exit 0; }
tmux new-session -d -s tb \
  "source /mnt/miniforge3/etc/profile.d/conda.sh && conda activate gsplat && \
   tensorboard --logdir $REPO/outputs --host 127.0.0.1 --port $PORT"
echo "TensorBoard on 127.0.0.1:$PORT (tmux session 'tb')"
