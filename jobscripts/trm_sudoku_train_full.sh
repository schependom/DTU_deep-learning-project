#!/bin/bash
#BSUB -J sudoku_att_full
#BSUB -q gpuv100                          # <-- change to your GPU queue
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"              # system RAM request
#BSUB -M 32GB                            # hard cap
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o sudoku_att_%J.out
#BSUB -e sudoku_att_%J.err
#BSUB -B
#BSUB -N
#BSUB -u s204093@student.dtu.dk

# --- Activate your venv ---
source /zhome/61/6/156072/venvs/trm/bin/activate

# --- Optional perf/env knobs ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
# export WANDB_MODE=offline   # uncomment if you don't want W&B network traffic

# --- Paths ---
REPO=/dtu/blackhole/08/156072/DTU_deep-learning-project
DATADIR=/dtu/blackhole/08/156072/DTU_deep-learning-project/data/sudoku-extreme-1k-aug-1000

cd "$REPO"

# Quick guard: fail fast if data isn't there
if [ ! -d "$DATADIR" ]; then
  echo "ERROR: Dataset folder not found: $DATADIR"
  echo "Build it first with build_sudoku_data.sh"
  exit 2
fi

run_name="pretrain_att_sudoku_full_$(date +%Y%m%d_%H%M%S)"

python pretrain.py \
  arch=trm \
  data_paths="[$DATADIR]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True
