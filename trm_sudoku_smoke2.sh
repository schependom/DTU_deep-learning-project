#!/bin/bash
#BSUB -q gpuv100
#BSUB -J trm_sudoku_smoke
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o smoke_%J.out
#BSUB -e smoke_%J.err
#BSUB -B
#BSUB -N
#BSUB -u s204093@student.dtu.dk

# --- Load CUDA that matches your PyTorch build (cu126) ---
module load cuda/12.6


# Activate your venv
source /zhome/61/6/156072/venvs/trm/bin/activate

# Go to repo
cd /dtu/blackhole/08/156072/DTU_deep-learning-project

echo "=== Sanity: Python & Torch ==="
python - <<'PY'
import torch, sys, os
print("python:", sys.version.replace("\n"," "))
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name[0]:", torch.cuda.get_device_name(0))
PY

echo "=== Build a tiny Sudoku dataset (fast) ==="
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-smoke-50-aug-5 \
  --subsample-size 50 \
  --num-aug 5

echo "=== Run a tiny training smoke test (a few minutes) ==="
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-smoke-50-aug-5]" \
  evaluators="[]" \
  epochs=200 eval_interval=50 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=2 arch.L_cycles=2 \
  +run_name=smoke_sudoku ema=False

echo "=== Done ==="
