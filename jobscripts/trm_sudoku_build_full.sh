#!/bin/bash
#BSUB -J sudoku_data
#BSUB -q gpuv100                 # GPU queue
#BSUB -W 01:00               # plenty for 1k+aug build
#BSUB -n 4                   # a few CPU cores
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"   # memory request
#BSUB -M 8GB                # hard cap (job killed if exceeded)
#BSUB -o sudoku_data_%J.out
#BSUB -e sudoku_data_%J.err
#BSUB -B
#BSUB -N
#BSUB -u s204093@student.dtu.dk

# --- Activate your venv (what you used for the smoke test) ---
# If you created a python venv under your zhome, e.g.:
source /zhome/61/6/156072/venvs/trm/bin/activate

# --- Paths ---
REPO=/dtu/blackhole/08/156072/DTU_deep-learning-project   # repo root (adjust if different)
OUTDIR=/dtu/blackhole/08/156072/DTU_deep-learning-project/data/sudoku-extreme-1k-aug-1000

mkdir -p "$OUTDIR"
cd "$REPO"

echo "Building Sudoku dataset into: $OUTDIR"
python dataset/build_sudoku_dataset.py \
  --output-dir "$OUTDIR" \
  --subsample-size 1000 \
  --num-aug 1000

echo "Done. Listing output:"
ls -lah "$OUTDIR"
