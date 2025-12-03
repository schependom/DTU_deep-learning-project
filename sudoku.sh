#!/bin/sh 

### LSF options
#BSUB -q gpua100
#BSUB -J trm_sudoku_quick
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -u s204093@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o sudoku_quick_%J.out
#BSUB -e sudoku_quick_%J.err

###############
# Load REPO path
###############
source .env

if [ -z "${REPO}" ]; then
    echo "ERROR: REPO not set in .env"
    exit 1
fi

echo "Using REPO=${REPO}"

###############
# Activate env
###############
module load python3/3.10.12
module load cuda/12.6

source "${REPO}/.venv/bin/activate"
cd "${REPO}"

###############
# Dataset
###############
DATA_DIR="${REPO}/src/data/sudoku"

if [ ! -d "${DATA_DIR}" ]; then
    echo "Sudoku dataset not found — building it..."
    python src/dataset/build_sudoku_dataset.py \
        --output-dir src/data/sudoku \
        --subsample-size 1000 --num-aug 10
else
    echo "Sudoku dataset found at ${DATA_DIR}"
fi

###############
# Quick TRAINING RUN
###############
echo "Starting quick Sudoku test run..."

date_tag=$(date +%Y%m%d_%H%M)
run_name="pretrain_sudoku_quick_${date_tag}"

python src/pretrain.py \
    arch=trm \
    data_paths="[src/data/sudoku]" \
    evaluators="[]" \
    epochs=2000 eval_interval=500 \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=6 \
    +run_name=${run_name} ema=True

echo "Job finished at:"
date
