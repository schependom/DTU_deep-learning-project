#!/bin/sh 

### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J trm_maze_quick
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### request x GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
#BSUB -u s204093@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o maze_quick_%J.out
#BSUB -e maze_quick_%J.err
# -- end of LSF options --

# Load environment variables from .env in the project root (must define REPO)
source .env

if [ -z "${REPO}" ]; then
    echo "ERROR: REPO environment variable is not set. Check .env"
    exit 1
fi

# Optional: warn if WANDB_API_KEY is missing
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set. W&B logging may fail."
fi

echo "Using REPO=${REPO}"

# Timestamp for this run
date_tag=$(date +%Y%m%d_%H%M)
mkdir -p "${REPO}/checkpoints/${date_tag}"

############################
# Activate environment
############################
echo "Loading modules and activating virtual environment ..."
module load python3/3.10.12
module load cuda/12.6

echo "Activating venv at ${REPO}/.venv ..."
source "${REPO}/.venv/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

cd "${REPO}"

############################
# Build Maze dataset if needed
############################
DATA_DIR="${REPO}/src/data/maze-30x30-hard-1k"

if [ ! -d "${DATA_DIR}" ]; then
    echo "Maze dataset not found at ${DATA_DIR}"
    echo "Building Maze dataset now..."
    python src/dataset/build_maze_dataset.py \
        --output-dir src/data/maze-30x30-hard-1k
    echo "Maze dataset build finished."
else
    echo "Maze dataset already present at ${DATA_DIR}"
fi

############################
# Quick training run
############################

echo "Starting quick Maze training run..."

run_name="pretrain_att_maze_quick_${date_tag}"

# This is a *short* run:
# - epochs=1000 instead of 50000
# - eval_interval=200 instead of 5000
# You can roughly estimate full runtime by:
#   full_time ≈ (50000 / 1000) * this_job_runtime = 50 × this_job_runtime

python src/pretrain.py \
    arch=trm \
    data_paths="[src/data/maze-30x30-hard-1k]" \
    evaluators="[]" \
    epochs=1000 eval_interval=200 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name=${run_name} ema=True

echo "Job finished at:"
date
