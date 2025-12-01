#!/bin/sh 

### General options
#BSUB -q gpua100
#BSUB -J trm_chess
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -u s214631@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

echo "Job started at:"
date

# Load environment
source .env
module load python3/3.10.12
module load cuda/12.6
source ${REPO}/.venv/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 
export DATE_TAG=$(date +%Y%m%d_%H%M)


# -------------------------------------------
# Chess Dataset Parameters
# -------------------------------------------

export CSV_PATH="dataset/chess_moves.csv"   # <--- path to your CSV
export OUT_NAME="chess"
export OUT_DIR="data/${OUT_NAME}"

export RUN_NAME="${OUT_NAME}_${DATE_TAG}"

echo "---------------------------------------------------------"
echo " CHESS TRM TRAINING RUN"
echo " Run Name: ${RUN_NAME}"
echo "---------------------------------------------------------"


# -------------------------------------------
# Dataset discovery / generation
# -------------------------------------------

if [ -d "${OUT_DIR}" ]; then
    export DATA_PATH="${OUT_DIR}"
    echo "Found existing dataset at ${DATA_PATH}"
else
    echo "Dataset not found. Generating at ${OUT_DIR}..."
    python src/dataset/build_chess_dataset.py \
        --csv "${CSV_PATH}" \
        --out "${OUT_DIR}" \
        --seed 42
    echo "Dataset generation complete."
    export DATA_PATH="${OUT_DIR}"
fi


# -------------------------------------------
# Create Run Directory
# -------------------------------------------

mkdir -p "${REPO}/checkpoints/${RUN_NAME}"


# -------------------------------------------
# TRM Architecture Notes for Chess
# -------------------------------------------
# - seq_len = 1 + 80 ≈ 81 tokens (similar to Sudoku)
# - vocab_size ~ FEN(≈40) + moves(4096)
# - Reasoning model should use transformer (mlp_t=False)
# - pos_encodings = rope or learned (Rope works well)
# - Evaluator: For now use "none" or a simple accuracy evaluator
# -------------------------------------------


# -------------------------------------------
# Start Training
# -------------------------------------------

echo "Starting Training Job: ${RUN_NAME}"
torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 ${REPO}/src/pretrain.py \
    arch=trm \
    data_paths="[${DATA_PATH}]" \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    evaluators="[{name: chess@ChessEvaluator}]" \
    epochs=500 \
    eval_interval=1 \
    min_eval_interval=0 \
    global_batch_size=128 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    lr_warmup_steps=1000 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +run_name=${RUN_NAME} \
    ema=True \
    arch.mlp_t=False \
    arch.pos_encodings=None

echo "Job finished at:"
date