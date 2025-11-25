#!/bin/sh 

### General options
#BSUB -q gpua100
#BSUB -J trm_nqueens_unamb
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s251739@dtu.dk
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
export N_SIZE=12

# --- Training Configuration ---
# N=12 Board is 12x12 = 144 sequence length.
# Sudoku is 81. Maze is 900.
# 144 is small enough for mlp_t (Mixer).

export RUN_NAME="nqueens_unamb_${N_SIZE}_${DATE_TAG}"

# Check for data in root 'data/' (preferred) or 'src/data/' (legacy)
if [ -d "data/n_queens_unamb_${N_SIZE}" ]; then
    export DATA_PATH="data/n_queens_unamb_${N_SIZE}"
    echo "Found data at ${DATA_PATH}"
elif [ -d "src/data/n_queens_unamb_${N_SIZE}" ]; then
    export DATA_PATH="src/data/n_queens_unamb_${N_SIZE}"
    echo "Found data at ${DATA_PATH}"
else
    # Generate in root 'data/' if not found
    export DATA_PATH="data/n_queens_unamb_${N_SIZE}"
    echo "Data not found. Generating at ${DATA_PATH}..."
    python src/dataset/build_n_queens_unamb.py \
        --out "${DATA_PATH}" \
        --n "${N_SIZE}" \
        --aug 1 \
        --seed 42
    echo "Data generation complete."
fi

echo "Starting Training: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

# Note: 
# 1. arch.mlp_t=True : Preferred for fixed-size grid logic (like Sudoku).
# 2. arch.pos_encodings=none : Because mlp_t implicitly learns positions.
# 3. Evaluator : n_queens@NQueensEvaluator

# L_layers = z_L and z_H
# H_cycles = T (T-1 steps without gradient + 1 step with gradient)
# L_cycles = n = number of updates of z_L per T cycle
# halt_max_steps = N_sup = deep supervision steps = 16

# epoch = amount of times each example is seen during training
# nb of required steps = N * N * #masks * epochs

# eval interval is in steps, not epochs


torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 ${REPO}/src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=6 \
evaluators="[{name: n_queens@NQueensEvaluator}]" \
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
arch.pos_encodings=none  # or learned

echo "Job finished at:"
date