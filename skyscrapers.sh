#!/bin/sh 
#BSUB -q gpua100
#BSUB -J trm_skyscrapers
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s251739@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

source .env
module load python3/3.10.12
module load cuda/12.6
source ${REPO}/.venv/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 
export DATE_TAG=$(date +%Y%m%d_%H%M)
export N_SIZE=6

export RUN_NAME="skyscrapers_${N_SIZE}_${DATE_TAG}"
export DATA_PATH="src/data/skyscrapers_${N_SIZE}"

echo "Starting Training: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

# Config Optimized for Skyscrapers N=6
# N=6 + 2 padding = 8x8 grid = 64 sequence length.
# This is small, so we can use larger L_cycles or larger batch size easily.

torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 ${REPO}/src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=6 \
evaluators="[{name: n_queens@NQueensEvaluator}]" \
epochs=10000 \
eval_interval=50 \
min_eval_interval=0 \
global_batch_size=512 \
lr=2e-4 \
puzzle_emb_lr=2e-4 \
lr_warmup_steps=2000 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
+run_name=${RUN_NAME} \
ema=True \
arch.mlp_t=True \
arch.pos_encodings=none