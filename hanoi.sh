#!/bin/sh 

### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J trm_hanoi_optimized
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 6:00
# request x GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
#BSUB -u s251739@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

echo "Job started at:"
date

# Load environment variables
source .env

# Fail early if REPO is not set
if [ -z "${REPO}" ]; then
    echo "ERROR: REPO environment variable is not set. Check .env"
    exit 1
fi

# Check for WANDB_API_KEY
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set. W&B logging may fail."
fi

# Activate venv
echo "Activating virtual environment ..."
module load python3/3.10.12
module load cuda/12.6
echo "Loading virtual environment from ${REPO}/.venv ..."
source ${REPO}/.venv/bin/activate

# Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

# Get the hostname of the compute node
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 
export DATE_TAG=$(date +%Y%m%d_%H%M)

# --- Define parameters for W&B grouping ---
export WANDB_GROUP="hanoi_optimized_${DATE_TAG}"

# =================================================================
# ===          JOB 1: TRAIN ON ACTION-BASED ENCODING            ===
# =================================================================
echo ""
echo "################################################################"
echo "### [1/2] STARTING HANOI TRAINING (ACTION ENCODING)"
echo "################################################################"
echo ""

export RUN_NAME="hanoi_action_opt_${DATE_TAG}"
export DATA_PATH="src/data/hanoi_action"

echo "Making directory for run: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 ${REPO}/src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=8 \
evaluators="[{name: hanoi@HanoiEvaluator}]" \
epochs=30000 \
eval_interval=3000 \
min_eval_interval=5000 \
global_batch_size=128 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
lr_warmup_steps=2000 \
weight_decay=1.5 \
puzzle_emb_weight_decay=1.0 \
+run_name=${RUN_NAME} \
+wandb.group=${WANDB_GROUP} \
ema=True \
arch.mlp_t=False \
arch.pos_encodings=default

echo ""
echo "### [1/2] ACTION ENCODING training finished."
echo ""


# =================================================================
# ===        JOB 2: TRAIN ON STATE-TO-STATE ENCODING            ===
# =================================================================
echo ""
echo "################################################################"
echo "### [2/2] STARTING HANOI TRAINING (STATE ENCODING)"
echo "################################################################"
echo ""

export RUN_NAME="hanoi_state_opt_${DATE_TAG}"
export DATA_PATH="src/data/hanoi_state"

echo "Making directory for run: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 ${REPO}/src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=8 \
evaluators="[{name: hanoi@HanoiEvaluator}]" \
epochs=30000 \
eval_interval=3000 \
min_eval_interval=5000 \
global_batch_size=128 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
lr_warmup_steps=2000 \
weight_decay=1.5 \
puzzle_emb_weight_decay=1.0 \
+run_name=${RUN_NAME} \
+wandb.group=${WANDB_GROUP} \
ema=True \
arch.mlp_t=False \
arch.pos_encodings=default

echo ""
echo "### [2/2] STATE ENCODING training finished."
echo "################################################################"
echo ""
echo "Both training runs completed successfully!"
echo "Check W&B group: ${WANDB_GROUP}"
echo ""
echo "Job finished at:"
date
