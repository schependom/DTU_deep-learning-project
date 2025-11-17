#!/bin/sh 

### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J trm_hanoi_compare
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
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

# Now, load environment variables (it will find .env in the current directory)
source .env

# Fail early if REPO is not set
if [ -z "${REPO}" ]; then
    echo "ERROR: REPO environment variable is not set. Check .env"
    exit 1
fi

# New check for WANDB_API_KEY
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
# Pick a random high port for communication
export MASTER_PORT=29500 
export DATE_TAG=$(date +%Y%m%d_%H%M)

# --- Define parameters for W&B grouping ---
# Both runs will be in the same group for easy comparison
export WANDB_GROUP="hanoi_compare_${DATE_TAG}"

# =================================================================
# ===          JOB 1: TRAIN ON ACTION-BASED ENCODING            ===
# =================================================================
echo ""
echo "################################################################"
echo "### [1/2] STARTING HANOI TRAINING (ACTION ENCODING)"
echo "################################################################"
echo ""

export RUN_NAME="hanoi_action_${DATE_TAG}"
export DATA_PATH="src/data/hanoi_action"

echo "Making directory for run: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

echo "Starting Action training (MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT})"
echo "  Run Name: ${RUN_NAME}"
echo "  Data Path: ${DATA_PATH}"
echo "  W&B Group: ${WANDB_GROUP}"

# Run the training, torchrun will pick up the env vars
torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=4 \
arch.L_cycles=6 \
evaluators=[] \
+run_name=${RUN_NAME} \
++global_batch_size=32 \
+wandb.group=${WANDB_GROUP} \
ema=True

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

export RUN_NAME="hanoi_state_${DATE_TAG}"
export DATA_PATH="src/data/hanoi_state"

echo "Making directory for run: ${RUN_NAME}"
mkdir -p "${REPO}/checkpoints/${RUN_NAME}"

echo "Starting State training (MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT})"
echo "  Run Name: ${RUN_NAME}"
echo "  Data Path: ${DATA_PATH}"
echo "  W&B Group: ${WANDB_GROUP}"

# Run the training
torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 src/pretrain.py \
arch=trm \
data_paths="[${DATA_PATH}]" \
arch.L_layers=2 \
arch.H_cycles=4 \
arch.L_cycles=6 \
evaluators=[] \
++global_batch_size=32 \
+run_name=${RUN_NAME} \
+wandb.group=${WANDB_GROUP} \
ema=True

echo ""
echo "### [2/2] STATE ENCODING training finished."
echo "################################################################"


echo ""
echo "Job finished at :"
date