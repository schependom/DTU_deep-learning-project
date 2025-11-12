#!/bin/sh 

### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J trm_hanoi_train
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


echo "making directory for run"
date=$(date +%Y%m%d_%H%M)
mkdir -p "${REPO}/checkpoints/${date}"


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

# run training
echo "Starting training"

# Get the hostname of the compute node
# This is often needed for distributed training
export MASTER_ADDR=$(hostname)
# Pick a random high port for communication
export MASTER_PORT=29500 

echo "Starting training (MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT})"

# Run the training, torchrun will pick up the env vars
# ! Important: change the number of GPU's to match the requested number above (BSUB -gpu "num=X")
torchrun --nproc-per-node 1 --rdzv_backend=c10d --nnodes=1 src/pretrain.py \
arch=trm \
data_paths="[src/data/hanoi]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

# Where 
#       arch=trm:       use the TRM architecture
#       data_paths:     path to the hanoi dataset (relative to DTU_deep-learning-project/)
#       arch.L_layers:  number of L layers
#       arch.H_cycles:  number of H cycles
#       arch.L_cycles:  number of L cycles
#       +run_name:      name of the run (used for checkpoint saving and wandb)
#       ema=True:       use exponential moving average of model weights
# and
#       torchrun:           launch script for distributed training
#       --nproc-per-node:   number of processes to launch (should match number of GPUs)
#       --rdzv_backend:     backend for rendezvous (c10d is default)
#       --rdzv_endpoint:    endpoint for rendezvous (localhost:0 means auto)
#       --nnodes:           number of nodes. A node is a physical machine. Here we use 1 node.

# rendezvous is needed for multi-GPU training to synchronize the processes.



echo "Job finished at :"
date