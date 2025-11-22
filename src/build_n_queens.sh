#!/bin/bash

# Parameters
N_SIZE=8  # N=12 gives 14,200 solutions. N=8 only has 92.
OUT_DIR="data/n_queens_${N_SIZE}"

echo "----------------------------------------------------------------"
echo "Generating N-Queens Dataset (N=${N_SIZE})"
echo "Target Directory: ${OUT_DIR}"
echo "----------------------------------------------------------------"

# Run generation
# Using --aug 1 enables 8x dihedral symmetries for training data
python dataset/build_n_queens_dataset.py \
    --out "${OUT_DIR}" \
    --n "${N_SIZE}" \
    --aug 1 \
    --seed 42

echo ""
echo "----------------------------------------------------------------"
echo "Visualizing Train Data Samples"
echo "----------------------------------------------------------------"
python dataset/visualize_n_queens.py --dir "${OUT_DIR}/train" --n 3

echo ""
echo "Dataset Ready."