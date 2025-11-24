#!/bin/bash
# Parameters
N_SIZE=6  # N=6 is challenging but solvable for backtrack in reasonable time
SAMPLES=2000 # 2000 unique base puzzles -> x8 aug = 16000 training examples
OUT_DIR="data/skyscrapers_${N_SIZE}"

echo "----------------------------------------------------------------"
echo "Generating Skyscrapers Dataset (N=${N_SIZE})"
echo "----------------------------------------------------------------"

python dataset/build_skyscrapers.py \
    --out "${OUT_DIR}" \
    --n "${N_SIZE}" \
    --samples "${SAMPLES}" \
    --seed 42

echo ""
echo "----------------------------------------------------------------"
echo "Visualizing Samples"
echo "----------------------------------------------------------------"
python dataset/visualize_skyscrapers.py --dir "${OUT_DIR}/train" --n 3