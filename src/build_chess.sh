#!/bin/bash

# Parameters
CSV_PATH="dataset/chess_moves.csv"   # <-- path to your CSV file
OUT_DIR="data/chess"            # <-- output directory
SEED=42

echo "----------------------------------------------------------------"
echo "Generating Chess TRM Dataset"
echo "Input CSV:      ${CSV_PATH}"
echo "Output Folder:  ${OUT_DIR}"
echo "----------------------------------------------------------------"

# Run dataset generation
python dataset/build_chess_dataset.py \
    --csv "${CSV_PATH}" \
    --out "${OUT_DIR}" \
    --seed "${SEED}"

echo ""
echo "----------------------------------------------------------------"
echo "Visualizing Train Data Samples (first 3)"
echo "----------------------------------------------------------------"

# OPTIONAL: if you want, I can write a visualization script similar to visualize_n_queens.py
# For now, this just prints sample entries

# python - << 'EOF'
# import numpy as np
# import os

# train_dir = "${OUT_DIR}/train"
# inputs = np.load(os.path.join(train_dir, "all__inputs.npy"))
# labels = np.load(os.path.join(train_dir, "all__labels.npy"))

# print("First 3 training samples:")
# for i in range(min(3, len(inputs))):
#     print(f"\nSample {i}:")
#     print("Inputs:", inputs[i])
#     print("Labels:", labels[i])
# EOF

echo ""
echo "Dataset Ready."
