#!/bin/bash

# Assumes the csv.

# Parameters
CSV_PATH="dataset/chess_200k.csv"  
OUT_DIR="data/chess"            
SEED=42

echo "----------------------------------------------------------------"
echo "Generating Chess TRM Dataset"
echo "Input CSV:      ${CSV_PATH}"
echo "Output Folder:  ${OUT_DIR}"
echo "----------------------------------------------------------------"

# Generate
python dataset/build_chess_dataset.py \
    --csv "${CSV_PATH}" \
    --out "${OUT_DIR}" \
    --seed "${SEED}"


echo ""
echo "Dataset Ready."
