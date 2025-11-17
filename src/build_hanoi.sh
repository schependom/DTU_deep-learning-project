#!/bin/bash
# Generate multiple Hanoi datasets for TRM experimentation

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Tower of Hanoi Dataset Generation for TRM           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. ACTION-BASED ENCODING (Recommended for TRM)
echo "  [1/2] Generating ACTION-BASED dataset (RECOMMENDED)..."
echo "         This encoding predicts which disk to move and where."
python dataset/build_hanoi_dataset.py \
    --encoding action \
    --out data/hanoi_action \
    --train-min 3 \
    --train-max 6 \
    --test-min 7 \
    --test-max 10 \
    --seed 42

echo "   ...Visualizing Action Dataset (Train Split)..."
python dataset/visualize_hanoi.py --dir data/hanoi_action/train

echo ""

# 2. STATE-TO-STATE ENCODING (Baseline comparison)
echo "  [2/2] Generating STATE-TO-STATE dataset (PROBABLY WORSE)..."
echo "         This encoding predicts the next complete state."
python dataset/build_hanoi_dataset.py \
    --encoding state \
    --out data/hanoi_state \
    --train-min 3 \
    --train-max 6 \
    --test-min 7 \
    --test-max 10 \
    --seed 42

echo "   ...Visualizing State Dataset (Train Split)..."
python dataset/visualize_hanoi.py --dir data/hanoi_state/train

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Generation Complete! ✓                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Datasets created:"
echo "   ├── data/hanoi_action/     RECOMMENDED:    Action prediction"
echo "   └── data/hanoi_state/      PROBABLY WORSE: State prediction"
echo ""