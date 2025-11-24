#!/bin/bash

# 1. Build the dataset
echo "Building Math Dataset..."
python dataset/build_math_dataset.py \
  --output-dir data/math-arithmetic \
  --train-size 100000 \
  --test-size 5000 \
  --seq-len 48

# 2. Visualize/Verify a few samples from the 'test' split
echo "Visualizing generated data..."
python dataset/visualize_math.py \
  --data-dir data/math-arithmetic \
  --split test \
  --num-samples 5