# Tiny Recursive Models (TRMs)

## Setup (local, CPU friendly)

### Python Environment

Create a new conda environment:

```bash
conda create -n trm python=3.10 -y
```

Activate the environment:

```bash
conda activate trm
```

Install the required packages:

```bash
pip install --no-cache-dir -r requirements-macos.txt
```

## Datasets

### Hierarchy

How does the `PuzzleDataset` class work?
The dataset has a 3-level hierarchy:

-   Examples (lowest level): Individual training samples (input-output pairs)
-   Puzzles (middle level): Collections of examples from the same puzzle
-   Groups (highest level): Collections of puzzles that should be sampled together

For example:

```txt
Group 0:
  ├─ Puzzle 0: [Example 0, Example 1, Example 2]
  ├─ Puzzle 1: [Example 3, Example 4]
  └─ Puzzle 2: [Example 5]
Group 1:
  ├─ Puzzle 3: [Example 6, Example 7, Example 8, Example 9]
  └─ Puzzle 4: [Example 10]
```

### File Structure

When you generate a dataset, you get these files:

```txt
data/hanoi/
├── train/
│   ├── dataset.json                    # Metadata
│   ├── all__inputs.npy                 # Input sequences [num_examples, seq_len]
│   ├── all__labels.npy                 # Target sequences [num_examples, seq_len]
│   ├── all__puzzle_identifiers.npy     # Which puzzle each belongs to [num_puzzles]
│   ├── all__puzzle_indices.npy         # Start index of each puzzle [num_puzzles+1]
│   └── all__group_indices.npy          # Start index of each group [num_groups+1]
└── test/
    └── (same structure)
```

How the indices work:

-   E.g. `puzzle_indices = [0, 3, 5, 6, 10]` means:
    -   Puzzle 0 has examples from index 0 to 2 (indices 0, 1, 2 -> 3 examples)
    -   Puzzle 1 has examples from index 3 to 4 (indices 3, 4 -> 2 examples)
    -   Puzzle 2 has examples from index 5 to 5 (index 5 -> 1 example)
    -   Puzzle 3 has examples from index 6 to 9 (indices 6, 7, 8, 9 -> 4 examples)
-   E.g. `group_indices = [0, 3, 6]` means:
    -   Group 0 has puzzles from index 0 to 2 (3 puzzles: 0, 1, 2)
    -   Group 1 has puzzles from index 3 to 5 (3 puzzles: 3, 4, 5)

### Training

In training Mode, `test_set_mode=False`.

The `_iter_train` function does this:

1. **Shuffle groups** randomly each epoch
2. For each group, **randomly pick one puzzle** from that group
3. From that puzzle, **randomly sample examples** without replacement
4. Pack examples into batches of size `global_batch_size`
5. Distribute across GPUs (each GPU gets `local_batch_size` examples)

### Testing

In testing Mode, `test_set_mode=True`.

The `_iter_test` function does this:

1. Go through **all examples sequentially** (no shuffling)
2. Pack into batches
3. Distribute across GPUs

This ensures we evaluate on every single example exactly once.
