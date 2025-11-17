"""
Predict the next state given the current state in Tower of Hanoi (alternative implementation).
"""

import os
import json
import numpy as np
import itertools
import argparse
from tqdm import tqdm

# --- Constants ---
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
# Vocab: 0=PAD, 1=A, 2=B, 3=C
VOCAB_SIZE = 4


def solve_hanoi_states(n_disks, source, target, aux):
    """
    Generates the optimal sequence of states for Tower of Hanoi.
    Returns a list of numpy arrays where arr[i] is the peg of disk i.
    Disk 0 is the smallest.
    """
    # State: index = disk size (0=smallest), value = peg ID
    current_state = np.full(n_disks, source, dtype=np.uint8)
    history = [current_state.copy()]

    def move(n, s, t, a):
        if n == 0:
            # Move disk 0 (smallest)
            current_state[0] = t
            history.append(current_state.copy())
            return

        # Move n-1 from Source to Aux
        move(n - 1, s, a, t)

        # Move disk n from Source to Target
        current_state[n] = t
        history.append(current_state.copy())

        # Move n-1 from Aux to Target
        move(n - 1, a, t, s)

    # The recursive function usually deals with disk indices 1..N or 0..N-1
    # Here we pass n_disks-1 as the largest disk index
    if n_disks > 0:
        move(n_disks - 1, source, target, aux)

    return history


def generate_dataset(output_dir="data/hanoi_trm", seed=42):
    np.random.seed(seed)

    # --- Configuration ---
    # Train on small/medium complexity
    train_range = range(3, 7)  # 3 to 6 disks
    # Test on higher complexity (OOD generalization)
    test_range = range(7, 11)  # 7 to 10 disks

    # Max sequence length = Max disks + 1 (for the Target Token)
    max_disks_global = max(test_range)
    seq_len = max_disks_global + 1

    print(f"Configuration:")
    print(f"  Train Disks: {list(train_range)}")
    print(f"  Test Disks:  {list(test_range)}")
    print(f"  Seq Len:     {seq_len} (Disks + TargetToken)")
    print(f"  Vocab Size:  {VOCAB_SIZE} (PAD + 3 Pegs)")

    peg_ids = [PEG_A, PEG_B, PEG_C]
    peg_perms = list(itertools.permutations(peg_ids, 3))  # 6 permutations

    splits = {"train": train_range, "test": test_range}

    for split_name, disk_range in splits.items():
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        inputs_list = []
        labels_list = []

        print(f"\nGenerating {split_name.upper()} set...")

        for n_disks in disk_range:
            # Augmentation: Iterate through all Source/Target/Aux permutations
            # This prevents the model from memorizing "Always move to C"
            for s, t, a in tqdm(peg_perms, desc=f"N={n_disks} Permutations"):
                # 1. Generate full optimal trajectory
                states = solve_hanoi_states(n_disks, s, t, a)

                # 2. Create (Input, Label) pairs
                # Input:  [State_t,   Target_Peg]
                # Label:  [State_t+1, Target_Peg]
                for i in range(len(states) - 1):
                    current_state = states[i]
                    next_state = states[i + 1]

                    # --- Formatting ---
                    # Initialize with PAD
                    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
                    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)

                    # Fill disk positions (0 to n_disks-1)
                    input_vec[:n_disks] = current_state
                    label_vec[:n_disks] = next_state

                    # Fill Target Token at the very end
                    # This serves as the "Goal" conditioning
                    input_vec[-1] = t
                    label_vec[-1] = t

                    inputs_list.append(input_vec)
                    labels_list.append(label_vec)

        # Convert to numpy
        inputs_np = np.stack(inputs_list)
        labels_np = np.stack(labels_list)

        print(f"  Saved {inputs_np.shape[0]} samples for {split_name}")

        # Save .npy files
        np.save(os.path.join(output_dir, split_name, "inputs.npy"), inputs_np)
        np.save(os.path.join(output_dir, split_name, "labels.npy"), labels_np)

        # Save Metadata (Compatible with TRM logic)
        metadata = {
            "seq_len": seq_len,
            "vocab_size": VOCAB_SIZE,
            "pad_id": PAD_ID,
            "n_samples": len(inputs_list),
            "disk_range": list(disk_range),
            "description": "State-to-NextState pairs for TRM. Format: [d0...dN, PAD..., TARGET]",
        }
        with open(os.path.join(output_dir, split_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="data/hanoi_trm", help="Output directory"
    )
    args = parser.parse_args()

    generate_dataset(args.out)
