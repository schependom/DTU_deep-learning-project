"""
Script that can generate both action-based and state-to-state Tower of Hanoi datasets.
Refactored to match the file structure and metadata format of the Sudoku dataset.
"""

import os
import json
import numpy as np
import itertools
import argparse
from tqdm import tqdm

# Import metadata schema from common (assumed to be in the same directory)
from common import PuzzleDatasetMetadata

# --- Constants ---
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
# Action tokens: which disk moves (disk 0 = smallest = token 4, disk 1 = token 5, etc.)
ACTION_DISK_OFFSET = 4  # Disk 0 → token 4, Disk 1 → token 5, etc.


def get_vocab_size(max_disks, use_action_encoding):
    """
    Vocab for action encoding: 0=PAD, 1=PEG_A, 2=PEG_B, 3=PEG_C,
                                4=DISK_0, 5=DISK_1, ..., (3+max_disks)=DISK_(max_disks-1)
    Vocab for state encoding: 0=PAD, 1=PEG_A, 2=PEG_B, 3=PEG_C
    """
    if use_action_encoding:
        return 4 + max_disks
    else:
        return 4


def solve_hanoi_with_actions(n_disks, source, target, aux):
    """
    Generates optimal sequence of (state, action) pairs for Tower of Hanoi.

    Returns:
        states: List of numpy arrays where arr[i] is the peg of disk i
        actions: List of tuples (disk_id, destination_peg)
    """
    current_state = np.full(n_disks, source, dtype=np.uint8)
    states = [current_state.copy()]
    actions = []

    def move(disk, from_peg, to_peg, aux_peg):
        if disk == 0:
            # Move smallest disk
            current_state[0] = to_peg
            states.append(current_state.copy())
            actions.append((0, to_peg))
            return

        # Move disk-1 disks from source to auxiliary
        move(disk - 1, from_peg, aux_peg, to_peg)

        # Move current disk from source to target
        current_state[disk] = to_peg
        states.append(current_state.copy())
        actions.append((disk, to_peg))

        # Move disk-1 disks from auxiliary to target
        move(disk - 1, aux_peg, to_peg, from_peg)

    if n_disks > 0:
        move(n_disks - 1, source, target, aux)

    return states, actions


def encode_state_and_action(state, action, target_peg, max_disks, seq_len):
    """
    Encode state and next action into input/label format for TRM.
    """
    n_disks = len(state)

    # Input: current state + target
    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    input_vec[:n_disks] = state
    input_vec[-1] = target_peg

    # Label: action (disk to move + destination) + target
    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    if action is not None:
        disk_id, dest_peg = action
        # Encode disk as token (disk_0 → 4, disk_1 → 5, etc.)
        label_vec[0] = ACTION_DISK_OFFSET + disk_id
        label_vec[1] = dest_peg
    label_vec[-1] = target_peg

    return input_vec, label_vec


def encode_state_to_state(state_t, state_t1, target_peg, max_disks, seq_len):
    """
    State-to-state encoding: predict next state directly.
    """
    n_disks = len(state_t)

    # Input: current state + target
    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    input_vec[:n_disks] = state_t
    input_vec[-1] = target_peg

    # Label: next state + target
    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    label_vec[:n_disks] = state_t1
    label_vec[-1] = target_peg

    return input_vec, label_vec


def generate_dataset(
    output_dir="data/hanoi",
    encoding_type="action",
    train_min=3,
    train_max=6,
    test_min=7,
    test_max=10,
    seed=42,
):
    np.random.seed(seed)
    use_action_encoding = encoding_type == "action"

    # --- Configuration ---
    train_range = range(train_min, train_max + 1)
    test_range = range(test_min, test_max + 1)

    max_disks_global = test_max
    vocab_size = get_vocab_size(max_disks_global, use_action_encoding)

    if use_action_encoding:
        # Need space for: state (max_disks) + action (2) + target (1)
        seq_len = max_disks_global + 3
    else:
        # State encoding: state (max_disks) + target (1)
        seq_len = max_disks_global + 1

    print(f"\n{'=' * 60}")
    print(f"Tower of Hanoi Dataset Generation")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Encoding:    {encoding_type.upper()}")
    print(f"  Vocab Size:  {vocab_size}")
    print(f"  Seq Len:     {seq_len}")

    peg_ids = [PEG_A, PEG_B, PEG_C]
    peg_perms = list(itertools.permutations(peg_ids, 3))

    splits = {"train": train_range, "test": test_range}

    for split_name, disk_range in splits.items():
        save_dir = os.path.join(output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)

        # Initialize lists for data
        inputs_list = []
        labels_list = []
        
        # Initialize lists for indices (matching Sudoku structure)
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []

        global_puzzle_counter = 0 # Counts number of games (source->target)
        global_example_counter = 0 # Counts total steps/samples

        print(f"Generating {split_name.upper()} set...")

        for n_disks in disk_range:
            for source, target, aux in tqdm(
                peg_perms, desc=f"  N={n_disks} disks", leave=False
            ):
                # Generate optimal trajectory for this specific game
                states, actions = solve_hanoi_with_actions(n_disks, source, target, aux)

                # Iterate through steps in the trajectory
                steps_in_puzzle = 0
                for i in range(len(states) - 1):
                    current_state = states[i]
                    next_state = states[i + 1]
                    action = actions[i]

                    if use_action_encoding:
                        input_vec, label_vec = encode_state_and_action(
                            current_state, action, target, max_disks_global, seq_len
                        )
                    else:
                        input_vec, label_vec = encode_state_to_state(
                            current_state, next_state, target, max_disks_global, seq_len
                        )

                    inputs_list.append(input_vec)
                    labels_list.append(label_vec)
                    
                    # Identifier: We use 0 to match Sudoku 'blank', 
                    # or we could use n_disks to distinguish types. 
                    # Using 0 keeps it simple and consistent with common.py blank_identifier_id
                    puzzle_identifiers.append(0) 
                    
                    global_example_counter += 1
                    steps_in_puzzle += 1
                    puzzle_indices.append(global_example_counter)

                # End of one game (puzzle)
                if steps_in_puzzle > 0:
                    global_puzzle_counter += 1
                    group_indices.append(global_puzzle_counter)

        # Convert to numpy
        results = {
            "inputs": np.stack(inputs_list),
            "labels": np.stack(labels_list),
            "group_indices": np.array(group_indices, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
            "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32)
        }

        print(f"  ✓ Generated {results['inputs'].shape[0]:,} samples ({global_puzzle_counter} games) for {split_name}")

        # Create Metadata object
        metadata = PuzzleDatasetMetadata(
            pad_id=PAD_ID,
            ignore_label_id=PAD_ID, 
            blank_identifier_id=0,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_puzzle_identifiers=1, # Only one type (standard Hanoi)
            total_groups=global_puzzle_counter,
            mean_puzzle_examples=float(global_example_counter) / max(1, global_puzzle_counter),
            total_puzzles=global_puzzle_counter,
            sets=["all"]
        )

        # Save Metadata
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        # Save Arrays with proper naming convention (all__{name}.npy)
        for k, v in results.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save global identifiers mapping in root folder (matching Sudoku structure)
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<standard_hanoi>"], f)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}/")
    print(f"\nFiles created (for each split):")
    print(f"  ├── dataset.json")
    print(f"  ├── all__inputs.npy")
    print(f"  ├── all__labels.npy")
    print(f"  ├── all__group_indices.npy")
    print(f"  ├── all__puzzle_indices.npy")
    print(f"  └── all__puzzle_identifiers.npy")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tower of Hanoi dataset aligned with Sudoku file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/hanoi",
        help="Output directory (default: data/hanoi)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["action", "state"],
        default="action",
        help="Encoding type: 'action' or 'state' (default: action)",
    )
    parser.add_argument("--train-min", type=int, default=3)
    parser.add_argument("--train-max", type=int, default=6)
    parser.add_argument("--test-min", type=int, default=7)
    parser.add_argument("--test-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.out,
        encoding_type=args.encoding,
        train_min=args.train_min,
        train_max=args.train_max,
        test_min=args.test_min,
        test_max=args.test_max,
        seed=args.seed,
    )