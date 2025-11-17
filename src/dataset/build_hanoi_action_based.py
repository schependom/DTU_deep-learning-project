"""
Action-based Tower of Hanoi dataset generation.
Given a current state, the model predicts which disk to move and where.
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
# Action tokens: which disk moves (disk 0 = smallest = token 4, disk 1 = token 5, etc.)
# Then destination peg (A=1, B=2, C=3)
ACTION_DISK_OFFSET = 4  # Disk 0 → token 4, Disk 1 → token 5, etc.


def get_vocab_size(max_disks):
    """
    Vocab: 0=PAD, 1=PEG_A, 2=PEG_B, 3=PEG_C,
           4=DISK_0, 5=DISK_1, ..., (3+max_disks)=DISK_(max_disks-1)
    """
    return 4 + max_disks


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

    Format:
        Input:  [disk_0_peg, disk_1_peg, ..., disk_N_peg, PAD..., target_peg]
        Label:  [disk_to_move, destination_peg, PAD..., target_peg]

    This way the model learns to:
    1. Look at current state
    2. Predict which disk to move and where (the action)
    3. Use target_peg as goal conditioning
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


def encode_state_to_state(state_t, state_t1, target_peg, seq_len):
    """
    Alternative encoding: predict next state directly (your original approach).
    Can be used for comparison.
    """
    n_disks = len(state_t)

    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    input_vec[:n_disks] = state_t
    input_vec[-1] = target_peg

    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    label_vec[:n_disks] = state_t1
    label_vec[-1] = target_peg

    return input_vec, label_vec


def generate_dataset(output_dir="data/hanoi_trm_v2", seed=42, use_action_encoding=True):
    """
    Generate Hanoi dataset with proper action encoding for TRM.

    Args:
        output_dir: Where to save the dataset
        seed: Random seed
        use_action_encoding: If True, use action-based encoding (recommended)
                            If False, use state-to-state encoding (original)
    """
    np.random.seed(seed)

    # --- Configuration ---
    train_range = range(3, 7)  # 3 to 6 disks
    test_range = range(7, 11)  # 7 to 10 disks

    max_disks_global = max(test_range)
    vocab_size = get_vocab_size(max_disks_global)

    if use_action_encoding:
        # Need space for: state (max_disks) + action (2) + target (1)
        seq_len = max_disks_global + 3
    else:
        # Original: state (max_disks) + target (1)
        seq_len = max_disks_global + 1

    print(f"Configuration:")
    print(f"  Train Disks: {list(train_range)}")
    print(f"  Test Disks:  {list(test_range)}")
    print(f"  Seq Len:     {seq_len}")
    print(f"  Vocab Size:  {vocab_size}")
    print(
        f"  Encoding:    {'Action-based' if use_action_encoding else 'State-to-State'}"
    )

    peg_ids = [PEG_A, PEG_B, PEG_C]
    peg_perms = list(itertools.permutations(peg_ids, 3))

    splits = {"train": train_range, "test": test_range}

    for split_name, disk_range in splits.items():
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        inputs_list = []
        labels_list = []

        print(f"\nGenerating {split_name.upper()} set...")

        for n_disks in disk_range:
            for source, target, aux in tqdm(
                peg_perms, desc=f"N={n_disks} Permutations"
            ):
                # Generate optimal trajectory with actions
                states, actions = solve_hanoi_with_actions(n_disks, source, target, aux)

                # Create training pairs
                for i in range(len(states) - 1):
                    current_state = states[i]
                    next_state = states[i + 1]
                    action = actions[i]

                    if use_action_encoding:
                        # Learn to predict action
                        input_vec, label_vec = encode_state_and_action(
                            current_state, action, target, max_disks_global, seq_len
                        )
                    else:
                        # Learn to predict next state
                        input_vec, label_vec = encode_state_to_state(
                            current_state, next_state, target, seq_len
                        )

                    inputs_list.append(input_vec)
                    labels_list.append(label_vec)

        # Convert to numpy
        inputs_np = np.stack(inputs_list)
        labels_np = np.stack(labels_list)

        print(f"  Saved {inputs_np.shape[0]} samples for {split_name}")

        # Save arrays
        np.save(os.path.join(output_dir, split_name, "inputs.npy"), inputs_np)
        np.save(os.path.join(output_dir, split_name, "labels.npy"), labels_np)

        # Save metadata
        metadata = {
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "pad_id": int(PAD_ID),
            "n_samples": len(inputs_list),
            "disk_range": list(disk_range),
            "max_disks": int(max_disks_global),
            "encoding_type": "action" if use_action_encoding else "state",
            "description": (
                "Action-based: Input=[state, target], Label=[disk_to_move, dest_peg, target]"
                if use_action_encoding
                else "State-based: Input=[state_t, target], Label=[state_t+1, target]"
            ),
        }

        with open(os.path.join(output_dir, split_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("\nDataset generation complete.")
    print(f"\nTo use this dataset:")
    print(f"  1. Model learns to predict actions from states")
    print(f"  2. During inference, apply predicted actions to update state")
    print(f"  3. Recurse until reaching goal state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tower of Hanoi dataset for TRM with proper action encoding"
    )
    parser.add_argument(
        "--out", type=str, default="data/hanoi_trm_v2", help="Output directory"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["action", "state"],
        default="action",
        help="Use action-based encoding (recommended) or state-to-state",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.out,
        seed=args.seed,
        use_action_encoding=(args.encoding == "action"),
    )
