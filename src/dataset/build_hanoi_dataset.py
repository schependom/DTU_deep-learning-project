"""
Script that can generate both action-based and state-to-state Tower of Hanoi datasets.
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


def encode_state_to_state(state_t, state_t1, target_peg, max_disks, seq_len):
    """
    State-to-state encoding: predict next state directly.

    Format:
        Input:  [disk_0_peg, disk_1_peg, ..., disk_N_peg, PAD..., target_peg]
        Label:  [next_disk_0_peg, next_disk_1_peg, ..., PAD..., target_peg]
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
    """
    Generate Hanoi dataset with configurable encoding for TRM.

    Args:
        output_dir: Where to save the dataset
        encoding_type: "action" or "state" encoding
        train_min, train_max: Training disk range (inclusive)
        test_min, test_max: Test disk range (inclusive)
        seed: Random seed
    """
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
    print(f"  Train Disks: {list(train_range)}")
    print(f"  Test Disks:  {list(test_range)}")
    print(f"  Seq Len:     {seq_len}")
    print(f"  Vocab Size:  {vocab_size}")

    if use_action_encoding:
        print(f"\n  Format:")
        print(f"    Input:  [state, target_peg]")
        print(f"    Label:  [disk_to_move, dest_peg, target_peg]")
    else:
        print(f"\n  Format:")
        print(f"    Input:  [state_t, target_peg]")
        print(f"    Label:  [state_t+1, target_peg]")
    print(f"{'=' * 60}\n")

    peg_ids = [PEG_A, PEG_B, PEG_C]
    peg_perms = list(itertools.permutations(peg_ids, 3))

    splits = {"train": train_range, "test": test_range}

    for split_name, disk_range in splits.items():
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        inputs_list = []
        labels_list = []

        print(f"Generating {split_name.upper()} set...")

        for n_disks in disk_range:
            for source, target, aux in tqdm(
                peg_perms, desc=f"  N={n_disks} disks", leave=False
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
                            current_state, next_state, target, max_disks_global, seq_len
                        )

                    inputs_list.append(input_vec)
                    labels_list.append(label_vec)

        # Convert to numpy
        inputs_np = np.stack(inputs_list)
        labels_np = np.stack(labels_list)

        print(f"  ✓ Generated {inputs_np.shape[0]:,} samples for {split_name}")

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
            "encoding_type": encoding_type,
            "description": (
                "Action-based: Input=[state, target], Label=[disk_to_move, dest_peg, target]"
                if use_action_encoding
                else "State-based: Input=[state_t, target], Label=[state_t+1, target]"
            ),
        }

        with open(os.path.join(output_dir, split_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  ├── train/")
    print(f"  │   ├── inputs.npy")
    print(f"  │   ├── labels.npy")
    print(f"  │   └── metadata.json")
    print(f"  └── test/")
    print(f"      ├── inputs.npy")
    print(f"      ├── labels.npy")
    print(f"      └── metadata.json")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tower of Hanoi dataset for TRM with configurable encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Generate action-based dataset (recommended for TRM)
        python build_hanoi_dataset.py --encoding action --out data/hanoi_action
        
        # Generate state-to-state dataset (baseline comparison)
        python build_hanoi_dataset.py --encoding state --out data/hanoi_state
        
        # Custom disk ranges
        python build_hanoi_dataset.py --encoding action --train-min 2 --train-max 5 --test-min 6 --test-max 8
        """,
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
        help="Encoding type: 'action' (predict moves) or 'state' (predict next state) (default: action)",
    )
    parser.add_argument(
        "--train-min",
        type=int,
        default=3,
        help="Minimum number of disks for training (default: 3)",
    )
    parser.add_argument(
        "--train-max",
        type=int,
        default=6,
        help="Maximum number of disks for training (default: 6)",
    )
    parser.add_argument(
        "--test-min",
        type=int,
        default=7,
        help="Minimum number of disks for testing (default: 7)",
    )
    parser.add_argument(
        "--test-max",
        type=int,
        default=10,
        help="Maximum number of disks for testing (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

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
