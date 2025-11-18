"""
Fixed Tower of Hanoi dataset generation.
Key fixes:
1. Correct puzzle_indices structure (each step is a separate puzzle)
2. Proper sequence length (just enough for one step's I/O)
3. Match Sudoku's single-example-per-puzzle pattern
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
ACTION_DISK_OFFSET = 4


def get_vocab_size(max_disks, use_action_encoding):
    if use_action_encoding:
        return 4 + max_disks  # PAD + 3 pegs + disk actions
    else:
        return 4  # PAD + 3 pegs


def solve_hanoi_with_actions(n_disks, source, target, aux):
    """Generates optimal sequence of (state, action) pairs for Tower of Hanoi."""
    current_state = np.full(n_disks, source, dtype=np.uint8)
    states = [current_state.copy()]
    actions = []

    def move(disk, from_peg, to_peg, aux_peg):
        if disk == 0:
            current_state[0] = to_peg
            states.append(current_state.copy())
            actions.append((0, to_peg))
            return

        move(disk - 1, from_peg, aux_peg, to_peg)
        current_state[disk] = to_peg
        states.append(current_state.copy())
        actions.append((disk, to_peg))
        move(disk - 1, aux_peg, to_peg, from_peg)

    if n_disks > 0:
        move(n_disks - 1, source, target, aux)

    return states, actions


def encode_state_and_action(state, action, target_peg, seq_len):
    """Encode state and next action into input/label format."""
    n_disks = len(state)

    # Input: [state..., target, PAD...]
    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    input_vec[:n_disks] = state
    input_vec[n_disks] = target_peg

    # Label: [disk_action, dest_peg, PAD..., target]
    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    if action is not None:
        disk_id, dest_peg = action
        label_vec[0] = ACTION_DISK_OFFSET + disk_id
        label_vec[1] = dest_peg
    # Put target at same position as input for consistency
    label_vec[n_disks] = target_peg

    return input_vec, label_vec


def encode_state_to_state(state_t, state_t1, target_peg, seq_len):
    """State-to-state encoding."""
    n_disks = len(state_t)

    # Input: [state_t..., target, PAD...]
    input_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    input_vec[:n_disks] = state_t
    input_vec[n_disks] = target_peg

    # Label: [state_t+1..., target, PAD...]
    label_vec = np.full(seq_len, PAD_ID, dtype=np.uint8)
    label_vec[:n_disks] = state_t1
    label_vec[n_disks] = target_peg

    return input_vec, label_vec


def convert_subset(
    split_name: str,
    disk_range: range,
    output_dir: str,
    encoding_type: str,
    max_disks_global: int,
    seq_len: int,
    vocab_size: int
):
    use_action_encoding = encoding_type == "action"
    
    # Initialize data structures
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],  # Cumulative count of examples
        "group_indices": [0]    # Cumulative count of puzzles
    }

    puzzle_id_counter = 0
    example_id_counter = 0

    print(f"Generating {split_name.upper()} set (Disks: {list(disk_range)})...")

    peg_ids = [PEG_A, PEG_B, PEG_C]
    peg_perms = list(itertools.permutations(peg_ids, 3))

    for n_disks in disk_range:
        for source, target, aux in tqdm(peg_perms, desc=f"  N={n_disks}", leave=False):
            states, actions = solve_hanoi_with_actions(n_disks, source, target, aux)
            
            # KEY FIX: Each step is a SEPARATE puzzle (like Sudoku)
            # This matches the Sudoku pattern: one input -> one output per puzzle
            for i in range(len(states) - 1):
                current_state = states[i]
                next_state = states[i + 1]
                action = actions[i]

                if use_action_encoding:
                    input_vec, label_vec = encode_state_and_action(
                        current_state, action, target, seq_len
                    )
                else:
                    input_vec, label_vec = encode_state_to_state(
                        current_state, next_state, target, seq_len
                    )

                results["inputs"].append(input_vec)
                results["labels"].append(label_vec)
                
                example_id_counter += 1
                puzzle_id_counter += 1
                
                # Each step is a separate puzzle
                results["puzzle_indices"].append(example_id_counter)
                results["puzzle_identifiers"].append(0)
            
            # Each Hanoi problem instance is a group
            results["group_indices"].append(puzzle_id_counter)

    # Convert to Numpy
    inputs_np = np.stack(results["inputs"])
    labels_np = np.stack(results["labels"])
    
    final_results = {
        "inputs": inputs_np.astype(np.int32),
        "labels": labels_np.astype(np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Save Directory
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save .npy files
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Metadata
    metadata = {
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "pad_id": int(PAD_ID),
        "ignore_label_id": int(PAD_ID),
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(final_results["group_indices"]) - 1,
        "mean_puzzle_examples": float(len(final_results["inputs"]) / (len(final_results["group_indices"]) - 1)),
        "total_puzzles": len(final_results["puzzle_indices"]) - 1,  # FIXED: Total individual puzzles
        "sets": ["all"]
    }

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  ✓ Saved {len(inputs_np)} examples to {save_dir}")
    print(f"    Total puzzles: {metadata['total_puzzles']}")
    print(f"    Total groups: {metadata['total_groups']}")


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
    max_disks_global = test_max
    vocab_size = get_vocab_size(max_disks_global, use_action_encoding)

    # FIXED: Sequence length should fit ONE step's I/O
    # For action encoding: [n_disks state + 1 target] for input
    #                      [2 action tokens + padding] for output
    # For state encoding:  [n_disks state + 1 target] for both
    seq_len = max_disks_global + 16  # Buffer for target and action tokens
    if seq_len % 16 != 0:
        seq_len = ((seq_len // 16) + 1) * 16

    print(f"Configuration: Enc={encoding_type}, SeqLen={seq_len}, Vocab={vocab_size}")
    print(f"  Max disks: {max_disks_global}")
    print(f"  Train range: {train_min}-{train_max}")
    print(f"  Test range: {test_min}-{test_max}")

    # Generate Train
    convert_subset(
        "train", range(train_min, train_max + 1), output_dir, 
        encoding_type, max_disks_global, seq_len, vocab_size
    )

    # Generate Test
    convert_subset(
        "test", range(test_min, test_max + 1), output_dir, 
        encoding_type, max_disks_global, seq_len, vocab_size
    )
    
    # Save identifiers
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
        
    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/hanoi")
    parser.add_argument("--encoding", type=str, choices=["action", "state"], default="action")
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