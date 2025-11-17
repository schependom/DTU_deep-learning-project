"""
Predict the next state given the current state in Tower of Hanoi.
"""

import os
import json
import numpy as np
import itertools
from typing import List, Dict
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

from common import PuzzleDatasetMetadata

cli = ArgParser()

# --- Token Definitions ---
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
# We add a buffer for the goal token in the vocab
# 0=PAD, 1=A, 2=B, 3=C.
VOCAB_SIZE = 4


class DataProcessConfig(BaseModel):
    output_dir: str = "data/hanoi"
    min_disks_train: int = 3
    max_disks_train: int = 6
    min_disks_test: int = 7
    max_disks_test: int = 10
    seed: int = 42


def generate_hanoi_states(
    n_disks: int, source_peg: int, target_peg: int, aux_peg: int, max_seq_len: int
) -> List[np.ndarray]:
    """
    Generates a list of all board states for an optimal ToH solution.
    State is padded to max_seq_len.
    """
    # state[i] = peg location of disk i (0=smallest)
    state = np.full(max_seq_len, PAD_ID, dtype=np.uint8)
    state[:n_disks] = source_peg

    all_states = [np.copy(state)]

    def solve_recursive(n: int, s: int, t: int, a: int):
        if n > 0:
            # Move n-1 from source to auxiliary
            solve_recursive(n - 1, s, a, t)

            # Move disk n-1 from source to target
            state[n - 1] = t
            all_states.append(np.copy(state))

            # Move n-1 from auxiliary to target
            solve_recursive(n - 1, a, t, s)

    solve_recursive(n_disks, source_peg, target_peg, aux_peg)
    return all_states


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    peg_map = {"A": PEG_A, "B": PEG_B, "C": PEG_C}
    peg_names = ["A", "B", "C"]
    peg_permutations = list(itertools.permutations(peg_names, 3))

    # FIX 1: Increase seq_len by 1 to accommodate the "Target Peg" token
    max_seq_len_disks = config.max_disks_test
    input_seq_len = max_seq_len_disks + 1

    all_disk_ranges = {
        "train": range(config.min_disks_train, config.max_disks_train + 1),
        "test": range(config.min_disks_test, config.max_disks_test + 1),
    }

    # Pre-calculate identifiers
    identifier_map = {}
    num_identifiers = 1
    for n_disks in list(all_disk_ranges["train"]) + list(all_disk_ranges["test"]):
        for s_name, t_name, a_name in peg_permutations:
            puzzle_id_str = f"hanoi_N{n_disks}_{s_name}_to_{t_name}"
            if puzzle_id_str not in identifier_map:
                identifier_map[puzzle_id_str] = num_identifiers
                num_identifiers += 1

    print(f"Total puzzle identifiers: {num_identifiers - 1}")

    for split_name in ["train", "test"]:
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        results = {
            k: []
            for k in [
                "inputs",
                "labels",
                "puzzle_identifiers",
                "puzzle_indices",
                "group_indices",
            ]
        }

        example_id = 0
        puzzle_id = 0
        results["puzzle_indices"].append(0)
        results["group_indices"].append(0)

        disk_range = all_disk_ranges[split_name]
        print(f"Generating {split_name.upper()} split for disks {list(disk_range)}...")

        for n_disks in disk_range:
            for s_name, t_name, a_name in tqdm(peg_permutations, desc=f"N={n_disks}"):
                puzzle_id_str = f"hanoi_N{n_disks}_{s_name}_to_{t_name}"
                puzzle_identifier_int = identifier_map[puzzle_id_str]

                s_peg, t_peg, a_peg = peg_map[s_name], peg_map[t_name], peg_map[a_name]

                # Generate states (length = max_seq_len_disks)
                states = generate_hanoi_states(
                    n_disks, s_peg, t_peg, a_peg, max_seq_len_disks
                )

                for i in range(len(states) - 1):
                    # FIX 2: Append the TARGET PEG to the input
                    # Input: [Disk0, Disk1, ... DiskN, PAD, ..., TARGET_PEG]
                    current_state = states[i]
                    target_token = np.array([t_peg], dtype=np.uint8)

                    # We append target at the very end of the sequence
                    input_vec = np.concatenate([current_state, target_token])

                    # Label: The next state (we don't strictly need the target in the label,
                    # but keeping shapes consistent is usually easier.
                    # Here I'll keep label as just the disk state for purity).
                    # NOTE: If your model expects input_len == output_len, append target to label too.
                    # Assuming typical AR model, we usually predict the full sequence.
                    # Let's append target to label to make it autoregressive-friendly.
                    label_vec = np.concatenate([states[i + 1], target_token])

                    results["inputs"].append(input_vec)
                    results["labels"].append(label_vec)
                    example_id += 1

                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(puzzle_identifier_int)
                puzzle_id += 1

            results["group_indices"].append(puzzle_id)

        # Save
        for k, v in results.items():
            if k in {"inputs", "labels"}:
                v_np = np.stack(v, 0)
            else:
                v_np = np.array(v, dtype=np.int32)
            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v_np)

        metadata = PuzzleDatasetMetadata(
            seq_len=input_seq_len,  # Updated length
            vocab_size=VOCAB_SIZE,
            pad_id=PAD_ID,
            ignore_label_id=PAD_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(results["group_indices"]) - 1,
            mean_puzzle_examples=len(results["inputs"])
            / len(results["puzzle_identifiers"])
            if len(results["puzzle_identifiers"]) > 0
            else 0,
            total_puzzles=len(results["puzzle_identifiers"]),
            sets=["all"],
        )

        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f, indent=2)

    print("Generation complete.")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
