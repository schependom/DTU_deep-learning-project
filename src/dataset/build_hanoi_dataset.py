import os
import json
import numpy as np
import itertools
from typing import List, Dict, Tuple
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# Import the metadata class from your common.py
# (Assuming common.py is in the same directory or Python path)
try:
    from common import PuzzleDatasetMetadata
except ImportError:
    print("Warning: common.py not found. Using dummy PuzzleDatasetMetadata.")

    class PuzzleDatasetMetadata(BaseModel):
        pad_id: int
        ignore_label_id: int
        blank_identifier_id: int
        vocab_size: int
        seq_len: int
        num_puzzle_identifiers: int
        total_groups: int
        mean_puzzle_examples: float
        total_puzzles: int
        sets: List[str]


cli = ArgParser()

# --- Token Definitions ---
# We will use a 1D vector to represent the state.
# The index `i` of the vector represents disk `i` (0-indexed, 0=smallest).
# The value at that index represents the peg.
PAD_ID = 0
PEG_A = 1
PEG_B = 2
PEG_C = 3
VOCAB_SIZE = 4  # 0: PAD, 1: Peg A, 2: Peg B, 3: Peg C


class DataProcessConfig(BaseModel):
    output_dir: str = "data/hanoi"
    # --- This is the key to testing generalization ---
    # Train on 3, 4, 5 disks
    min_disks_train: int = 3
    max_disks_train: int = 5
    # Test on 6, 7 disks (which the model has never seen)
    min_disks_test: int = 6
    max_disks_test: int = 8
    # -------------------------------------------------
    seed: int = 42


def generate_hanoi_states(
    n_disks: int, source_peg: int, target_peg: int, aux_peg: int, max_seq_len: int
) -> List[np.ndarray]:
    """
    Generates a list of all board states for an optimal ToH solution.
    """

    # The state is a vector of length max_seq_len
    # state[i] = location of disk i (0-indexed, 0 is smallest)
    state = np.full(max_seq_len, PAD_ID, dtype=np.uint8)
    # Initialize all N disks to be on the source peg
    state[:n_disks] = source_peg

    all_states = [np.copy(state)]

    def solve_recursive(n: int, s: int, t: int, a: int):
        """
        Inner recursive function to modify the state.
        n = number of disks to move
        s = source, t = target, a = auxiliary
        """
        if n > 0:
            # We are moving the (n-1)-th disk (0-indexed)
            disk_index = n - 1

            # 1. Move n-1 disks from source to auxiliary
            solve_recursive(n - 1, s, a, t)

            # 2. Move disk n-1 from source to target
            state[disk_index] = t
            all_states.append(np.copy(state))

            # 3. Move n-1 disks from auxiliary to target
            solve_recursive(n - 1, a, t, s)

    # Start the recursive process
    solve_recursive(n_disks, source_peg, target_peg, aux_peg)

    return all_states


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # The "augmentations" for ToH are just the 6 permutations
    # of (source, target, auxiliary) peg assignments.
    peg_map = {"A": PEG_A, "B": PEG_B, "C": PEG_C}
    peg_names = ["A", "B", "C"]
    peg_permutations = list(itertools.permutations(peg_names, 3))

    # seq_len is fixed to the max disks we'll ever test
    max_seq_len = config.max_disks_test

    # Map puzzle string IDs to integer IDs
    print("Pre-calculating all puzzle identifiers...")
    identifier_map = {}
    num_identifiers = 1  # 0 is blank

    all_disk_ranges = {
        "train": range(config.min_disks_train, config.max_disks_train + 1),
        "test": range(config.min_disks_test, config.max_disks_test + 1),
    }

    for n_disks in list(all_disk_ranges["train"]) + list(all_disk_ranges["test"]):
        for s_name, t_name, a_name in peg_permutations:
            puzzle_id_str = f"hanoi_N{n_disks}_{s_name}_to_{t_name}"
            if puzzle_id_str not in identifier_map:
                identifier_map[puzzle_id_str] = num_identifiers
                num_identifiers += 1

    # This is the total count that must be shared by all metadata files
    final_num_identifiers = num_identifiers
    print(f"Total puzzle identifiers found: {final_num_identifiers - 1}")

    # --- Step 2 ---
    # Now, generate the data for each split

    for split_name in ["train", "test"]:
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Prepare data containers, following the ARC/Sudoku format
        results: Dict[str, List] = {
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
        print(f"Generating {split_name.upper()} split...")

        if split_name == "train":
            disk_range = range(config.min_disks_train, config.max_disks_train + 1)
            print(
                f"Generating TRAIN split (disks {config.min_disks_train}-{config.max_disks_train})..."
            )
        else:
            disk_range = range(config.min_disks_test, config.max_disks_test + 1)
            print(
                f"Generating TEST split (disks {config.min_disks_test}-{config.max_disks_test})..."
            )

        # A "group" is one N-disk setting (e.g., all 4-disk puzzles)
        for n_disks in disk_range:
            # A "puzzle" is one specific permutation (e.r., N=4, A->C, B-aux)
            for s_name, t_name, a_name in tqdm(
                peg_permutations, desc=f"N={n_disks} disks"
            ):
                # Get a unique integer ID for this puzzle configuration
                # Get the integer ID from the pre-built map
                puzzle_id_str = f"hanoi_N{n_disks}_{s_name}_to_{t_name}"
                puzzle_identifier_int = identifier_map[puzzle_id_str]  # Use the map

                if puzzle_id_str not in identifier_map:
                    identifier_map[puzzle_id_str] = num_identifiers
                    num_identifiers += 1

                puzzle_identifier_int = identifier_map[puzzle_id_str]

                s_peg, t_peg, a_peg = peg_map[s_name], peg_map[t_name], peg_map[a_name]

                # Generate all 2^n states for this puzzle
                states = generate_hanoi_states(
                    n_disks, s_peg, t_peg, a_peg, max_seq_len
                )

                # Create the (state, next_state) examples
                # This puzzle will have (2^n_disks - 1) examples
                for i in range(len(states) - 1):
                    results["inputs"].append(states[i])
                    results["labels"].append(states[i + 1])
                    example_id += 1

                # --- Update indices (like build_arc_dataset.py) ---
                # After all examples for *this puzzle* are added
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(puzzle_identifier_int)
                puzzle_id += 1

            # After all puzzles (permutations) for *this group* (n_disks) are added
            results["group_indices"].append(puzzle_id)

        # --- Save to .npy files ---
        total_examples = len(results["inputs"])
        total_puzzles = len(results["puzzle_identifiers"])
        total_groups = len(results["group_indices"]) - 1

        print(f"Split '{split_name}':")
        print(f"  Total examples (state transitions): {total_examples}")
        print(f"  Total puzzles (N, s, t configs): {total_puzzles}")
        print(f"  Total groups (N-disk settings): {total_groups}")

        for k, v in results.items():
            if k in {"inputs", "labels"}:
                v_np = np.stack(v, 0)
            else:
                v_np = np.array(v, dtype=np.int32)

            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v_np)

        # --- Save metadata ---
        metadata = PuzzleDatasetMetadata(
            seq_len=max_seq_len,
            vocab_size=VOCAB_SIZE,
            pad_id=PAD_ID,
            ignore_label_id=PAD_ID,
            blank_identifier_id=0,
            # Use the FINAL total count, not the partial one
            num_puzzle_identifiers=final_num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles
            if total_puzzles > 0
            else 0,
            total_puzzles=total_puzzles,
            sets=["all"],
        )

        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f, indent=2)

    # Save the global ID mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump(
            [ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f, indent=2
        )

    print("\nDataset generation complete.")
    print(f"Total unique puzzle identifiers: {num_identifiers - 1}")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
