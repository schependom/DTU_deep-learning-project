from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "LangAGI-Lab/Sudoku-Easy"
    output_dir: str = "data/easy-sudoku"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..4, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 5)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(2)
    row_perm = np.concatenate([b * 2 + np.random.permutation(2) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(2)
    col_perm = np.concatenate([s * 2 + np.random.permutation(2) for s in stacks])

    # Build an 16->16 mapping. For each new cell at (i, j)
    # (row index = i // 4, col index = i % 4),
    # its value comes from old row = row_perm[i//4] and old col = col_perm[i%4].
    mapping = np.array([row_perm[i // 4] * 4 + col_perm[i % 4] for i in range(16)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(4, 4).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # --- THIS FUNCTION HAS BEEN UPDATED ---

    # 1. Define the correct filenames from Hugging Face
    filename_map = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }

    if set_name not in filename_map:
        print(f"Unknown set_name: {set_name}. Skipping.")
        return

    filename = filename_map[set_name]
    print(f"Downloading and reading {filename} from {config.source_repo}...")

    # 2. Download and read the Parquet file using pandas
    # We construct the hf:// URL to stream-read the file
    hf_url = f"hf://datasets/{config.source_repo}/{filename}"

    try:
        df = pd.read_parquet(hf_url)
    except Exception as e:
        print(
            f"Error reading parquet file. Make sure you are logged in (`huggingface-cli login`)"
        )
        print(f"Also ensure you have 'pandas' and 'pyarrow' installed.")
        raise e

    print(f"Successfully loaded {len(df)} puzzles for '{set_name}'.")

    # 3. Process the DataFrame
    inputs = []
    labels = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # 'row' will have columns 'puzzle' (the unsolved) and 'solution' (the solved)
        # We also filter by difficulty if specified
        if (config.min_difficulty is None) or (
            row.get("difficulty", 0) >= config.min_difficulty
        ):
            q = row["initial_board"]
            a = row["solution"]

            assert len(q) == 16 and len(a) == 16  # 4x4 Sudoku

            # Convert string "1.2.3..." to numpy array [1, 0, 2, 0, 3...]
            inputs.append(
                np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(
                    4, 4
                )
                - ord("0")
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(4, 4) - ord("0")
            )

    # --- THE REST OF THE FUNCTION IS THE SAME AS BEFORE ---

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            print(
                f"Subsampling from {total_samples} to {config.subsample_size} examples."
            )
            indices = np.random.choice(
                total_samples, size=config.subsample_size, replace=False
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0
    print(f"Applying {num_augments} augmentations for '{set_name}' set...")

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
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        # --- REMEMBER THIS FIX ---
        # We removed the "+ 1" here to keep blanks as 0.
        return arr

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=16,
        vocab_size=4 + 1,  # PAD + "0" ... "3"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1 + num_augments,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving processed data to {save_dir}...")

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"Finished processing '{set_name}'.")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
