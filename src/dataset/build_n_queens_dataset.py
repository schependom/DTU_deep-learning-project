import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Tuple

from common import PuzzleDatasetMetadata, dihedral_transform

# Constants
EMPTY = 0
QUEEN = 1


class NQueensSolver:
    """Finds all solutions for N-Queens using backtracking."""

    def __init__(self, n: int):
        self.n = n
        self.solutions = []

    def solve(self):
        board = [-1] * self.n
        self._place_queen(board, 0)
        return self.solutions

    def _place_queen(self, board, row):
        if row == self.n:
            self.solutions.append(self._board_to_grid(board))
            return

        for col in range(self.n):
            if self._is_safe(board, row, col):
                board[row] = col
                self._place_queen(board, row + 1)
                board[row] = -1

    def _is_safe(self, board, row, col):
        for r in range(row):
            c = board[r]
            # Check column and diagonals
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    def _board_to_grid(self, board_cols):
        grid = np.zeros((self.n, self.n), dtype=np.uint8)
        for r, c in enumerate(board_cols):
            grid[r, c] = QUEEN
        return grid


def create_masked_problem(solution: np.ndarray, mask_ratio: float = 0.5) -> np.ndarray:
    """
    Creates a problem instance by keeping only some queens as hints.
    mask_ratio: approximate percentage of queens to remove (hide).
    """
    n = solution.shape[0]
    problem = np.zeros_like(solution)

    # Get coordinates of all queens
    rows, cols = np.where(solution == QUEEN)
    queen_coords = list(zip(rows, cols))

    # Determine how many to keep (at least 1 to anchor the solution if possible, though 0 is valid hard mode)
    num_queens = len(queen_coords)
    num_to_keep = max(0, int(num_queens * (1 - mask_ratio)))

    # Randomly select queens to keep
    indices = np.random.choice(num_queens, num_to_keep, replace=False)

    for idx in indices:
        r, c = queen_coords[idx]
        problem[r, c] = QUEEN

    return problem


def build_dataset(
    output_dir: str, n_size: int, num_aug: int, seed: int, split_ratio: float = 0.9
):
    np.random.seed(seed)

    print(f"Solving N-Queens for N={n_size}...")
    solver = NQueensSolver(n_size)
    solutions = solver.solve()
    print(f"Found {len(solutions)} unique solutions.")

    if len(solutions) == 0:
        raise ValueError(f"No solutions found for N={n_size}. This shouldn't happen.")

    # Shuffle solutions
    np.random.shuffle(solutions)

    # Split into Train/Test sets (Split by distinct solutions to test generalization)
    split_idx = int(len(solutions) * split_ratio)
    train_solutions = solutions[:split_idx]
    test_solutions = solutions[split_idx:]

    print(
        f"Split: {len(train_solutions)} Train, {len(test_solutions)} Test base solutions."
    )

    process_subset("train", train_solutions, output_dir, num_aug, n_size)
    process_subset(
        "test", test_solutions, output_dir, 0, n_size
    )  # No augmentation for test usually, or minimal


def process_subset(
    set_name: str,
    base_solutions: List[np.ndarray],
    output_dir: str,
    num_aug: int,
    n_size: int,
):
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

    # We will generate multiple "masked" versions per solution to increase dataset size
    # For training: Generate `num_aug` variants per solution
    # If num_aug is 0 (test), we just do 1 version

    iterations = (
        1 if num_aug == 0 else 5
    )  # Generate 5 different mask patterns per solution for training

    for sol in tqdm(base_solutions, desc=f"Processing {set_name}"):
        # For each base solution, generate a few random problems (masking)
        for _ in range(iterations):
            # Randomly choose how hard the puzzle is (how many queens hidden)
            # 0.2 (easy, mostly filled) to 1.0 (empty board)
            mask_ratio = np.random.uniform(0.2, 1.0)
            inp = create_masked_problem(sol, mask_ratio)

            # Augmentations (Dihedral symmetries)
            # We apply the symmetry to BOTH input and output
            # Since N-Queens is symmetric, the transformed board is also a valid N-Queens state
            aug_range = 8 if num_aug > 0 else 1

            for aug_idx in range(aug_range):
                aug_inp = dihedral_transform(inp, aug_idx)
                aug_sol = dihedral_transform(sol, aug_idx)

                results["inputs"].append(aug_inp)
                results["labels"].append(aug_sol)

                example_id += 1
                puzzle_id += 1

                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)

            # Push group (grouping augmentations together)
            results["group_indices"].append(puzzle_id)

    # Flatten and save
    def _seq_to_numpy(seq):
        # Add +1 to values so 0 can be PAD, 1=Empty(if distinct), 2=Queen
        # Actually, standard TRM Sudoku uses: 0=PAD, 1..9=Digits.
        # Here: 0=PAD.
        # Let's map: Empty(0) -> 1, Queen(1) -> 2.
        # Input/Label will be in range [1, 2].
        # 0 is reserved for Padding in the collator.
        arr = np.array(seq, dtype=np.uint8).reshape(len(seq), -1)
        return arr + 1

    final_results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=n_size * n_size,
        vocab_size=3,  # PAD(0), Empty(1), Queen(2)
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_path = os.path.join(output_dir, set_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    for k, v in final_results.items():
        np.save(os.path.join(save_path, f"all__{k}.npy"), v)

    # Save identifiers mapping
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/n_queens_12")
    parser.add_argument("--n", type=int, default=12, help="Board size NxN")
    parser.add_argument(
        "--aug", type=int, default=1, help="Enable augmentations (1=yes, 0=no)"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_dataset(args.out, args.n, args.aug, args.seed)
