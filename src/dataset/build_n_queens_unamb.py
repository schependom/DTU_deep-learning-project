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
    """Finds solutions for N-Queens using backtracking."""

    def __init__(self, n: int):
        self.n = n
        self.solutions = []

    def solve(self, return_first=False):
        """Finds all valid full boards."""
        self.solutions = []
        board = [-1] * self.n
        self._place_queen(board, 0, count_only=False, limit=1 if return_first else None)
        return self.solutions

    def check_uniqueness(self, partial_grid: np.ndarray) -> bool:
        """
        Returns True if the partial_grid has exactly one solution.
        """
        self.solution_count = 0
        
        # Lock fixed positions from the partial grid
        fixed_queens = [-1] * self.n
        rows, cols = np.where(partial_grid == QUEEN)
        for r, c in zip(rows, cols):
            fixed_queens[r] = c
            
        board = [-1] * self.n
        self._place_queen_constrained(board, 0, fixed_queens)
        
        return self.solution_count == 1

    def _place_queen(self, board, row, count_only=False, limit=None):
        if limit and len(self.solutions) >= limit:
            return

        if row == self.n:
            self.solutions.append(self._board_to_grid(board))
            return

        for col in range(self.n):
            if self._is_safe(board, row, col):
                board[row] = col
                self._place_queen(board, row + 1, count_only, limit)
                board[row] = -1

    def _place_queen_constrained(self, board, row, fixed_queens):
        if self.solution_count > 1:
            return

        if row == self.n:
            self.solution_count += 1
            return

        if fixed_queens[row] != -1:
            col = fixed_queens[row]
            if self._is_safe(board, row, col):
                board[row] = col
                self._place_queen_constrained(board, row + 1, fixed_queens)
                board[row] = -1
        else:
            for col in range(self.n):
                if self._is_safe(board, row, col):
                    board[row] = col
                    self._place_queen_constrained(board, row + 1, fixed_queens)
                    board[row] = -1

    def _is_safe(self, board, row, col):
        for r in range(row):
            c = board[r]
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    def _board_to_grid(self, board_cols):
        grid = np.zeros((self.n, self.n), dtype=np.uint8)
        for r, c in enumerate(board_cols):
            grid[r, c] = QUEEN
        return grid


def create_unique_masked_problem(solution: np.ndarray, solver: NQueensSolver, min_mask=0.3, max_mask=0.7) -> Tuple[np.ndarray, bool]:
    n = solution.shape[0]
    problem = np.zeros_like(solution)
    rows, cols = np.where(solution == QUEEN)
    queen_coords = list(zip(rows, cols))
    num_queens = len(queen_coords)

    # Try up to 10 times to find a unique mask configuration
    for _ in range(10):
        mask_ratio = np.random.uniform(0.2, 0.7)
        num_to_keep = max(0, int(num_queens * (1 - mask_ratio)))
        
        indices = np.random.choice(num_queens, num_to_keep, replace=False)
        
        problem.fill(0)
        for idx in indices:
            r, c = queen_coords[idx]
            problem[r, c] = QUEEN
            
        if solver.check_uniqueness(problem):
            return problem, True

    return problem, False


def build_dataset(output_dir: str, n_size: int, num_aug: int, seed: int):
    np.random.seed(seed)

    print(f"Solving N-Queens for N={n_size}...")
    solver = NQueensSolver(n_size)
    solutions = solver.solve()
    print(f"Found {len(solutions)} unique base solutions.")

    np.random.shuffle(solutions)
    split_idx = int(len(solutions) * 0.9)
    train_sol = solutions[:split_idx]
    test_sol = solutions[split_idx:]

    process_subset("train", train_sol, output_dir, num_aug, solver)
    process_subset("test", test_sol, output_dir, 0, solver)


def process_subset(set_name: str, base_solutions: List[np.ndarray], output_dir: str, num_aug: int, solver: NQueensSolver):
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    iterations = 1 if num_aug == 0 else 3 
    discard_count = 0
    total_attempts = 0

    for sol in tqdm(base_solutions, desc=f"Processing {set_name}"):
        for _ in range(iterations):
            total_attempts += 1
            
            inp, success = create_unique_masked_problem(sol, solver)
            
            if not success:
                discard_count += 1
                continue 
            
            aug_range = 8 if num_aug > 0 else 1
            for aug_idx in range(aug_range):
                t_inp = dihedral_transform(inp, aug_idx)
                t_sol = dihedral_transform(sol, aug_idx)
                
                formatted_inp = t_inp.copy()
                formatted_inp[formatted_inp == 0] = 2 
                
                results["inputs"].append(formatted_inp)
                results["labels"].append(t_sol)

                example_id += 1
                puzzle_id += 1
                
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)

        if puzzle_id > results["group_indices"][-1]:
            results["group_indices"].append(puzzle_id)

    print(f"Dataset {set_name}: Generated {len(results['inputs'])} examples.")
    print(f"Uniqueness Check: Discarded {discard_count} ambiguous puzzles out of {total_attempts} attempts ({discard_count/total_attempts:.1%}).")

    def _seq_to_numpy(seq):
        # Input 2 -> Token 3 (Unknown)
        # Input 1 -> Token 2 (Queen)
        # Label 0 -> Token 1 (Empty)
        # Label 1 -> Token 2 (Queen)
        return np.array(seq, dtype=np.uint8).reshape(len(seq), -1) + 1

    final_results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    n_size = base_solutions[0].shape[0]
    metadata = PuzzleDatasetMetadata(
        seq_len=n_size * n_size,
        vocab_size=4, 
        pad_id=0, ignore_label_id=0, blank_identifier_id=0,
        num_puzzle_identifiers=1, total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1, total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_path = os.path.join(output_dir, set_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    for k, v in final_results.items():
        np.save(os.path.join(save_path, f"all__{k}.npy"), v)
    
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/n_queens_unique")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--aug", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.out, args.n, args.aug, args.seed)