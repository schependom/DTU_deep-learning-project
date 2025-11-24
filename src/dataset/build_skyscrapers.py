import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from itertools import permutations

from common import PuzzleDatasetMetadata, dihedral_transform

def calculate_visibility(line):
    """Calculates how many skyscrapers are visible in a line."""
    max_height = 0
    visible_count = 0
    for height in line:
        if height > max_height:
            visible_count += 1
            max_height = height
    return visible_count

def solve_skyscrapers_backtrack(grid, clues, n):
    """
    Optimized solver with incremental visibility checking.
    """
    solutions = []
    
    # Pre-compute fixed values to speed up lookups
    fixed = grid[1:n+1, 1:n+1].copy()

    def check_line_visibility(line, target):
        if target == 0: return True
        count = 0
        max_h = 0
        for h in line:
            if h > max_h:
                count += 1
                max_h = h
        return count == target

    def is_row_valid_vis(r):
        # Row data: grid[r+1, 1..N]
        row_data = grid[r+1, 1:n+1]
        
        # Left clue: grid[r+1, 0]
        if not check_line_visibility(row_data, grid[r+1, 0]): return False
        
        # Right clue: grid[r+1, n+1]
        if not check_line_visibility(row_data[::-1], grid[r+1, n+1]): return False
        return True

    def is_col_valid_vis(c):
        # Col data: grid[1..N, c+1]
        col_data = grid[1:n+1, c+1]
        
        # Top clue: grid[0, c+1]
        if not check_line_visibility(col_data, grid[0, c+1]): return False
        
        # Bottom clue: grid[n+1, c+1]
        if not check_line_visibility(col_data[::-1], grid[n+1, c+1]): return False
        return True

    def backtrack(idx):
        if len(solutions) > 1: return

        if idx == n * n:
            solutions.append(grid.copy())
            return

        row, col = divmod(idx, n)

        # Decide value to try (Fixed or Loop 1..N)
        if fixed[row, col] != 0:
            candidates = [fixed[row, col]]
            is_fixed_step = True
        else:
            candidates = range(1, n + 1)
            is_fixed_step = False

        for val in candidates:
            # 1. Standard Latin Square Check
            # Row check
            if val in grid[row+1, 1:col+1]: continue # Check previous cells in row
            # Forward row check is hard, just standard check:
            # Simpler: check if val exists in current row/col (ignoring self if fixed)
            is_dup = False
            
            # Check Row (optimized: only check filled parts or full scan if simpler)
            # Full scan is safer for fixed steps
            for k in range(n):
                if k != col and grid[row+1, k+1] == val: 
                    is_dup = True; break
            if is_dup: continue
                
            # Check Col
            for k in range(n):
                if k != row and grid[k+1, col+1] == val: 
                    is_dup = True; break
            if is_dup: continue

            # Place value
            grid[row+1, col+1] = val

            # 2. INCREMENTAL PRUNING (The Speedup)
            valid_partial = True
            
            # If we just finished a ROW (col == n-1), check row clues
            if col == n - 1:
                if not is_row_valid_vis(row):
                    valid_partial = False

            # If we just finished a COL (row == n-1), check col clues
            # Note: Because we fill row-by-row, we only finish a column when we are at the very bottom row.
            if valid_partial and row == n - 1:
                if not is_col_valid_vis(col):
                    valid_partial = False
            
            if valid_partial:
                backtrack(idx + 1)

            # Cleanup (only if not fixed)
            if not is_fixed_step:
                grid[row+1, col+1] = 0

    backtrack(0)
    return solutions

def generate_board(n):
    """Generates a random valid full board and clues."""
    # Start with a random Latin Square (approximate generation)
    # Correct random LS generation is hard, using simple shuffle swap
    inner = np.zeros((n, n), dtype=int)
    first_row = np.arange(1, n+1)
    np.random.shuffle(first_row)
    for i in range(n):
        inner[i] = np.roll(first_row, i)
    
    # Shuffle rows and cols to make it random
    np.random.shuffle(inner)
    inner = inner[:, np.random.permutation(n)]
    
    # Build padded grid
    full_grid = np.zeros((n+2, n+2), dtype=int)
    full_grid[1:n+1, 1:n+1] = inner
    
    # Calculate clues
    # Top
    for c in range(n): full_grid[0, c+1] = calculate_visibility(inner[:, c])
    # Bottom
    for c in range(n): full_grid[n+1, c+1] = calculate_visibility(inner[:, c][::-1])
    # Left
    for r in range(n): full_grid[r+1, 0] = calculate_visibility(inner[r, :])
    # Right
    for r in range(n): full_grid[r+1, n+1] = calculate_visibility(inner[r, :][::-1])
    
    return full_grid

def mask_board(full_grid, n, keep_ratio=0.4):
    """Removes internal cells, keeps clues."""
    problem = full_grid.copy()
    
    # Number of internal cells
    total_cells = n * n
    num_keep = int(total_cells * keep_ratio)
    
    # Indices to keep
    indices = np.random.choice(total_cells, num_keep, replace=False)
    mask = np.ones(total_cells, dtype=bool)
    mask[indices] = False
    
    # Apply mask to inner grid
    flat_inner = problem[1:n+1, 1:n+1].flatten()
    flat_inner[mask] = 0
    problem[1:n+1, 1:n+1] = flat_inner.reshape(n, n)
    
    return problem

def build_dataset(output_dir: str, n_size: int, num_samples: int, seed: int):
    np.random.seed(seed)
    
    sets = {"train": num_samples, "test": int(num_samples * 0.1)}
    
    # Vocab: 0 (pad/empty), 1..N (heights/clues)
    # Output format is flat array of (N+2)*(N+2)
    
    for set_name, count in sets.items():
        results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
        puzzle_id = 0
        example_id = 0
        results["puzzle_indices"].append(0)
        results["group_indices"].append(0)
        
        print(f"Generating {set_name} set...")
        
        valid_count = 0
        pbar = tqdm(total=count)
        
        while valid_count < count:
            # 1. Generate full board
            solution = generate_board(n_size)
            
            # 2. Mask it (keep clues + some internal numbers)
            # Hard puzzles have fewer internal numbers.
            puzzle = mask_board(solution, n_size, keep_ratio=np.random.uniform(0.0, 0.3))
            
            # 3. Check Uniqueness
            # We pass a copy because solver modifies in-place
            sols = solve_skyscrapers_backtrack(puzzle.copy(), None, n_size)
            
            if len(sols) == 1:
                # 4. Dihedral Augmentation
                aug_range = 8 if set_name == "train" else 1
                for aug_idx in range(aug_range):
                    # Rotate/Flip the whole (N+2)x(N+2) grid
                    # Clues rotate naturally with the grid
                    p_aug = dihedral_transform(puzzle, aug_idx)
                    s_aug = dihedral_transform(solution, aug_idx)
                    
                    results["inputs"].append(p_aug)
                    results["labels"].append(s_aug)
                    
                    example_id += 1
                    puzzle_id += 1
                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(0)
                
                results["group_indices"].append(puzzle_id)
                valid_count += 1
                pbar.update(1)
            
        pbar.close()
        
        # Save
        def _seq_to_numpy(seq):
            return np.array(seq, dtype=np.uint8).reshape(len(seq), -1)

        final_results = {
            "inputs": _seq_to_numpy(results["inputs"]),
            "labels": _seq_to_numpy(results["labels"]),
            "group_indices": np.array(results["group_indices"], dtype=np.int32),
            "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
            "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        }

        metadata = PuzzleDatasetMetadata(
            seq_len=(n_size+2)**2,
            vocab_size=n_size + 1,
            pad_id=0, ignore_label_id=0, blank_identifier_id=0,
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
            
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/skyscrapers_6")
    parser.add_argument("--n", type=int, default=6) # 6 is a good difficulty balance
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.out, args.n, args.samples, args.seed)