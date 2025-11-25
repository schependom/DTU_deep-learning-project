import os
import json
import random
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from common import PuzzleDatasetMetadata

cli = ArgParser()

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# Vocabulary: Digits + Operators + Equals + Pad
# PAD is index 0.
CHARSET = "0123456789+-*%()="
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARSET)}
ID2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

class DataProcessConfig(BaseModel):
    output_dir: str = "data/math-arithmetic" # Updated to match your workflow
    
    # Dataset Size
    train_size: int = 100000
    test_size: int = 10000
    
    # Sequence Parameters
    seq_len: int = 48        # Increased slightly to accommodate deeper expressions
    max_depth: int = 4       # Increased depth for difficulty
    max_int: int = 20        # Max value for leaf integers
    
    seed: int = 42

# -----------------------------------------------------------------------------
# Math Generation Logic
# -----------------------------------------------------------------------------

def generate_expression(depth, max_depth, max_int, rng):
    """
    Recursively generates a math expression string and its value.
    """
    # Base case: Return an integer
    if depth >= max_depth or rng.random() < 0.1:
        val = rng.randint(1, max_int)
        return str(val), val

    # Recursive step: Choose an operator
    op = rng.choice(['+', '-', '*', '%'])
    
    left_str, left_val = generate_expression(depth + 1, max_depth, max_int, rng)
    right_str, right_val = generate_expression(depth + 1, max_depth, max_int, rng)

    # Avoid division by zero or modulo zero scenarios
    if op == '%' and right_val == 0:
        right_val = 1
        right_str = "1"

    # Construct string
    expr_str = f"({left_str}{op}{right_str})"
    
    # Calculate value
    if op == '+':
        val = left_val + right_val
    elif op == '-':
        val = left_val - right_val
    elif op == '*':
        val = left_val * right_val
    elif op == '%':
        val = left_val % right_val

    return expr_str, val

# -----------------------------------------------------------------------------
# Data Conversion
# -----------------------------------------------------------------------------

def tokenize(text: str, length: int) -> np.ndarray:
    """Converts string to padded numpy array of IDs."""
    ids = [CHAR2ID[c] for c in text]
    # Check if sequence is too long, return None if so
    if len(ids) > length:
        return None
        
    arr = np.zeros(length, dtype=np.uint8) # Pad with 0
    arr[:len(ids)] = ids
    return arr

def convert_subset(set_name: str, num_samples: int, config: DataProcessConfig):
    print(f"Generating {set_name} set with {num_samples} samples...")
    
    # Seeding per subset to ensure reproducibility
    rng = random.Random(config.seed + (1 if set_name == 'test' else 0))

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_indices": [0],
        "group_indices": [0],
        "puzzle_identifiers": []
    }

    puzzle_id = 0
    example_id = 0

    pbar = tqdm(total=num_samples)
    while len(results["inputs"]) < num_samples:
        # Generate Math: "((12+4)*3)=" and val 48
        expr_part, val = generate_expression(0, config.max_depth, config.max_int, rng)
        
        # FULL STRING: "((12+4)*3)=48"
        full_str = f"{expr_part}={val}"
        
        # INPUT STRING: "((12+4)*3)="
        input_str = f"{expr_part}=" 
        
        # Tokenize Full String first (Label candidate)
        full_arr = tokenize(full_str, config.seq_len)
        
        # Skip if too long
        if full_arr is None:
            continue

        # 1. Create Input: Mask everything AFTER the "=" (The Answer)
        inp_arr = full_arr.copy()
        cutoff = len(input_str)
        inp_arr[cutoff:] = 0 
        
        # 2. Create Label: Mask everything BEFORE the Answer (The Question)
        # This fixes the gradient issues. We only calculate loss on the answer digits.
        lbl_arr = full_arr.copy()
        lbl_arr[:cutoff] = 0 

        results["inputs"].append(inp_arr)
        results["labels"].append(lbl_arr) 
        
        # --- INDEXING FIX START (Matched to build_maze_dataset.py) ---
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
        # --- INDEXING FIX END ---
        
        pbar.update(1)
    
    pbar.close()

    # Convert to Numpy
    results["inputs"] = np.stack(results["inputs"])
    results["labels"] = np.stack(results["labels"])
    results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)
    results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=len(CHARSET) + 1, # +1 for PAD
        pad_id=0,
        ignore_label_id=0, # Matches the 0s we put in the label array
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save Data & Metadata
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save Identifiers (Dummy for visualization)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
        
    # Save Vocabulary for reference
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(CHAR2ID, f)

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_subset("train", config.train_size, config)
    convert_subset("test", config.test_size, config)

if __name__ == "__main__":
    cli()